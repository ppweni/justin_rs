//! Location heads (`justin/model/location.py`): point at one (channel, patch) of the series, or at
//! "not present" (always the last class).

use candle_core::{D, Tensor};
use serde::Deserialize;

use crate::error::{Result, format_error};
use crate::nn::{LayerNorm, Linear, MultiheadAttention, additive_mask, gelu};
use crate::weights::Weights;

/// The `head` metadata of a location export.
#[derive(Debug, Clone, Deserialize)]
pub struct LocationSpec {
    #[serde(rename = "type")]
    pub kind: String,
    pub signal_dim: usize,
    pub text_dim: usize,
    pub hidden_dim: usize,
    pub patch_size: usize,
    #[serde(default)]
    pub layers: usize,
    #[serde(default)]
    pub heads: usize,
    #[serde(default)]
    pub kernel: usize,
    #[serde(default)]
    pub max_tokens: usize,
}

/// `LocationHead`: one dot product between the pooled question and each patch.
pub struct PointerHead {
    query: Linear,
    key: Linear,
    null_key: Tensor,
    scale: f64,
}

/// `LocationAttentionHead`: cross-attention to the question, a local convolution along time and
/// self-attention over all patches and a null token.
pub struct AttentionHead {
    signal: Linear,
    question: Linear,
    position: Tensor,
    null_token: Tensor,
    blocks: Vec<AttentionBlock>,
    out_norm: LayerNorm,
    patch_logit: Linear,
    null_logit: Linear,
    max_tokens: usize,
}

struct AttentionBlock {
    norms: Vec<LayerNorm>,
    question: MultiheadAttention,
    conv_weight: Tensor, // [dim, kernel]
    conv_bias: Tensor,   // [dim]
    mixing: MultiheadAttention,
    mlp: (Linear, Linear),
}

pub enum LocationModel {
    Pointer(PointerHead),
    Attention(AttentionHead),
}

impl LocationModel {
    pub fn load(spec: &LocationSpec, weights: &Weights) -> Result<Self> {
        match spec.kind.as_str() {
            "pointer" => Ok(Self::Pointer(PointerHead {
                query: Linear::load(weights, "query_projection", true)?,
                key: Linear::load(weights, "key_projection", true)?,
                null_key: weights.get("null_key")?,
                scale: (spec.hidden_dim as f64).powf(-0.5),
            })),
            "attn" => {
                let blocks = (0..spec.layers)
                    .map(|i| {
                        let p = format!("blocks.{i}");
                        let weight = weights.get(&format!("{p}.local.weight"))?; // [dim, 1, kernel]
                        let (dim, _, kernel) = weight.dims3()?;
                        Ok(AttentionBlock {
                            norms: (0..4)
                                .map(|j| LayerNorm::load(weights, &format!("{p}.norms.{j}"), 1e-5))
                                .collect::<Result<_>>()?,
                            question: MultiheadAttention::load(
                                weights,
                                &format!("{p}.question"),
                                spec.heads,
                            )?,
                            conv_weight: weight.reshape((dim, kernel))?,
                            conv_bias: weights.get(&format!("{p}.local.bias"))?,
                            mixing: MultiheadAttention::load(
                                weights,
                                &format!("{p}.mixing"),
                                spec.heads,
                            )?,
                            mlp: (
                                Linear::load(weights, &format!("{p}.mlp.0"), true)?,
                                Linear::load(weights, &format!("{p}.mlp.3"), true)?,
                            ),
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(Self::Attention(AttentionHead {
                    signal: Linear::load(weights, "signal_projection", true)?,
                    question: Linear::load(weights, "question_projection", true)?,
                    position: weights.get("position.weight")?,
                    null_token: weights.get("null_token")?,
                    blocks,
                    out_norm: LayerNorm::load(weights, "out_norm", 1e-5)?,
                    patch_logit: Linear::load(weights, "patch_logit", true)?,
                    null_logit: Linear::load(weights, "null_logit", true)?,
                    max_tokens: spec.max_tokens,
                }))
            }
            other => Err(format_error(format!("unknown location head type: {other}"))),
        }
    }

    /// Logits [B, M + 1] from signal [B, M, d_s] with `valid` [B, M] (u8/u32, 1 = selectable),
    /// question tokens [B, L, d_t] with `question_valid` [B, L]; `tokens` patches per channel.
    pub fn forward(
        &self,
        signal: &Tensor,
        valid: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
        tokens: usize,
    ) -> Result<Tensor> {
        let keep = valid.to_dtype(signal.dtype())?;
        let keep_q = question_valid.to_dtype(question.dtype())?;
        let signal = signal.broadcast_mul(&keep.unsqueeze(D::Minus1)?)?;
        let question = question.broadcast_mul(&keep_q.unsqueeze(D::Minus1)?)?;
        let blocked = additive_mask(valid, f32::NEG_INFINITY)?;
        match self {
            Self::Pointer(head) => {
                let pooled = question.sum(1)?.broadcast_div(&keep_q.sum_keepdim(1)?)?;
                let query = head.query.forward(&pooled)?; // [B, h]
                let keys = head.key.forward(&signal)?; // [B, M, h]
                let patch = (keys
                    .matmul(&query.unsqueeze(D::Minus1)?)?
                    .squeeze(D::Minus1)?
                    * head.scale)?;
                let patch = patch.broadcast_add(&blocked)?;
                let null = (query
                    .broadcast_mul(&head.null_key)?
                    .sum_keepdim(D::Minus1)?
                    * head.scale)?;
                Ok(Tensor::cat(&[&patch, &null], D::Minus1)?)
            }
            Self::Attention(head) => {
                head.forward(&signal, valid, &question, question_valid, &blocked, tokens)
            }
        }
    }
}

impl AttentionHead {
    fn forward(
        &self,
        signal: &Tensor,
        valid: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
        blocked: &Tensor,
        tokens: usize,
    ) -> Result<Tensor> {
        let (batch, count, _) = signal.dims3()?;
        if tokens == 0 || tokens > self.max_tokens || count % tokens != 0 {
            return Err(format_error(
                "tokens_per_channel must divide the signal tokens and not exceed max_tokens",
            ));
        }
        let keep = valid.to_dtype(signal.dtype())?.unsqueeze(D::Minus1)?;
        let steps: Vec<u32> = (0..count).map(|i| (i % tokens) as u32).collect();
        let steps = Tensor::from_vec(steps, count, signal.device())?;
        let x = self.signal.forward(signal)?;
        let x = x
            .broadcast_add(&self.position.index_select(&steps, 0)?)?
            .broadcast_mul(&keep)?;
        let hidden = x.dim(D::Minus1)?;
        let null = self
            .null_token
            .reshape((1, 1, hidden))?
            .repeat((batch, 1, 1))?;
        let mut x = Tensor::cat(&[&x, &null], 1)?;
        let all = Tensor::cat(&[valid, &valid.ones_like()?.narrow(1, 0, 1)?], 1)?;
        let question = self.question.forward(question)?;
        for block in &self.blocks {
            x = block.forward(&x, &all, &question, question_valid, tokens)?;
        }
        let x = self.out_norm.forward(&x)?;
        let patch = self
            .patch_logit
            .forward(&x.narrow(1, 0, count)?)?
            .squeeze(D::Minus1)?;
        let patch = patch.broadcast_add(blocked)?;
        let null = self
            .null_logit
            .forward(&x.narrow(1, count, 1)?.squeeze(1)?)?;
        Ok(Tensor::cat(&[&patch, &null], D::Minus1)?)
    }
}

impl AttentionBlock {
    fn forward(
        &self,
        x: &Tensor,
        valid: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
        tokens: usize,
    ) -> Result<Tensor> {
        let (batch, total, dim) = x.dims3()?;
        let m = total - 1;
        let h = self.norms[0].forward(x)?;
        let x = (x + self
            .question
            .forward(&h, question, question, Some(question_valid))?)?;
        // Depthwise convolution along each channel's patches (zero padding at both ends).
        let keep = valid
            .narrow(1, 0, m)?
            .to_dtype(x.dtype())?
            .unsqueeze(D::Minus1)?;
        let patches = self.norms[1]
            .forward(&x.narrow(1, 0, m)?)?
            .broadcast_mul(&keep)?;
        let rows = batch * (m / tokens);
        let patches = patches.reshape((rows, tokens, dim))?;
        let kernel = self.conv_weight.dim(1)?;
        let pad = kernel / 2;
        let zeros = Tensor::zeros((rows, pad, dim), x.dtype(), x.device())?;
        let padded = Tensor::cat(&[&zeros, &patches, &zeros], 1)?;
        let mut local = self
            .conv_bias
            .reshape((1, 1, dim))?
            .broadcast_as((rows, tokens, dim))?
            .contiguous()?;
        for j in 0..kernel {
            let weight = self.conv_weight.narrow(1, j, 1)?.reshape((1, 1, dim))?;
            local = (local + padded.narrow(1, j, tokens)?.broadcast_mul(&weight)?)?;
        }
        let local = local.reshape((batch, m, dim))?;
        let x = Tensor::cat(&[&(x.narrow(1, 0, m)? + local)?, &x.narrow(1, m, 1)?], 1)?;
        let h = self.norms[2].forward(&x)?;
        let x = (&x + self.mixing.forward(&h, &h, &h, Some(valid))?)?;
        let h = self
            .mlp
            .1
            .forward(&gelu(&self.mlp.0.forward(&self.norms[3].forward(&x)?)?)?)?;
        Ok((x + h)?)
    }
}

/// A decoded location: the channel, its patch, and the samples that patch covers.
#[derive(Debug, Clone, PartialEq)]
pub struct Location {
    pub channel: usize,
    pub name: Option<String>,
    pub patch: usize,
    /// First sample of the patch (inclusive), in the coordinates of the given series.
    pub start: usize,
    /// End of the patch (exclusive).
    pub end: usize,
    pub probability: f32,
}
