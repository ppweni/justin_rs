//! MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0) as Justin's text encoder.

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::error::{Error, Result, format_error};
use crate::nn::masked_mean;
use crate::weights::Weights;

pub const MAX_LENGTH: usize = 256;

/// Token states and pooled sentence embeddings, as `justin.model.encoders.TextEncoder`.
pub struct TextEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
    hidden_size: usize,
    device: Device,
}

impl TextEncoder {
    /// `weights` are the BERT weights (adapters already merged), `config` its `config.json`,
    /// `tokenizer` its `tokenizer.json`. Truncation (256 tokens) and padding (longest in the batch)
    /// are set here, as the Python tokenizer call does, whatever the file says.
    pub fn load(config: &Path, tokenizer: &Path, weights: Weights) -> Result<Self> {
        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(config)?)?;
        let device = weights.device().clone();
        let mut tensors = weights.into_map();
        tensors.retain(|name, _| !name.ends_with("position_ids"));
        let builder = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = BertModel::load(builder, &config)?;
        let mut tokenizer =
            Tokenizer::from_file(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_LENGTH,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                ..Default::default()
            }))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let pad_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or_else(|| format_error("tokenizer has no [PAD]"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));
        Ok(Self {
            model,
            tokenizer,
            hidden_size: config.hidden_size,
            device,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Token ids [B, L] and attention mask [B, L] (u32, 1 = token) of a batch of texts.
    pub fn tokenize(&self, texts: &[String]) -> Result<(Tensor, Tensor)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let length = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let mut ids = Vec::with_capacity(texts.len() * length);
        let mut mask = Vec::with_capacity(texts.len() * length);
        for encoding in &encodings {
            ids.extend_from_slice(encoding.get_ids());
            mask.extend_from_slice(encoding.get_attention_mask());
        }
        let ids = Tensor::from_vec(ids, (texts.len(), length), &self.device)?;
        let mask = Tensor::from_vec(mask, (texts.len(), length), &self.device)?;
        Ok((ids, mask))
    }

    /// Per-token last hidden states [B, L, H] and the mask [B, L] (u32, 1 = valid).
    pub fn token_states(&self, texts: &[String]) -> Result<(Tensor, Tensor)> {
        let (ids, mask) = self.tokenize(texts)?;
        let types = ids.zeros_like()?;
        let hidden = self.model.forward(&ids, &types, Some(&mask))?;
        Ok((hidden, mask))
    }

    /// Masked-mean pooled, L2-normalized sentence embeddings [B, H].
    pub fn embed(&self, texts: &[String]) -> Result<Tensor> {
        let (hidden, mask) = self.token_states(texts)?;
        let pooled = masked_mean(&hidden, &mask, 1)?;
        let norm = pooled
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        Ok(pooled.broadcast_div(&norm)?)
    }
}
