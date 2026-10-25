//! `SystemOne`'s trained part (`justin/model/model.py`, `justin/model/fusion.py`): question
//! conditioning, channel biases, cross-attention fusion and the heads.

use candle_core::{D, Tensor};
use serde::Deserialize;

use crate::error::{Result, format_error, input_error};
use crate::nn::{Encoder, LayerNorm, Linear, Mlp, MultiheadAttention, gelu, masked_mean};
use crate::weights::Weights;

/// The model-defining fields of an export's config (`MODEL_FIELDS` in `justin/export.py`).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub num_classes: Option<usize>,
    pub score_levels: usize,
    pub hidden_dim: usize,
    pub fusion_layers: usize,
    #[serde(default = "default_true")]
    pub with_text: bool,
    #[serde(default)]
    pub option_attention: bool,
    #[serde(default)]
    pub noul_attention: bool,
    #[serde(default)]
    pub signal_skip: bool,
    #[serde(default)]
    pub channel_names: bool,
    #[serde(default)]
    pub channel_features: usize,
    #[serde(default)]
    pub channel_head: bool,
    #[serde(default)]
    pub referents: bool,
    pub series_encoder: String,
    pub text_encoder: String,
    #[serde(default)]
    pub series_depth: Option<usize>,
    #[serde(default)]
    pub series_condition: usize,
    #[serde(default)]
    pub series_lora_layers: usize,
    #[serde(default)]
    pub minilm_lora: bool,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: usize,
}

fn default_true() -> bool {
    true
}
fn default_rank() -> usize {
    8
}
fn default_alpha() -> usize {
    16
}

const HEADS: usize = 8;

/// One `_FusionBlock`: question tokens attend over keys, then a feed-forward layer (post-norm).
struct FusionBlock {
    attention: MultiheadAttention,
    norm1: LayerNorm,
    ffn: (Linear, Linear),
    norm2: LayerNorm,
}

impl FusionBlock {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            attention: MultiheadAttention::load(weights, &format!("{prefix}.attention"), HEADS)?,
            norm1: LayerNorm::load(weights, &format!("{prefix}.norm1"), 1e-5)?,
            ffn: (
                Linear::load(weights, &format!("{prefix}.ffn.0"), true)?,
                Linear::load(weights, &format!("{prefix}.ffn.3"), true)?,
            ),
            norm2: LayerNorm::load(weights, &format!("{prefix}.norm2"), 1e-5)?,
        })
    }

    fn forward(&self, queries: &Tensor, keys: &Tensor, keys_valid: &Tensor) -> Result<Tensor> {
        let attended = self
            .attention
            .forward(queries, keys, keys, Some(keys_valid))?;
        let queries = self.norm1.forward(&(queries + attended)?)?;
        let ffn = self.ffn.1.forward(&gelu(&self.ffn.0.forward(&queries)?)?)?;
        self.norm2.forward(&(queries + ffn)?)
    }
}

/// `CrossAttentionFusion`.
struct Fusion {
    query: Linear,
    kv: Linear,
    blocks: Vec<FusionBlock>,
    residual: Linear,
    skip: Option<(LayerNorm, Linear)>,
    norm: LayerNorm,
}

impl Fusion {
    fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            query: Linear::load(weights, "fusion.query_projection", true)?,
            kv: Linear::load(weights, "fusion.kv_projection", true)?,
            blocks: (0..config.fusion_layers)
                .map(|i| FusionBlock::load(weights, &format!("fusion.blocks.{i}")))
                .collect::<Result<_>>()?,
            residual: Linear::load(weights, "fusion.residual", true)?,
            skip: if config.signal_skip {
                Some((
                    LayerNorm::load(weights, "fusion.signal_skip.0", 1e-5)?,
                    Linear::load(weights, "fusion.signal_skip.1", true)?,
                ))
            } else {
                None
            },
            norm: LayerNorm::load(weights, "fusion.norm", 1e-5)?,
        })
    }

    fn forward(
        &self,
        signal: &Tensor,
        signal_valid: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
    ) -> Result<Tensor> {
        let mut queries = self.query.forward(question)?;
        let keys = self.kv.forward(signal)?;
        for block in &self.blocks {
            queries = block.forward(&queries, &keys, signal_valid)?;
        }
        let pooled = masked_mean(&queries, question_valid, 1)?;
        let question_mean = masked_mean(question, question_valid, 1)?;
        let mut z = (pooled + self.residual.forward(&question_mean)?)?;
        if let Some((norm, linear)) = &self.skip {
            let signal_mean = masked_mean(signal, signal_valid, 1)?;
            z = (z + linear.forward(&norm.forward(&signal_mean)?)?)?;
        }
        gelu(&self.norm.forward(&z)?)
    }
}

/// `OptionAttention`: each option's query reads the signal (or channel) tokens.
struct OptionAttention {
    kv: Linear,
    attention: MultiheadAttention,
    norm: LayerNorm,
    scorer: Linear,
}

impl OptionAttention {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            kv: Linear::load(weights, &format!("{prefix}.kv_projection"), true)?,
            attention: MultiheadAttention::load(weights, &format!("{prefix}.attention"), HEADS)?,
            norm: LayerNorm::load(weights, &format!("{prefix}.norm"), 1e-5)?,
            scorer: Linear::load(weights, &format!("{prefix}.scorer"), true)?,
        })
    }

    /// z [B, d], keys [N, d] or [B, N, d] (per sample), signal [B, M, d_s], valid [B, M] -> [B, N].
    fn forward(
        &self,
        z: &Tensor,
        keys: &Tensor,
        signal: &Tensor,
        valid: &Tensor,
    ) -> Result<Tensor> {
        let keys = if keys.rank() == 3 {
            keys.clone()
        } else {
            keys.unsqueeze(0)?
        };
        let queries = z.unsqueeze(1)?.broadcast_add(&keys)?;
        let kv = self.kv.forward(signal)?;
        let routed = self.attention.forward(&queries, &kv, &kv, Some(valid))?;
        Ok(self
            .scorer
            .forward(&self.norm.forward(&(queries + routed)?)?)?
            .squeeze(D::Minus1)?)
    }
}

/// Yes/no as a choice between two options (`noul_keys`, `noul_question`, `noul_attention`): each
/// option is a learned key plus the projected question embedding and reads the signal itself.
struct NoulAttention {
    keys: Tensor,
    question: Linear,
    attention: OptionAttention,
}

impl NoulAttention {
    fn load(weights: &Weights) -> Result<Self> {
        Ok(Self {
            keys: weights.get("noul_keys")?,
            question: Linear::load(weights, "noul_question", true)?,
            attention: OptionAttention::load(weights, "noul_attention")?,
        })
    }

    /// The yes-minus-no evidence [B] added to the linear yes/no logit.
    fn forward(
        &self,
        fused: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
        signal: &Tensor,
        signal_valid: &Tensor,
    ) -> Result<Tensor> {
        let pooled = masked_mean(question, question_valid, 1)?;
        let norm = pooled
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        let projected = self.question.forward(&pooled.broadcast_div(&norm)?)?;
        let keys = self
            .keys
            .unsqueeze(0)?
            .broadcast_add(&projected.unsqueeze(1)?)?;
        let evidence = self.attention.forward(fused, &keys, signal, signal_valid)?;
        Ok((evidence.narrow(1, 0, 1)? - evidence.narrow(1, 1, 1)?)?.squeeze(1)?)
    }
}

/// `ChannelHead`: one token per channel, self-attention between channels, read by the question.
struct ChannelHead {
    pool: (LayerNorm, Linear),
    names: Option<Linear>,
    features: Option<Mlp>,
    encoder: Encoder,
    query: Linear,
    read: FusionBlock,
    norm: LayerNorm,
}

impl ChannelHead {
    fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            pool: (
                LayerNorm::load(weights, "channel_head.pool.0", 1e-5)?,
                Linear::load(weights, "channel_head.pool.1", true)?,
            ),
            names: if config.channel_names {
                Some(Linear::load(weights, "channel_head.names", true)?)
            } else {
                None
            },
            features: if config.channel_features > 0 {
                Some(Mlp::load(weights, "channel_head.features", 2)?)
            } else {
                None
            },
            encoder: Encoder::load(weights, "channel_head.encoder", 1, HEADS)?,
            query: Linear::load(weights, "channel_head.query_projection", true)?,
            read: FusionBlock::load(weights, "channel_head.read")?,
            norm: LayerNorm::load(weights, "channel_head.norm", 1e-5)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        signal: &Tensor,
        signal_valid: &Tensor,
        channels: usize,
        question: &Tensor,
        question_valid: &Tensor,
        names: Option<&Tensor>,
        features: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, tokens, dim) = signal.dims3()?;
        if tokens % channels != 0 {
            return Err(input_error(
                "signal tokens must divide evenly among the channels",
            ));
        }
        let per = tokens / channels;
        let patches = signal.reshape((batch, channels, per, dim))?;
        let valid = signal_valid.reshape((batch, channels, per))?;
        let mean = masked_mean(&patches, &valid, 2)?;
        let channel_valid = valid
            .to_dtype(candle_core::DType::F32)?
            .max(2)?
            .ne(0.0)?
            .to_dtype(candle_core::DType::U8)?;
        let mut x = self.pool.1.forward(&self.pool.0.forward(&mean)?)?;
        if let Some(linear) = &self.names {
            let names =
                names.ok_or_else(|| input_error("this channel head needs channel names"))?;
            x = (x + linear.forward(names)?)?;
        }
        if let Some(mlp) = &self.features {
            let features =
                features.ok_or_else(|| input_error("this channel head needs channel features"))?;
            x = (x + mlp.forward(features)?)?;
        }
        let x = self.encoder.forward(&x, Some(&channel_valid))?;
        let queries = self
            .read
            .forward(&self.query.forward(question)?, &x, &channel_valid)?;
        let summary = masked_mean(&queries, question_valid, 1)?;
        Ok((self.norm.forward(&summary)?, x, channel_valid))
    }
}

/// The referent path: question features into the fused vector, option features into the scores.
struct Referents {
    question: Mlp,
    option: Linear,
    fuse: Linear,
    mixer: Encoder,
    norm: LayerNorm,
    score: Linear,
}

/// What a head is asked: an option list (or the fixed classes), yes/no or a score.
pub enum Head<'a> {
    Choice(Option<&'a Tensor>),
    YesNo,
    Score,
}

/// Everything `answer_cached` reads besides the series features, already on the device.
pub struct Extras<'a> {
    pub channel_embeddings: Option<&'a Tensor>,
    pub channel_features: Option<&'a Tensor>,
    /// ([B, QUESTION], [B, n, OPTION] when the question has options)
    pub referents: Option<(&'a Tensor, Option<&'a Tensor>)>,
}

/// The trained part of SystemOne.
pub struct System {
    pub config: ModelConfig,
    pub film: Vec<Linear>,
    pub channel_projection: Option<Linear>,
    feature_projection: Option<Mlp>,
    fusion: Fusion,
    option_projection: Linear,
    channel_head: Option<ChannelHead>,
    channel_options: Option<OptionAttention>,
    referents: Option<Referents>,
    option_attention: Option<OptionAttention>,
    classes: Option<Tensor>,
    noul: Linear,
    noul_attention: Option<NoulAttention>,
    score: Linear,
}

impl System {
    pub fn load(config: ModelConfig, weights: &Weights) -> Result<Self> {
        if !config.with_text {
            return Err(format_error(
                "models without a text encoder are not supported",
            ));
        }
        Ok(Self {
            film: (0..config.series_condition)
                .map(|i| Linear::load(weights, &format!("condition_film.{i}"), true))
                .collect::<Result<_>>()?,
            channel_projection: if config.channel_names {
                Some(Linear::load(weights, "channel_projection", true)?)
            } else {
                None
            },
            feature_projection: if config.channel_features > 0 {
                Some(Mlp::load(weights, "feature_projection", 2)?)
            } else {
                None
            },
            fusion: Fusion::load(weights, &config)?,
            option_projection: Linear::load(weights, "option_projection", true)?,
            channel_head: if config.channel_head {
                Some(ChannelHead::load(weights, &config)?)
            } else {
                None
            },
            channel_options: if config.channel_head {
                Some(OptionAttention::load(weights, "channel_options")?)
            } else {
                None
            },
            referents: if config.referents {
                Some(Referents {
                    question: Mlp::load(weights, "referent_question", 2)?,
                    option: Linear::load(weights, "referent_option", true)?,
                    fuse: Linear::load(weights, "referent_fuse", true)?,
                    mixer: Encoder::load(weights, "referent_mixer", 2, HEADS)?,
                    norm: LayerNorm::load(weights, "referent_norm", 1e-5)?,
                    score: Linear::load(weights, "referent_score", true)?,
                })
            } else {
                None
            },
            option_attention: if config.option_attention {
                Some(OptionAttention::load(weights, "option_attention")?)
            } else {
                None
            },
            classes: match config.num_classes {
                Some(_) => Some(weights.get("classes.weight")?),
                None => None,
            },
            noul: Linear::load(weights, "noul", true)?,
            noul_attention: if config.noul_attention {
                Some(NoulAttention::load(weights)?)
            } else {
                None
            },
            score: Linear::load(weights, "score", true)?,
            config,
        })
    }

    pub fn uses_features(&self) -> bool {
        self.feature_projection.is_some() || self.referents.is_some()
    }

    /// FiLM (scale, shift) pairs [B, d] from pooled question embeddings [B, d_text].
    pub fn film(&self, embedding: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
        self.film
            .iter()
            .map(|layer| {
                let out = layer.forward(embedding)?;
                let half = out.dim(D::Minus1)? / 2;
                Ok((
                    out.narrow(D::Minus1, 0, half)?,
                    out.narrow(D::Minus1, half, half)?,
                ))
            })
            .collect()
    }

    /// Add the per-channel biases (name projection and feature projection) to the signal tokens.
    pub fn add_channel_bias(&self, signal: &Tensor, extras: &Extras) -> Result<Tensor> {
        let mut bias: Option<Tensor> = None;
        if let Some(names) = extras.channel_embeddings {
            let projection = self
                .channel_projection
                .as_ref()
                .ok_or_else(|| input_error("this model was built without channel_names"))?;
            bias = Some(projection.forward(names)?);
        }
        match (extras.channel_features, &self.feature_projection) {
            (Some(_), None) => {
                return Err(input_error("this model was built without channel_features"));
            }
            (None, Some(_)) => return Err(input_error("this model needs channel_features")),
            (Some(features), Some(mlp)) => {
                let extra = mlp.forward(features)?;
                bias = Some(match bias {
                    Some(b) => (b + extra)?,
                    None => extra,
                });
            }
            (None, None) => {}
        }
        let Some(bias) = bias else {
            return Ok(signal.clone());
        };
        let (batch, tokens, dim) = signal.dims3()?;
        let channels = bias.dim(1)?;
        if tokens % channels != 0 {
            return Err(input_error(
                "signal tokens must divide evenly among the channels",
            ));
        }
        let shaped = signal.reshape((batch, channels, tokens / channels, dim))?;
        Ok(shaped
            .broadcast_add(&bias.unsqueeze(2)?)?
            .reshape((batch, tokens, dim))?)
    }

    /// `answer_cached` + `_decode` for one batch: signal [B, M, d_s] (raw encoder states),
    /// question tokens [B, L, d_t]. Returns logits: Choice [B, N], YesNo [B], Score [B, K].
    pub fn decode(
        &self,
        signal: &Tensor,
        signal_valid: &Tensor,
        question: &Tensor,
        question_valid: &Tensor,
        head: Head,
        extras: &Extras,
    ) -> Result<Tensor> {
        let channels = match (extras.channel_embeddings, extras.channel_features) {
            (Some(t), _) | (None, Some(t)) => Some(t.dim(1)?),
            (None, None) => None,
        };
        let signal = self.add_channel_bias(signal, extras)?;
        if extras.referents.is_some() != self.referents.is_some() {
            return Err(input_error(if self.referents.is_some() {
                "this model needs referent_question"
            } else {
                "this model was built without referents"
            }));
        }
        let mut fused = self
            .fusion
            .forward(&signal, signal_valid, question, question_valid)?;
        let mut channel_tokens = None;
        if let Some(channel_head) = &self.channel_head {
            let count = channels
                .ok_or_else(|| input_error("this model needs channel_names or channel_features"))?;
            let (summary, tokens, valid) = channel_head.forward(
                &signal,
                signal_valid,
                count,
                question,
                question_valid,
                extras.channel_embeddings,
                extras.channel_features,
            )?;
            fused = (fused + summary)?;
            channel_tokens = Some((tokens, valid));
        }
        if let (Some(referents), Some((question_features, _))) = (&self.referents, extras.referents)
        {
            fused = (fused + referents.question.forward(question_features)?)?;
        }
        match head {
            Head::YesNo => {
                let logit = self.noul.forward(&fused)?.squeeze(D::Minus1)?;
                match &self.noul_attention {
                    Some(attention) => Ok((logit
                        + attention.forward(
                            &fused,
                            question,
                            question_valid,
                            &signal,
                            signal_valid,
                        )?)?),
                    None => Ok(logit),
                }
            }
            Head::Score => self.score.forward(&fused),
            Head::Choice(options) => {
                let keys = match (options, &self.classes) {
                    (Some(options), _) => self.option_projection.forward(options)?,
                    (None, Some(classes)) => classes.clone(),
                    (None, None) => {
                        return Err(input_error("Choice requires options or num_classes"));
                    }
                };
                let scale = (fused.dim(D::Minus1)? as f64).sqrt();
                let mut logits = (fused.matmul(&keys.t()?)? / scale)?;
                if let (Some(referents), Some((_, Some(option_features)))) =
                    (&self.referents, extras.referents)
                    && option_features.dim(1)? == logits.dim(1)?
                {
                    let tokens = referents
                        .fuse
                        .forward(&fused)?
                        .unsqueeze(1)?
                        .broadcast_add(&referents.option.forward(option_features)?)?;
                    let mixed = referents.mixer.forward(&tokens, None)?;
                    let extra = referents
                        .score
                        .forward(&referents.norm.forward(&mixed)?)?
                        .squeeze(D::Minus1)?;
                    logits = (logits + extra)?;
                }
                if let (Some(attention), Some(_)) = (&self.option_attention, options) {
                    logits = (logits + attention.forward(&fused, &keys, &signal, signal_valid)?)?;
                }
                if let (Some(attention), Some(_), Some((tokens, valid))) =
                    (&self.channel_options, options, &channel_tokens)
                {
                    logits = (logits + attention.forward(&fused, &keys, tokens, valid)?)?;
                }
                Ok(logits)
            }
        }
    }
}
