//! Timestral (Mistral AI, code Apache-2.0, weights Modified MIT): the encoder half of the
//! forecaster, as `justin/model/timestral.py` vendors it.
//!
//! The causal per-patch scaler runs in float64 on the CPU in plain Rust (as upstream; Metal has no
//! float64). Tensors inside the stack are [B, C, N, d]: time attention is causal along N within
//! each (sample, channel), variate attention runs along C within each (sample, patch).

use std::path::Path;

use candle_core::{D, Device, Tensor};
use serde::Deserialize;

use crate::error::{Result, format_error, input_error};
use crate::nn::{Linear, additive_mask, sigmoid, silu, softmax};
use crate::weights::Weights;

/// The fields of a checkpoint's `config.json` that shape the encoder.
#[derive(Debug, Clone, Deserialize)]
pub struct TimestralConfig {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub context_length: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial")]
    pub rope_partial_factor: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: String,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "default_minimum_scale")]
    pub minimum_scale: f64,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
}

fn default_theta() -> f64 {
    10000.0
}
fn default_partial() -> f64 {
    0.5
}
fn default_norm_eps() -> f64 {
    1e-6
}
fn default_gate() -> String {
    "sdpa_sigmoid".into()
}
fn default_minimum_scale() -> f64 {
    1e-6
}
fn default_true() -> bool {
    true
}

impl TimestralConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(format_error("d_model must be divisible by num_heads"));
        }
        let rope = self.rope_dim();
        if rope == 0 || !rope.is_multiple_of(2) {
            return Err(format_error(format!(
                "rope_partial_factor gives an invalid rope_dim={rope}"
            )));
        }
        for kind in &self.block_schedule {
            axes(kind)?;
        }
        for gate in [&self.time_attn_gate, &self.variate_attn_gate] {
            if gate != "none" && gate != "sdpa_sigmoid" {
                return Err(format_error(format!("unknown attention gate {gate}")));
            }
        }
        if self.stabilize_with_global {
            return Err(format_error("stabilize_with_global is not supported"));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rope_partial_factor) as usize
    }
}

/// The attention axes of a layer kind, in order; true = time.
fn axes(kind: &str) -> Result<Vec<bool>> {
    Ok(match kind {
        "time" => vec![true],
        "variate" => vec![false],
        "time_variate" => vec![true, false],
        "variate_time" => vec![false, true],
        _ => return Err(format_error(format!("unknown layer kind {kind}"))),
    })
}

/// Timestral's RMSNorm: `x * rsqrt(mean(x^2) + eps) * scale`.
struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(weights: &Weights, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            scale: weights.get(&format!("{prefix}.scale"))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        Ok(x.broadcast_mul(&rms)?.broadcast_mul(&self.scale)?)
    }
}

struct Attention {
    qkv: Linear,
    out: Linear,
    gate: Option<Linear>,
    heads: usize,
    head_dim: usize,
    is_time: bool,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (rows, seq, dim) = x.dims3()?;
        // The fused weight is laid out as (qkv, head_dim, heads).
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((rows, seq, 3, self.head_dim, self.heads))?;
        let qkv = qkv.permute((2, 0, 1, 4, 3))?; // [3, rows, seq, heads, head_dim]
        let pick = |i: usize| -> Result<Tensor> { Ok(qkv.get(i)?.contiguous()?) };
        let (mut q, mut k, v) = (pick(0)?, pick(1)?, pick(2)?);
        if self.is_time {
            let (cos, sin) =
                rope.ok_or_else(|| format_error("time attention needs rotary tables"))?;
            let cos = cos.unsqueeze(2)?;
            let sin = sin.unsqueeze(2)?;
            q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
            k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;
        }
        let q = q.transpose(1, 2)?.contiguous()?; // [rows, heads, seq, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let scores = (q.matmul(&k.t()?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
        let scores = scores.broadcast_add(mask)?;
        let out = softmax(&scores)?.matmul(&v)?;
        let mut out = out.transpose(1, 2)?.reshape((rows, seq, dim))?;
        if let Some(gate) = &self.gate {
            out = (out * sigmoid(&gate.forward(x)?)?)?;
        }
        self.out.forward(&out)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, last - half)?;
    Ok(Tensor::cat(&[&second.neg()?, &first], D::Minus1)?)
}

struct Block {
    axes: Vec<bool>,
    norms: Vec<RmsNorm>,
    attentions: Vec<Attention>,
    mlp_norm: RmsNorm,
    w12: Linear,
    w3: Linear,
}

impl Block {
    fn forward(&self, x: &Tensor, masks: &Masks) -> Result<Tensor> {
        let (b, c, n, d) = x.dims4()?;
        let mut x = x.clone();
        for ((norm, attention), is_time) in self.norms.iter().zip(&self.attentions).zip(&self.axes)
        {
            let y = norm.forward(&x)?;
            if *is_time {
                let out = attention.forward(
                    &y.reshape((b * c, n, d))?,
                    &masks.time,
                    Some((&masks.cos, &masks.sin)),
                )?;
                x = (x + out.reshape((b, c, n, d))?)?;
            } else {
                let y = y.transpose(1, 2)?.reshape((b * n, c, d))?;
                let out = attention.forward(&y, &masks.variate, None)?;
                x = (x + out.reshape((b, n, c, d))?.transpose(1, 2)?)?;
            }
        }
        let h = self.w12.forward(&self.mlp_norm.forward(&x)?)?;
        let hidden = h.dim(D::Minus1)? / 2;
        let gate = h.narrow(D::Minus1, 0, hidden)?;
        let up = h.narrow(D::Minus1, hidden, hidden)?;
        Ok((&x + self.w3.forward(&(gate * silu(&up)?)?)?)?)
    }
}

/// The masks and rotary tables of one pass, built from the token validity.
struct Masks {
    time: Tensor,
    variate: Tensor,
    cos: Tensor,
    sin: Tensor,
}

/// Prepared encoder input: the patch embedding's inputs and the patch validity.
pub struct Prepared {
    /// [B * C, N, 2 * patch]: asinh-scaled values then missing flags, per patch.
    pub patches: Tensor,
    /// [B, C, N] true where a patch has any observed value.
    pub valid: Vec<bool>,
    pub batch: usize,
    pub channels: usize,
    pub patches_per_channel: usize,
}

/// The Timestral encoder: series -> per-(channel, patch) states.
pub struct Timestral {
    pub config: TimestralConfig,
    hidden: Linear,
    output: Linear,
    residual: Linear,
    blocks: Vec<Block>,
    out_norm: RmsNorm,
    inv_freq: Vec<f32>,
    device: Device,
}

impl Timestral {
    /// Build from a checkpoint's weights (names as in the released `model.safetensors`; the
    /// forecast head and query are ignored). `depth` keeps only the leading blocks.
    pub fn load(config: TimestralConfig, weights: &Weights, depth: Option<usize>) -> Result<Self> {
        let d = config.d_model;
        let total = config.block_schedule.len();
        let depth = depth.unwrap_or(total);
        if depth == 0 || depth > total {
            return Err(format_error(format!(
                "series_depth must be in [1, {total}]"
            )));
        }
        let mut blocks = Vec::with_capacity(depth);
        for (i, kind) in config.block_schedule.iter().take(depth).enumerate() {
            let prefix = format!("transformer.layers.{i}");
            let block_axes = axes(kind)?;
            let mut norms = Vec::new();
            let mut attentions = Vec::new();
            for (j, is_time) in block_axes.iter().enumerate() {
                norms.push(RmsNorm::load(
                    weights,
                    &format!("{prefix}.attn_norms.{j}"),
                    config.norm_eps,
                )?);
                let gate = if *is_time {
                    &config.time_attn_gate
                } else {
                    &config.variate_attn_gate
                };
                let at = format!("{prefix}.attns.{j}");
                attentions.push(Attention {
                    qkv: Linear::load(weights, &format!("{at}.wQKV"), config.attn_bias)?,
                    out: Linear::load(weights, &format!("{at}.wO"), config.attn_bias)?,
                    gate: if gate == "sdpa_sigmoid" {
                        Some(Linear::load(
                            weights,
                            &format!("{at}._gate"),
                            config.attn_bias,
                        )?)
                    } else {
                        None
                    },
                    heads: config.num_heads,
                    head_dim: config.head_dim(),
                    is_time: *is_time,
                });
            }
            blocks.push(Block {
                axes: block_axes,
                norms,
                attentions,
                mlp_norm: RmsNorm::load(weights, &format!("{prefix}.mlp_norm"), config.norm_eps)?,
                w12: Linear::load(weights, &format!("{prefix}.mlp.w12"), config.mlp_bias)?,
                w3: Linear::load(weights, &format!("{prefix}.mlp.w3"), config.mlp_bias)?,
            });
        }
        let head_dim = config.head_dim();
        let rotary = config.rope_dim();
        let mut inv_freq = vec![0f32; head_dim / 2];
        for (i, slot) in inv_freq.iter_mut().take(rotary / 2).enumerate() {
            let index = (2 * i) as f32;
            *slot = 1.0 / (config.rope_theta as f32).powf(index / rotary as f32);
        }
        let prefix = "representation.patch_proj";
        let encoder = Self {
            hidden: Linear::load(weights, &format!("{prefix}.hidden_layer"), true)?,
            output: Linear::load(weights, &format!("{prefix}.output_layer"), true)?,
            residual: Linear::load(weights, &format!("{prefix}.residual_layer"), true)?,
            blocks,
            out_norm: RmsNorm::load(weights, "transformer.out_norm", config.norm_eps)?,
            inv_freq,
            device: weights.device().clone(),
            config,
        };
        debug_assert_eq!(encoder.hidden.weight().dims()[0], 4 * d);
        Ok(encoder)
    }

    pub fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Crop, pad and scale a batch of series: `series[b][c]` all of one length T (NaN = missing),
    /// `padded[b][c]` optional left padding flags of the same shape. Keeps the most recent
    /// `context_length` steps and left-pads to a multiple of the patch size, as the Python wrapper.
    pub fn prepare(
        &self,
        series: &[Vec<Vec<f32>>],
        padded: Option<&[Vec<Vec<bool>>]>,
    ) -> Result<Prepared> {
        let batch = series.len();
        let channels = series.first().map_or(0, Vec::len);
        let length = series.first().and_then(|s| s.first()).map_or(0, Vec::len);
        if batch == 0 || channels == 0 || length == 0 {
            return Err(input_error(
                "series must have nonempty [batch, channels, time] shape",
            ));
        }
        for sample in series {
            if sample.len() != channels || sample.iter().any(|row| row.len() != length) {
                return Err(input_error(
                    "every series in a batch needs the same channels and length",
                ));
            }
        }
        let p = self.config.patch_size;
        let start = length.saturating_sub(self.config.context_length);
        let kept = length - start;
        let pad = (p - kept % p) % p;
        let t = kept + pad;
        let n = t / p;
        let mut patches = vec![0f32; batch * channels * n * 2 * p];
        let mut valid = vec![false; batch * channels * n];
        let mut values = vec![0f64; t];
        let mut observed = vec![false; t];
        for b in 0..batch {
            for c in 0..channels {
                for i in 0..t {
                    if i < pad {
                        values[i] = 0.0;
                        observed[i] = false;
                    } else {
                        let x = series[b][c][start + i - pad];
                        let is_padding = padded.is_some_and(|m| m[b][c][start + i - pad]);
                        observed[i] = !is_padding && !x.is_nan();
                        values[i] = if observed[i] { x as f64 } else { 0.0 };
                    }
                }
                let scaled = self.scale(&values, &observed);
                let row = b * channels + c;
                for patch in 0..n {
                    let base = (row * n + patch) * 2 * p;
                    for j in 0..p {
                        let i = patch * p + j;
                        patches[base + j] = scaled[i];
                        patches[base + p + j] = if observed[i] { 0.0 } else { 1.0 };
                        if observed[i] {
                            valid[row * n + patch] = true;
                        }
                    }
                }
            }
        }
        let patches = Tensor::from_vec(patches, (batch * channels, n, 2 * p), &Device::Cpu)?
            .to_device(&self.device)?;
        Ok(Prepared {
            patches,
            valid,
            batch,
            channels,
            patches_per_channel: n,
        })
    }

    /// Causal per-patch normalization of one row: running mean and Bessel-corrected std (Welford
    /// form, float64), read at the end of each patch, then `asinh((x - loc) / scale)` in float32.
    fn scale(&self, values: &[f64], observed: &[bool]) -> Vec<f32> {
        let p = self.config.patch_size;
        let t = values.len();
        let mut count = 0f64;
        let mut weighted = 0f64;
        let mut m2 = 0f64;
        let mut previous = 0f64;
        let mut mean = vec![0f64; t];
        let mut std = vec![0f64; t];
        for i in 0..t {
            let w = if observed[i] { 1.0 } else { 0.0 };
            count += w;
            weighted += w * values[i];
            let n = count.max(1.0);
            let current = weighted / n;
            m2 += (values[i] - previous) * (values[i] - current) * w;
            mean[i] = current;
            std[i] = (m2 / (n - 1.0).max(1.0))
                .sqrt()
                .max(self.config.minimum_scale);
            previous = current;
        }
        let mut out = vec![0f32; t];
        for i in 0..t {
            if !observed[i] {
                continue;
            }
            let end = (i / p) * p + p - 1;
            let loc = mean[end] as f32;
            let scale = (std[end] as f32).max(f32::EPSILON);
            let z = (values[i] as f32 - loc) / scale;
            out[i] = if self.config.use_asinh { z.asinh() } else { z };
        }
        out
    }

    /// Run the prepared series through the patch embedding and the stack. `film` holds one
    /// (scale, shift) pair of [B, d] per leading block. Returns ([B, C * N, d], valid [B, C * N]).
    pub fn forward(
        &self,
        prepared: &Prepared,
        film: &[(Tensor, Tensor)],
    ) -> Result<(Tensor, Tensor)> {
        let (b, c, n) = (
            prepared.batch,
            prepared.channels,
            prepared.patches_per_channel,
        );
        let d = self.config.d_model;
        let embedded = (self
            .output
            .forward(&silu(&self.hidden.forward(&prepared.patches)?)?)?
            + self.residual.forward(&prepared.patches)?)?;
        let mut x = embedded.reshape((b, c, n, d))?;
        let masks = self.masks(&prepared.valid, b, c, n)?;
        for (i, block) in self.blocks.iter().enumerate() {
            if let Some((scale, shift)) = film.get(i) {
                let scale = scale.reshape((b, 1, 1, d))?;
                let shift = shift.reshape((b, 1, 1, d))?;
                x = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
            }
            x = block.forward(&x, &masks)?;
        }
        let x = self.out_norm.forward(&x)?.reshape((b, c * n, d))?;
        let valid: Vec<u8> = prepared.valid.iter().map(|&v| v as u8).collect();
        let valid = Tensor::from_vec(valid, (b, c * n), &Device::Cpu)?.to_device(&self.device)?;
        Ok((x, valid))
    }

    fn masks(&self, valid: &[bool], b: usize, c: usize, n: usize) -> Result<Masks> {
        // A missing patch after a channel has started is still a key (attend_missing_patches).
        let mut keys = valid.to_vec();
        if self.config.attend_missing_patches {
            for row in 0..b * c {
                let mut seen = false;
                for patch in 0..n {
                    seen |= valid[row * n + patch];
                    keys[row * n + patch] = seen;
                }
            }
        }
        let mut time = vec![0u8; b * c * n * n];
        for row in 0..b * c {
            for i in 0..n {
                for j in 0..n {
                    time[(row * n + i) * n + j] = ((j <= i && keys[row * n + j]) || i == j) as u8;
                }
            }
        }
        let mut variate = vec![0u8; b * n * c * c];
        for sample in 0..b {
            for patch in 0..n {
                for i in 0..c {
                    for j in 0..c {
                        let key = keys[(sample * c + j) * n + patch];
                        variate[((sample * n + patch) * c + i) * c + j] = (key || i == j) as u8;
                    }
                }
            }
        }
        let time = Tensor::from_vec(time, (b * c, 1, n, n), &Device::Cpu)?;
        let variate = Tensor::from_vec(variate, (b * n, 1, c, c), &Device::Cpu)?;
        // Positions count from the first patch any channel of the sample observes.
        let head_dim = self.config.head_dim();
        let half = head_dim / 2;
        let mut cos = vec![0f32; b * c * n * head_dim];
        let mut sin = vec![0f32; b * c * n * head_dim];
        for sample in 0..b {
            let first = (0..n)
                .find(|&patch| (0..c).any(|ch| valid[(sample * c + ch) * n + patch]))
                .unwrap_or(0);
            for ch in 0..c {
                let row = sample * c + ch;
                for patch in 0..n {
                    let position = patch as f32 - first as f32;
                    for k in 0..half {
                        let angle = self.inv_freq[k] * position;
                        let (s, co) = (angle.sin(), angle.cos());
                        let base = (row * n + patch) * head_dim;
                        cos[base + k] = co;
                        cos[base + half + k] = co;
                        sin[base + k] = s;
                        sin[base + half + k] = s;
                    }
                }
            }
        }
        let device = &self.device;
        Ok(Masks {
            time: additive_mask(&time, f32::MIN)?.to_device(device)?,
            variate: additive_mask(&variate, f32::MIN)?.to_device(device)?,
            cos: Tensor::from_vec(cos, (b * c, n, head_dim), &Device::Cpu)?.to_device(device)?,
            sin: Tensor::from_vec(sin, (b * c, n, head_dim), &Device::Cpu)?.to_device(device)?,
        })
    }
}

/// Load the released weights of a Timestral checkpoint directory's `model.safetensors`, dropping
/// the forecast head and query.
pub fn load_backbone_weights(path: &Path, device: &Device) -> Result<Weights> {
    let mut weights = Weights::load(path, device)?;
    for name in weights.names_with_prefix("head.") {
        weights.remove(&name);
    }
    weights.remove("representation.forecast_query");
    Ok(weights)
}
