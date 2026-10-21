//! The layers the Python model uses, with PyTorch's parameter names and semantics.

use candle_core::{D, DType, Tensor};

use crate::error::Result;
use crate::weights::Weights;

/// `nn.Linear`: `x @ W^T + b` over the last axis of any batch shape.
#[derive(Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn load(weights: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = if bias {
            Some(weights.get(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let input = *dims.last().expect("linear input has an axis");
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.reshape((rows, input))?.contiguous()?;
        let mut y = flat.matmul(&self.weight.t()?)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }
        let mut shape = dims[..dims.len() - 1].to_vec();
        shape.push(self.out_features());
        Ok(y.reshape(shape)?)
    }
}

/// `nn.LayerNorm` over the last axis.
#[derive(Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn load(weights: &Weights, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.weight"))?,
            bias: weights.get(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

/// Exact (erf) GELU, PyTorch's default.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

/// Softmax over the last axis.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax_last_dim(&x.contiguous()?)?)
}

/// An additive attention mask from a boolean "allowed" mask (1 = attend): 0 where allowed, `fill`
/// elsewhere. `fill` is `f32::MIN` for Timestral (PyTorch's `finfo.min`) and `-inf` where PyTorch's
/// `nn.MultiheadAttention` converts a key padding mask.
pub fn additive_mask(allowed: &Tensor, fill: f32) -> Result<Tensor> {
    let allowed = allowed.to_dtype(DType::F32)?;
    let blocked = (1.0 - &allowed)?;
    // Multiplying an infinite fill by zero would give NaN: select instead.
    let zeros = allowed.zeros_like()?;
    let filled = (zeros.ones_like()? * fill as f64)?;
    Ok(blocked.ne(0.0)?.where_cond(&filled, &zeros)?)
}

/// Mean over `axis` of the entries where `mask` (same shape without the feature axis, 1 = valid)
/// is set, with the count clamped to at least one (as the Python code does).
pub fn masked_mean(x: &Tensor, mask: &Tensor, axis: usize) -> Result<Tensor> {
    let weights = mask.to_dtype(x.dtype())?.unsqueeze(D::Minus1)?;
    let total = x.broadcast_mul(&weights)?.sum(axis)?;
    let count = weights.sum(axis)?.clamp(1.0, f64::MAX)?;
    Ok(total.broadcast_div(&count)?)
}

/// `nn.MultiheadAttention` (batch first) with its packed `in_proj` and `out_proj` parameters.
#[derive(Clone)]
pub struct MultiheadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl MultiheadAttention {
    pub fn load(weights: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.in_proj_weight"))?;
        let bias = weights.get(&format!("{prefix}.in_proj_bias"))?;
        let dim = weight.dims()[1];
        let part = |i: usize| -> Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, i * dim, dim)?,
                Some(bias.narrow(0, i * dim, dim)?),
            ))
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: Linear::load(weights, &format!("{prefix}.out_proj"), true)?,
            heads,
        })
    }

    /// query [B, Lq, d], key/value [B, Lk, d]; `key_valid` [B, Lk] (1 = attend) or `None`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_valid: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, lq, dim) = query.dims3()?;
        let lk = key.dims()[1];
        let head_dim = dim / self.heads;
        let split = |x: Tensor, length: usize| -> Result<Tensor> {
            Ok(x.reshape((batch, length, self.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = split(self.q.forward(query)?, lq)?;
        let k = split(self.k.forward(key)?, lk)?;
        let v = split(self.v.forward(value)?, lk)?;
        let mut scores = (q.matmul(&k.t()?)? * (1.0 / (head_dim as f64).sqrt()))?;
        if let Some(valid) = key_valid {
            let mask = additive_mask(valid, f32::NEG_INFINITY)?.reshape((batch, 1, 1, lk))?;
            scores = scores.broadcast_add(&mask)?;
        }
        let attended = softmax(&scores)?.matmul(&v)?;
        let merged = attended.transpose(1, 2)?.reshape((batch, lq, dim))?;
        self.out.forward(&merged)
    }
}

/// `nn.TransformerEncoderLayer(norm_first=True, activation="gelu")` in eval mode.
#[derive(Clone)]
pub struct EncoderLayer {
    attention: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn load(weights: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiheadAttention::load(weights, &format!("{prefix}.self_attn"), heads)?,
            linear1: Linear::load(weights, &format!("{prefix}.linear1"), true)?,
            linear2: Linear::load(weights, &format!("{prefix}.linear2"), true)?,
            norm1: LayerNorm::load(weights, &format!("{prefix}.norm1"), 1e-5)?,
            norm2: LayerNorm::load(weights, &format!("{prefix}.norm2"), 1e-5)?,
        })
    }

    pub fn forward(&self, x: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let x = (x + self.attention.forward(&h, &h, &h, valid)?)?;
        let h = self
            .linear2
            .forward(&gelu(&self.linear1.forward(&self.norm2.forward(&x)?)?)?)?;
        Ok((x + h)?)
    }
}

/// `nn.TransformerEncoder` of `EncoderLayer`s (no final norm).
#[derive(Clone)]
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn load(weights: &Weights, prefix: &str, layers: usize, heads: usize) -> Result<Self> {
        let layers = (0..layers)
            .map(|i| EncoderLayer::load(weights, &format!("{prefix}.layers.{i}"), heads))
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, valid)?;
        }
        Ok(x)
    }
}

/// `nn.Sequential(Linear, GELU, Linear)` with the Python indices 0 and 2.
#[derive(Clone)]
pub struct Mlp {
    first: Linear,
    second: Linear,
}

impl Mlp {
    pub fn load(weights: &Weights, prefix: &str, second_index: usize) -> Result<Self> {
        Ok(Self {
            first: Linear::load(weights, &format!("{prefix}.0"), true)?,
            second: Linear::load(weights, &format!("{prefix}.{second_index}"), true)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.second.forward(&gelu(&self.first.forward(x)?)?)
    }
}

/// Merge LoRA adapters into their base weights: `W += (alpha / rank) * B @ A`, for every
/// `<name>.a.weight` / `<name>.b.weight` pair in `adapters` under `prefix`, whose base is
/// `<name>.weight` in `base` (without the prefix). Returns how many layers were merged.
pub fn merge_lora(
    base: &mut Weights,
    adapters: &Weights,
    prefix: &str,
    alpha: f64,
    rank: usize,
) -> Result<usize> {
    let scaling = alpha / rank as f64;
    let mut merged = 0;
    for name in adapters.names_with_prefix(prefix) {
        let Some(stem) = name.strip_suffix(".a.weight") else {
            continue;
        };
        let a = adapters.get(&name)?;
        let b = adapters.get(&format!("{stem}.b.weight"))?;
        let target = format!("{}.weight", &stem[prefix.len()..]);
        let weight = base.remove(&target).ok_or_else(|| {
            crate::error::format_error(format!("LoRA adapter {stem} has no base {target}"))
        })?;
        let delta = (b.matmul(&a)? * scaling)?;
        base.insert(target, (weight + delta)?);
        merged += 1;
    }
    Ok(merged)
}
