//! Learned metric kinds (`justin/model/kinds.py`): each channel goes to the nearest of K
//! centroids of name embeddings, and its deviation sizes are summarized per kind (max, mean, and
//! whether the kind has a channel), once over every channel and once over the channels the
//! question quotes. An MLP turns the two tables into a vector added to the fused vector.

use std::sync::LazyLock;

use candle_core::{Device, Tensor};
use regex::Regex;

use crate::error::{Result, input_error};
use crate::features::N_FEATURES;
use crate::nn::Mlp;
use crate::weights::Weights;

/// Leading deviation-feature columns that measure how far a channel deviates.
const SIZES: usize = 2;
/// Per kind: max and mean of each size, and whether the kind has a channel.
const STATS: usize = 2 * SIZES + 1;

static NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

/// `sensor_6636_latency` -> `sensor_#_latency`: numbers in names are IDs, never kinds.
pub fn kind_pattern(name: &str) -> String {
    NUMBER.replace_all(name, "#").into_owned()
}

pub struct KindSummary {
    centroids: Vec<Vec<f32>>,
    mlp: Mlp,
}

impl KindSummary {
    pub fn load(weights: &Weights) -> Result<Self> {
        Ok(Self {
            centroids: weights
                .get("kind_summary.centroids")?
                .to_dtype(candle_core::DType::F32)?
                .to_vec2::<f32>()?,
            mlp: Mlp::load(weights, "kind_summary.mlp", 2)?,
        })
    }

    /// The kind of each name embedding (rows of `embeddings`): its nearest centroid.
    fn assign(&self, embeddings: &[Vec<f32>]) -> Vec<usize> {
        embeddings
            .iter()
            .map(|e| {
                let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let mut best = (0, f32::INFINITY);
                for (k, c) in self.centroids.iter().enumerate() {
                    let distance: f32 = e.iter().zip(c).map(|(x, y)| (x / norm - y).powi(2)).sum();
                    if distance < best.1 {
                        best = (k, distance);
                    }
                }
                best.0
            })
            .collect()
    }

    /// `embeddings` [1, C, d_txt] of the channels' name patterns, `features` per channel, and
    /// which channels the question quotes -> [1, hidden].
    pub fn forward(
        &self,
        embeddings: &Tensor,
        features: &[[f32; N_FEATURES]],
        quoted: &[bool],
        device: &Device,
    ) -> Result<Tensor> {
        let embeddings = embeddings
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;
        if embeddings.len() != features.len() || quoted.len() != features.len() {
            return Err(input_error(
                "kind summary: one name, feature row and quote flag per channel",
            ));
        }
        let kinds = self.assign(&embeddings);
        let mut table = self.table(&kinds, features, &vec![true; features.len()]);
        table.extend(self.table(&kinds, features, quoted));
        let width = table.len();
        let x = Tensor::from_vec(table, (1, width), device)?;
        self.mlp.forward(&x)
    }

    /// [K * STATS], kind by kind: max of each size, mean of each size, present.
    fn table(&self, kinds: &[usize], features: &[[f32; N_FEATURES]], keep: &[bool]) -> Vec<f32> {
        let k = self.centroids.len();
        let mut out = Vec::with_capacity(k * STATS);
        for kind in 0..k {
            let rows: Vec<&[f32; N_FEATURES]> = (0..features.len())
                .filter(|&c| kinds[c] == kind && keep[c])
                .map(|c| &features[c])
                .collect();
            if rows.is_empty() {
                out.extend([0.0; STATS]);
                continue;
            }
            for s in 0..SIZES {
                out.push(rows.iter().map(|r| r[s]).fold(f32::NEG_INFINITY, f32::max));
            }
            for s in 0..SIZES {
                out.push(rows.iter().map(|r| r[s]).sum::<f32>() / rows.len() as f32);
            }
            out.push(1.0);
        }
        out
    }
}
