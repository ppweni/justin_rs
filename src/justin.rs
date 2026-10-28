//! `Justin`: load an exported model (and optionally its location head) and answer a list of
//! queries about one multichannel series.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde::Deserialize;

use crate::error::{Result, format_error, input_error};
use crate::features::{self, Baseline, FEATURE_VERSION};
use crate::kinds::kind_pattern;
use crate::location::{Location, LocationModel, LocationSpec};
use crate::nn::{merge_lora, sigmoid, softmax};
use crate::source::{Backbones, Source, hub_file};
use crate::system::{Extras, Head, ModelConfig, System};
use crate::text::TextEncoder;
use crate::timestral::{Timestral, TimestralConfig, load_backbone_weights};
use crate::weights::{Weights, fingerprint, read_metadata};

const FORMAT_VERSION: &str = "1";
const LOCATION_FORMAT_VERSION: &str = "1";
const SERIES_PREFIX: &str = "series_encoder.backbone.";
const TEXT_PREFIX: &str = "text_encoder.backbone.";

/// How to load a model.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub device: Device,
    pub backbones: Backbones,
    /// A location head trained on this model, for `Query::Location`.
    pub location: Option<Source>,
    /// The Hugging Face cache to read and download into (default: the standard cache).
    pub cache_dir: Option<PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            backbones: Backbones::Hub,
            location: None,
            cache_dir: None,
        }
    }
}

/// What a question may be answered with.
#[derive(Debug, Clone, PartialEq)]
pub enum Answers {
    /// Pick one of these options.
    Options(Vec<String>),
    /// Yes or no.
    YesNo,
    /// A level of the score rubric.
    Score,
    /// One of the model's fixed classes (models trained with `num_classes`).
    Classes,
}

/// One thing to ask about the series.
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// A question with its possible answers.
    Question { question: String, answers: Answers },
    /// Where something is ("Locate the onset of the spike in cpu."); needs a location head.
    Location { question: String },
}

impl Query {
    pub fn choice(question: impl Into<String>, options: &[&str]) -> Self {
        Self::Question {
            question: question.into(),
            answers: Answers::Options(options.iter().map(|s| s.to_string()).collect()),
        }
    }

    pub fn yes_no(question: impl Into<String>) -> Self {
        Self::Question {
            question: question.into(),
            answers: Answers::YesNo,
        }
    }

    pub fn locate(question: impl Into<String>) -> Self {
        Self::Location {
            question: question.into(),
        }
    }
}

/// One series and the queries about it.
#[derive(Debug, Clone, Default)]
pub struct Input {
    /// `[channels][time]`, raw values; NaN marks a missing point (location queries only).
    pub series: Vec<Vec<f32>>,
    /// One name per channel; required by models trained with channel names.
    pub channel_names: Option<Vec<String>>,
    /// Seconds between points; adds the sampling sentence to questions and scales the timing features.
    pub step: Option<f64>,
    /// What counts as normal behavior for the deviation features.
    pub baseline: Baseline,
    pub queries: Vec<Query>,
}

/// How many locations [`Output::Location::candidates`] lists at most.
pub const LOCATION_CANDIDATES: usize = 5;

/// The answer to one query; probabilities use the calibration temperature of the head.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    Choice {
        logits: Vec<f32>,
        probabilities: Vec<f32>,
        best: usize,
        answer: Option<String>,
    },
    YesNo {
        logit: f32,
        probability: f32,
        answer: bool,
    },
    Score {
        logits: Vec<f32>,
        probabilities: Vec<f32>,
        best: usize,
    },
    Location {
        logits: Vec<f32>,
        null_probability: f32,
        location: Option<Location>,
        /// The most probable locations, best first, at most [`LOCATION_CANDIDATES`]; the null
        /// class is left out.
        candidates: Vec<Location>,
    },
}

#[derive(Deserialize)]
struct Revision {
    repo: String,
    revision: String,
}

/// A trained Justin model: Timestral + MiniLM + fusion and heads, optionally a location head.
pub struct Justin {
    pub config: ModelConfig,
    pub temperatures: HashMap<String, f64>,
    pub series: Timestral,
    pub text: TextEncoder,
    pub system: System,
    location: Option<(LocationModel, LocationSpec)>,
    device: Device,
}

fn timestral_size(name: &str) -> Result<()> {
    match name {
        "timestral5m" | "timestral28m" | "timestral123m" | "timestral209m" => Ok(()),
        other => Err(format_error(format!(
            "series encoder {other} is not supported (only Timestral)"
        ))),
    }
}

/// The adapted linear layers in Timestral blocks [boundary, depth) of a config.
fn lora_layer_count(config: &TimestralConfig, boundary: usize, depth: usize) -> usize {
    config.block_schedule[boundary..depth]
        .iter()
        .map(|kind| {
            let attentions: Vec<bool> = match kind.as_str() {
                "time" => vec![true],
                "variate" => vec![false],
                _ => vec![true, false],
            };
            attentions
                .iter()
                .map(|&time| {
                    let gate = if time {
                        &config.time_attn_gate
                    } else {
                        &config.variate_attn_gate
                    };
                    2 + usize::from(gate == "sdpa_sigmoid")
                })
                .sum::<usize>()
                + 2
        })
        .sum()
}

fn channel_text(name: &str) -> String {
    name.replace(['_', '-'], " ")
}

impl Justin {
    /// Load an export (`justin export`) from the Hub or a local file, its pinned backbones, and
    /// `options.location` if given (`justin export-location`, checked against this export).
    pub fn from_pretrained(model: Source, options: LoadOptions) -> Result<Self> {
        let cache = options.cache_dir.as_deref();
        let path = model.resolve(cache)?;
        let metadata = read_metadata(&path)?;
        if metadata.get("justin_format").map(String::as_str) != Some(FORMAT_VERSION) {
            return Err(format_error(format!(
                "{} is not a Justin export",
                path.display()
            )));
        }
        let field = |name: &str| -> Result<&String> {
            metadata
                .get(name)
                .ok_or_else(|| format_error(format!("export metadata misses {name}")))
        };
        let config: ModelConfig = serde_json::from_str(field("config")?)?;
        let revisions: HashMap<String, Revision> = serde_json::from_str(field("revisions")?)?;
        let temperatures: HashMap<String, f64> = serde_json::from_str(field("temperatures")?)?;
        if config.channel_features > 0 || config.referents {
            let version: String = serde_json::from_str(field("feature_version")?)?;
            if version != FEATURE_VERSION {
                return Err(format_error(format!(
                    "the export was trained with feature version {version}, this code computes {FEATURE_VERSION}"
                )));
            }
        }
        timestral_size(&config.series_encoder)?;
        if config.text_encoder != "minilm" {
            return Err(format_error(format!(
                "text encoder {} is not supported (only MiniLM)",
                config.text_encoder
            )));
        }
        let device = options.device.clone();
        let (series_dir, text_dir) = match &options.backbones {
            Backbones::Local { series, text } => {
                (Dirs::Local(series.clone()), Dirs::Local(text.clone()))
            }
            Backbones::Hub => {
                let get = |key: &str| -> Result<Dirs> {
                    let r = revisions
                        .get(key)
                        .ok_or_else(|| format_error(format!("export records no {key} backbone")))?;
                    Ok(Dirs::Hub(r.repo.clone(), r.revision.clone()))
                };
                (get("series")?, get("text")?)
            }
        };
        let export = Weights::load(&path, &device)?;

        let series_config = TimestralConfig::from_file(&series_dir.file("config.json", cache)?)?;
        let mut series_weights =
            load_backbone_weights(&series_dir.file("model.safetensors", cache)?, &device)?;
        let depth = config
            .series_depth
            .unwrap_or(series_config.block_schedule.len());
        if config.series_lora_layers > depth {
            return Err(format_error(
                "series_lora_layers exceeds the encoder's blocks",
            ));
        }
        let merged = merge_lora(
            &mut series_weights,
            &export,
            SERIES_PREFIX,
            config.lora_alpha as f64,
            config.lora_rank,
        )?;
        let expected = lora_layer_count(&series_config, depth - config.series_lora_layers, depth);
        if merged != expected {
            return Err(format_error(format!(
                "the export holds {merged} series LoRA adapters, the config needs {expected}"
            )));
        }
        let series = Timestral::load(series_config, &series_weights, config.series_depth)?;

        let mut text_weights = Weights::load(&text_dir.file("model.safetensors", cache)?, &device)?;
        let merged = merge_lora(
            &mut text_weights,
            &export,
            TEXT_PREFIX,
            config.lora_alpha as f64,
            config.lora_rank,
        )?;
        if config.minilm_lora != (merged > 0) {
            return Err(format_error(
                "the export's MiniLM adapters do not match minilm_lora",
            ));
        }
        let text = TextEncoder::load(
            &text_dir.file("config.json", cache)?,
            &text_dir.file("tokenizer.json", cache)?,
            text_weights,
        )?;

        let system = System::load(config.clone(), &export)?;
        let unused = export.unused();
        if !unused.is_empty() {
            return Err(format_error(format!(
                "{} does not match the model it describes (unused: {})",
                path.display(),
                unused[0]
            )));
        }
        let location = match &options.location {
            None => None,
            Some(source) => Some(Self::load_location(
                source,
                &path,
                cache,
                &device,
                series.hidden_size(),
                text.hidden_size(),
            )?),
        };
        Ok(Self {
            config,
            temperatures,
            series,
            text,
            system,
            location,
            device,
        })
    }

    fn load_location(
        source: &Source,
        base: &Path,
        cache: Option<&Path>,
        device: &Device,
        signal_dim: usize,
        text_dim: usize,
    ) -> Result<(LocationModel, LocationSpec)> {
        let path = source.resolve(cache)?;
        let metadata = read_metadata(&path)?;
        if metadata.get("justin_location_format").map(String::as_str)
            != Some(LOCATION_FORMAT_VERSION)
        {
            return Err(format_error(format!(
                "{} is not a Justin location export",
                path.display()
            )));
        }
        let spec: LocationSpec = serde_json::from_str(
            metadata
                .get("head")
                .ok_or_else(|| format_error("location metadata misses head"))?,
        )?;
        let recorded: String = serde_json::from_str(
            metadata
                .get("base_fingerprint")
                .ok_or_else(|| format_error("location metadata misses base_fingerprint"))?,
        )?;
        if recorded != fingerprint(base)? {
            return Err(format_error(
                "this location head does not belong to the loaded model (fingerprint differs)",
            ));
        }
        if spec.signal_dim != signal_dim || spec.text_dim != text_dim {
            return Err(format_error(
                "the location head's sizes do not fit the model",
            ));
        }
        let weights = Weights::load(&path, device)?;
        let head = LocationModel::load(&spec, &weights)?;
        let unused = weights.unused();
        if !unused.is_empty() {
            return Err(format_error(format!(
                "location file holds tensors the head does not use: {}",
                unused[0]
            )));
        }
        Ok((head, spec))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn has_location(&self) -> bool {
        self.location.is_some()
    }

    /// Answer every query about `input.series`, in order.
    pub fn forward(&self, input: &Input) -> Result<Vec<Output>> {
        let series = &input.series;
        if series.is_empty()
            || series[0].is_empty()
            || series.iter().any(|r| r.len() != series[0].len())
        {
            return Err(input_error(
                "series must have shape [channels, time] with at least one point",
            ));
        }
        if let Some(names) = &input.channel_names
            && names.len() != series.len()
        {
            return Err(input_error("channel_names needs one name per channel"));
        }
        input
            .queries
            .iter()
            .map(|query| match query {
                Query::Question { question, answers } => self.answer(input, question, answers),
                Query::Location { question } => self.locate(input, question),
            })
            .collect()
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn channel_embeddings(&self, names: &[String]) -> Result<Tensor> {
        let texts: Vec<String> = names.iter().map(|n| channel_text(n)).collect();
        Ok(self.text.embed(&texts)?.unsqueeze(0)?)
    }

    /// Encode the series [1, C, T] conditioned on `question` -> (signal [1, M, d], valid [1, M], N).
    fn encode(&self, series: &[Vec<f32>], question: &str) -> Result<(Tensor, Tensor, usize)> {
        let film = if self.system.film.is_empty() {
            Vec::new()
        } else {
            self.system
                .film(&self.text.embed(&Self::texts(&[question]))?)?
        };
        let prepared = self.series.prepare(&[series.to_vec()], None)?;
        let tokens = prepared.patches_per_channel;
        let (signal, valid) = self.series.forward(&prepared, &film)?;
        Ok((signal, valid, tokens))
    }

    fn answer(&self, input: &Input, question: &str, answers: &Answers) -> Result<Output> {
        let options: &[String] = match answers {
            Answers::Options(options) => {
                if !(2..=255).contains(&options.len()) {
                    return Err(input_error("Choice requires 2 to 255 options"));
                }
                options
            }
            _ => &[],
        };
        let system = &self.system;
        let uses_features = system.uses_features();
        let names = input.channel_names.as_deref();
        if uses_features && system.channel_projection.is_some() && names.is_none() {
            return Err(input_error(
                "this model was trained with channel names: pass channel_names",
            ));
        }
        let (series, question, prepared) = if uses_features {
            let prepared = features::prepare(
                &input.series,
                question,
                options,
                names,
                input.step,
                input.baseline,
            )?;
            (
                prepared.series.clone(),
                prepared.question.clone(),
                Some(prepared),
            )
        } else {
            (input.series.clone(), question.to_string(), None)
        };
        let (signal, valid, _) = self.encode(&series, &question)?;
        let (tokens, tokens_valid) = self.text.token_states(&Self::texts(&[question.as_str()]))?;
        let takes_names = system.channel_projection.is_some() || !uses_features;
        let channel_embeddings = match names {
            Some(names) if takes_names => Some(self.channel_embeddings(names)?),
            _ => None,
        };
        let device = &self.device;
        let mut channel_features = None;
        let mut referent_question = None;
        let mut referent_options = None;
        if let Some(prepared) = &prepared {
            if self.config.channel_features > 0 {
                let flat: Vec<f32> = prepared
                    .channel_features
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                channel_features = Some(Tensor::from_vec(
                    flat,
                    (1, prepared.channel_features.len(), features::N_FEATURES),
                    device,
                )?);
            }
            if self.config.referents {
                referent_question = Some(Tensor::from_vec(
                    prepared.referent_question.clone(),
                    (1, features::QUESTION),
                    device,
                )?);
                if !prepared.referent_options.is_empty() {
                    let flat: Vec<f32> = prepared
                        .referent_options
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    referent_options = Some(Tensor::from_vec(
                        flat,
                        (1, prepared.referent_options.len(), features::OPTION),
                        device,
                    )?);
                }
            }
        }
        let mut kinds = None;
        if let (Some(summary), Some(names), Some(prepared)) =
            (&system.kind_summary, names, &prepared)
        {
            let patterns: Vec<String> = names.iter().map(|n| kind_pattern(n)).collect();
            let quoted = features::quoted_channels(&question, names);
            kinds = Some(summary.forward(
                &self.channel_embeddings(&patterns)?,
                &prepared.channel_features,
                &quoted,
                device,
            )?);
        }
        let extras = Extras {
            channel_embeddings: channel_embeddings.as_ref(),
            channel_features: channel_features.as_ref(),
            referents: referent_question
                .as_ref()
                .map(|q| (q, referent_options.as_ref())),
            kinds: kinds.as_ref(),
        };
        let option_embeddings = if options.is_empty() {
            None
        } else {
            Some(self.text.embed(options)?)
        };
        let (head, name) = match answers {
            Answers::Options(_) => (Head::Choice(option_embeddings.as_ref()), "choice"),
            Answers::Classes => (Head::Choice(None), "choice"),
            Answers::YesNo => (Head::YesNo, "noul"),
            Answers::Score => (Head::Score, "score"),
        };
        let logits = system.decode(&signal, &valid, &tokens, &tokens_valid, head, &extras)?;
        let temperature = *self.temperatures.get(name).unwrap_or(&1.0);
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format_error("temperature must be finite and positive"));
        }
        let scaled = (&logits / temperature)?;
        Ok(match answers {
            Answers::YesNo => {
                let logit = logits.flatten_all()?.to_vec1::<f32>()?[0];
                let probability = sigmoid(&scaled)?.flatten_all()?.to_vec1::<f32>()?[0];
                Output::YesNo {
                    logit,
                    probability,
                    answer: logit >= 0.0,
                }
            }
            Answers::Score => {
                let probabilities = softmax(&scaled)?.flatten_all()?.to_vec1::<f32>()?;
                let logits = logits.flatten_all()?.to_vec1::<f32>()?;
                Output::Score {
                    best: argmax(&logits),
                    logits,
                    probabilities,
                }
            }
            Answers::Options(_) | Answers::Classes => {
                let probabilities = softmax(&scaled)?.flatten_all()?.to_vec1::<f32>()?;
                let logits = logits.flatten_all()?.to_vec1::<f32>()?;
                let best = argmax(&logits);
                Output::Choice {
                    answer: options.get(best).cloned(),
                    best,
                    logits,
                    probabilities,
                }
            }
        })
    }

    fn locate(&self, input: &Input, question: &str) -> Result<Output> {
        let (head, spec) = self
            .location
            .as_ref()
            .ok_or_else(|| input_error("no location head loaded: pass LoadOptions::location"))?;
        let names = input.channel_names.as_deref();
        let (signal, valid, per_channel) = self.encode(&input.series, question)?;
        let signal = match (&self.system.channel_projection, names) {
            (Some(_), None) => {
                return Err(input_error(
                    "this model was trained with channel names: pass channel_names",
                ));
            }
            (Some(_), Some(names)) => {
                let embeddings = self.channel_embeddings(names)?;
                let extras = Extras {
                    channel_embeddings: Some(&embeddings),
                    channel_features: None,
                    referents: None,
                    kinds: None,
                };
                self.add_names(&signal, &extras)?
            }
            (None, _) => signal,
        };
        let (tokens, tokens_valid) = self.text.token_states(&Self::texts(&[question]))?;
        let logits = head.forward(&signal, &valid, &tokens, &tokens_valid, per_channel)?;
        let probabilities = softmax(&logits)?.flatten_all()?.to_vec1::<f32>()?;
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        let channels = input.series.len();
        let length = input.series[0].len();
        let patch = spec.patch_size;
        let offset = length.saturating_sub(self.series.config.context_length);
        let kept = length - offset;
        let patches = kept.div_ceil(patch);
        if logits.len() - 1 != channels * patches {
            return Err(format_error(
                "the location head needs an encoder with only patch tokens",
            ));
        }
        let pad = patches * patch - kept;
        let decode = |i: usize| {
            let (channel, index) = (i / patches, i % patches);
            Location {
                channel,
                name: names.map(|n| n[channel].clone()),
                patch: index,
                start: offset + (index * patch).saturating_sub(pad),
                end: offset + ((index + 1) * patch - pad).min(kept),
                probability: probabilities[i],
            }
        };
        let best = argmax(&probabilities);
        let location = (best < channels * patches).then(|| decode(best));
        let mut order: Vec<usize> = (0..channels * patches).collect();
        order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
        let candidates = order
            .into_iter()
            .take(LOCATION_CANDIDATES)
            .map(decode)
            .collect();
        Ok(Output::Location {
            null_probability: *probabilities.last().expect("null class"),
            logits,
            location,
            candidates,
        })
    }

    /// The channel-name bias only (the location head was trained without channel features).
    fn add_names(&self, signal: &Tensor, extras: &Extras) -> Result<Tensor> {
        let projection = self
            .system
            .channel_projection
            .as_ref()
            .expect("checked by the caller");
        let bias = projection.forward(extras.channel_embeddings.expect("names"))?;
        let (batch, tokens, dim) = signal.dims3()?;
        let channels = bias.dim(1)?;
        let shaped = signal.reshape((batch, channels, tokens / channels, dim))?;
        Ok(shaped
            .broadcast_add(&bias.unsqueeze(2)?)?
            .reshape((batch, tokens, dim))?)
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// A backbone directory: local, or a Hub repo at a revision.
enum Dirs {
    Local(PathBuf),
    Hub(String, String),
}

impl Dirs {
    fn file(&self, name: &str, cache: Option<&Path>) -> Result<PathBuf> {
        match self {
            Self::Local(dir) => {
                let path = dir.join(name);
                if path.is_file() {
                    Ok(path)
                } else {
                    Err(format_error(format!("{} is missing", path.display())))
                }
            }
            Self::Hub(repo, revision) => hub_file(repo, revision, name, cache),
        }
    }
}
