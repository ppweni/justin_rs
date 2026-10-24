//! Justin in Rust: a time series and questions in, answers out.
//!
//! A port of the Python package `justin` on [candle](https://github.com/huggingface/candle):
//! the Timestral series encoder (any of the four released sizes) with question conditioning
//! (FiLM) and merged LoRA adapters, MiniLM as the text encoder, the cross-attention fusion and the
//! heads (choice, yes/no, score), the feature code the observability models read, and the
//! location heads. Weights come from the files `justin export` and `justin export-location`
//! write, locally or from the Hugging Face Hub; the backbones from their pinned Hub revisions or
//! local directories.
//!
//! ```no_run
//! use justin::{Input, Justin, LoadOptions, Query, Source};
//!
//! let model = Justin::from_pretrained(
//!     Source::local("runs/export/justin.safetensors"),
//!     LoadOptions { location: Some(Source::local("runs/export/justin-location.safetensors")), ..Default::default() },
//! )?;
//! let outputs = model.forward(&Input {
//!     series: vec![vec![0.0; 256], vec![1.0; 256]],
//!     channel_names: Some(vec!["cartservice_cpu".into(), "cartservice_mem".into()]),
//!     queries: vec![
//!         Query::yes_no("Is any metric abnormal?"),
//!         Query::choice("Which service is faulty?", &["cartservice", "No service is faulty"]),
//!         Query::locate("Locate the onset of the spike in cartservice_cpu in this window."),
//!     ],
//!     ..Default::default()
//! })?;
//! # Ok::<(), justin::Error>(())
//! ```

pub mod error;
pub mod features;
pub mod justin;
pub mod location;
pub mod nn;
pub mod source;
pub mod system;
pub mod text;
pub mod timestral;
pub mod weights;

pub use crate::error::{Error, Result};
pub use crate::features::Baseline;
pub use crate::justin::{Answers, Input, Justin, LOCATION_CANDIDATES, LoadOptions, Output, Query};
pub use crate::location::Location;
pub use crate::source::{Backbones, Source};
