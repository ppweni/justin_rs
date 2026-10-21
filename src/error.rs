//! The crate's error type.

/// Everything that can go wrong while loading a model or answering.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("hub: {0}")]
    Hub(String),
    /// A file that is not what it claims to be, or does not fit the model it is loaded into.
    #[error("{0}")]
    Format(String),
    /// Inputs the model cannot answer (wrong shapes, missing channel names, unknown head).
    #[error("{0}")]
    Input(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn format_error(message: impl Into<String>) -> Error {
    Error::Format(message.into())
}

pub(crate) fn input_error(message: impl Into<String>) -> Error {
    Error::Input(message.into())
}
