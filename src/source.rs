//! Where weights come from: a local file, or a file of a Hugging Face Hub repo.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// A Justin file (model export or location head).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// `filename` of the Hub model repo `repo` at `revision` (a branch, tag or commit; `main` when
    /// `None`). Read from the local Hugging Face cache when present, else downloaded into it.
    Hub {
        repo: String,
        filename: String,
        revision: Option<String>,
    },
    /// A file on disk.
    Local(PathBuf),
}

impl Source {
    pub fn hub(repo: impl Into<String>, filename: impl Into<String>) -> Self {
        Self::Hub {
            repo: repo.into(),
            filename: filename.into(),
            revision: None,
        }
    }

    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    pub(crate) fn resolve(&self, cache: Option<&Path>) -> Result<PathBuf> {
        match self {
            Self::Local(path) => {
                if path.is_file() {
                    Ok(path.clone())
                } else {
                    Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("{} is not a file", path.display()),
                    )))
                }
            }
            Self::Hub {
                repo,
                filename,
                revision,
            } => hub_file(repo, revision.as_deref().unwrap_or("main"), filename, cache),
        }
    }
}

/// Where the pretrained backbones (Timestral and MiniLM) come from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Backbones {
    /// The Hub repos and pinned revisions recorded in the export.
    #[default]
    Hub,
    /// Local directories: `series` with Timestral's `config.json` and `model.safetensors`, `text`
    /// with MiniLM's `config.json`, `model.safetensors` and `tokenizer.json`.
    Local { series: PathBuf, text: PathBuf },
}

/// The Hugging Face cache: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
pub fn default_cache() -> PathBuf {
    if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return PathBuf::from(path).join("hub");
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// A file of a Hub model repo: from the cache (`refs/<revision>` or a commit snapshot), else
/// downloaded with `hf-hub` into the same cache.
pub fn hub_file(
    repo: &str,
    revision: &str,
    filename: &str,
    cache: Option<&Path>,
) -> Result<PathBuf> {
    let cache = cache.map(Path::to_path_buf).unwrap_or_else(default_cache);
    let folder = cache.join(format!("models--{}", repo.replace('/', "--")));
    let commit = std::fs::read_to_string(folder.join("refs").join(revision))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| revision.to_string());
    let cached = folder.join("snapshots").join(&commit).join(filename);
    if cached.is_file() {
        return Ok(cached);
    }
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache)
        .build()
        .map_err(|e| Error::Hub(e.to_string()))?;
    let repo = hf_hub::Repo::with_revision(
        repo.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    );
    api.repo(repo)
        .get(filename)
        .map_err(|e| Error::Hub(format!("{filename}: {e}")))
}
