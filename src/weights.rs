//! Named tensors from safetensors files, with the metadata Justin stores in its exports.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};

use crate::error::{Result, format_error};

/// A set of named f32 tensors on one device. Every `get` is recorded, so a loader can check that a
/// file holds nothing it did not use (`unused`).
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    used: Mutex<HashSet<String>>,
    device: Device,
}

impl Weights {
    pub fn new(tensors: HashMap<String, Tensor>, device: &Device) -> Self {
        Self {
            tensors,
            used: Mutex::new(HashSet::new()),
            device: device.clone(),
        }
    }

    /// All tensors of a safetensors file as f32 on `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let raw = candle_core::safetensors::load(path, &Device::Cpu)?;
        let mut tensors = HashMap::with_capacity(raw.len());
        for (name, tensor) in raw {
            tensors.insert(name, tensor.to_dtype(DType::F32)?.to_device(device)?);
        }
        Ok(Self::new(tensors, device))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| format_error(format!("missing tensor {name}")))?;
        self.used
            .lock()
            .expect("weights lock")
            .insert(name.to_string());
        Ok(tensor.clone())
    }

    /// Names that start with `prefix`, sorted.
    pub fn names_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Names never read with `get`, sorted.
    pub fn unused(&self) -> Vec<String> {
        let used = self.used.lock().expect("weights lock");
        let mut names: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| !used.contains(*name))
            .cloned()
            .collect();
        names.sort();
        names
    }

    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.tensors.insert(name, tensor);
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    pub fn into_map(self) -> HashMap<String, Tensor> {
        self.tensors
    }
}

/// The string metadata of a safetensors file.
pub fn read_metadata(path: &Path) -> Result<BTreeMap<String, String>> {
    let bytes = std::fs::read(path)?;
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes)?;
    Ok(metadata
        .metadata()
        .clone()
        .map(|m| m.into_iter().collect())
        .unwrap_or_default())
}

/// sha256 over a file's tensors: for every name in sorted order, the name, a zero byte and the
/// tensor's raw bytes. The same digest as `justin.export.fingerprint` in Python.
pub fn fingerprint(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let file = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<String> = file.names().into_iter().map(str::to_string).collect();
    names.sort();
    let mut digest = Sha256::new();
    for name in names {
        digest.update(name.as_bytes());
        digest.update([0u8]);
        digest.update(file.tensor(&name)?.data());
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
