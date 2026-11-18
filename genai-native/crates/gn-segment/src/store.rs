//! A named tensor store for the hand-written loaders (Hiera, the green-screen keyer): a
//! safetensors file read to host memory, with wrapper prefixes stripped, or (in tests) random
//! tensors of whatever shape the loader asks for.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};

enum Source {
    Map(HashMap<String, Tensor>),
    /// Every lookup succeeds with a small random tensor of the asked shape, recorded under its
    /// name (so a test can rebuild the same model from a map).
    #[cfg_attr(not(test), allow(dead_code))]
    Random(RefCell<HashMap<String, Tensor>>),
}

pub(crate) struct Store {
    src: Source,
    device: Device,
}

impl Store {
    /// Read a safetensors file to host memory and strip wrapper prefixes (`module.`,
    /// `state_dict.`, ...): a first name segment that every tensor shares.
    pub(crate) fn open(path: &Path, device: &Device) -> Result<Store> {
        let raw = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("read {}", path.display()))?;
        ensure!(!raw.is_empty(), "{} holds no tensors", path.display());
        Ok(Store::from_map(raw, device))
    }

    /// A store over named tensors, wrapper prefixes stripped.
    pub(crate) fn from_map(map: HashMap<String, Tensor>, device: &Device) -> Store {
        Store { src: Source::Map(strip_wrappers(map)), device: device.clone() }
    }

    /// A store that invents random weights (for shape tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn random(device: &Device) -> Store {
        Store { src: Source::Random(RefCell::new(HashMap::new())), device: device.clone() }
    }

    pub(crate) fn is_random(&self) -> bool {
        matches!(self.src, Source::Random(_))
    }

    /// What a random store has handed out so far.
    #[cfg(test)]
    pub(crate) fn recorded(&self) -> HashMap<String, Tensor> {
        match &self.src {
            Source::Random(made) => made.borrow().clone(),
            Source::Map(_) => HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn has(&self, name: &str) -> bool {
        match &self.src {
            Source::Map(m) => m.contains_key(name),
            Source::Random(_) => false,
        }
    }

    /// The stored shape of `name`, if present.
    pub(crate) fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match &self.src {
            Source::Map(m) => m.get(name).map(|t| t.dims().to_vec()),
            Source::Random(_) => None,
        }
    }

    /// Every stored name (empty for a random store).
    pub(crate) fn names(&self) -> Vec<String> {
        match &self.src {
            Source::Map(m) => {
                let mut v: Vec<String> = m.keys().cloned().collect();
                v.sort();
                v
            }
            Source::Random(_) => Vec::new(),
        }
    }

    /// The first of `names` the store holds (a random store takes the first).
    pub(crate) fn first_of(&self, names: &[String]) -> Option<String> {
        match &self.src {
            Source::Map(m) => names.iter().find(|n| m.contains_key(n.as_str())).cloned(),
            Source::Random(_) => names.first().cloned(),
        }
    }

    /// `name` as `dtype` on the device, checked to have `shape` (the element count must match;
    /// a 1x1 conv kernel stored as (O, I, 1, 1) reads as (O, I) and the other way round).
    pub(crate) fn get(&self, name: &str, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let t = match &self.src {
            Source::Map(m) => m.get(name).with_context(|| format!("missing tensor `{name}`"))?.clone(),
            Source::Random(made) => {
                let t = if name.ends_with("running_var") {
                    Tensor::rand(0.5f32, 1.5, shape, &Device::Cpu)?
                } else {
                    Tensor::rand(-0.2f32, 0.2, shape, &Device::Cpu)?
                };
                made.borrow_mut().insert(name.to_string(), t.clone());
                t
            }
        };
        let n: usize = shape.iter().product();
        if t.dims() != shape {
            ensure!(t.elem_count() == n, "tensor `{name}` is {:?}, expected {shape:?}", t.dims());
        }
        Ok(t.reshape(shape)?.to_dtype(dtype)?.to_device(&self.device)?)
    }

    /// The first present of `names`, as [`Store::get`].
    pub(crate) fn get_any(&self, names: &[String], shape: &[usize], dtype: DType) -> Result<Tensor> {
        match self.first_of(names) {
            Some(n) => self.get(&n, shape, dtype),
            None => bail!("none of {names:?} in the weights"),
        }
    }

    /// The first present of `names`, or `None` when none is (a random store gives one).
    pub(crate) fn get_opt(&self, names: &[String], shape: &[usize], dtype: DType) -> Result<Option<Tensor>> {
        match self.first_of(names) {
            Some(n) => Ok(Some(self.get(&n, shape, dtype)?)),
            None => Ok(None),
        }
    }
}

/// Strip first name segments every key shares (`module.`, `model.`, `state_dict.`): a wrapper,
/// since a real model has several top-level modules.
fn strip_wrappers(mut map: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    loop {
        let mut first: Option<&str> = None;
        let mut shared = map.len() > 1;
        for k in map.keys() {
            let Some((head, _)) = k.split_once('.') else {
                shared = false;
                break;
            };
            match first {
                None => first = Some(head),
                Some(f) if f == head => {}
                Some(_) => {
                    shared = false;
                    break;
                }
            }
        }
        let Some(head) = first.filter(|_| shared).map(str::to_string) else { return map };
        let cut = head.len() + 1;
        map = map.into_iter().map(|(k, v)| (k[cut..].to_string(), v)).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_wrappers_are_stripped_but_real_modules_are_not() -> Result<()> {
        let t = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
        let map: HashMap<String, Tensor> = ["module.state_dict.encoder.a", "module.state_dict.refiner.b"].iter().map(|k| (k.to_string(), t.clone())).collect();
        let mut keys: Vec<String> = strip_wrappers(map).into_keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["encoder.a".to_string(), "refiner.b".to_string()]);
        Ok(())
    }

    #[test]
    fn one_by_one_kernels_read_either_way() -> Result<()> {
        let mut map = HashMap::new();
        map.insert("w".to_string(), Tensor::zeros((4, 3, 1, 1), DType::F32, &Device::Cpu)?);
        let s = Store::from_map(map, &Device::Cpu);
        assert_eq!(s.get("w", &[4, 3], DType::F32)?.dims(), &[4, 3]);
        assert!(s.get("w", &[4, 2], DType::F32).is_err());
        Ok(())
    }
}
