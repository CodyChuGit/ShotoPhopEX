//! A [`VarBuilder`] over safetensors files mapped once and faulted in on every core before the
//! upload. candle's mapped VarBuilder touched each page as it copied the tensor out: 16.3 GB of
//! Qwen3-VL 8B reached the card at 3.4 GB/s (5.0 s), soft faults on pages already in the file
//! cache; faulted in on 8 threads first (0.8 s) and uploaded straight from the mapping (1.55 s),
//! 2.3 s (`tests/load_bench.rs`, RTX 5090, 2026-11-01). The bytes on the card are the same.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;

use crate::layered_files::TensorFile;
use crate::Result;

/// Threads that fault the mapping in (a share of the pages each).
fn fault_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16)
}

struct MappedSet {
    files: Vec<TensorFile>,
    /// Which file holds each tensor.
    index: HashMap<String, usize>,
}

impl candle_nn::var_builder::SimpleBackend for MappedSet {
    fn get(&self, shape: Shape, name: &str, _: candle_nn::Init, dt: DType, device: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dt, device)?;
        if t.shape() != &shape {
            candle_core::bail!("{name}: expected {shape:?}, got {:?}", t.shape());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dt: DType, device: &Device) -> candle_core::Result<Tensor> {
        let file = self.index.get(name).and_then(|i| self.files.get(*i)).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() })?;
        file.tensor(name, device).map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?.to_dtype(dt)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

/// `files` as a [`VarBuilder`] of `dt` on `dev`: mapped, the tensors `keep` names (what the
/// caller will read) faulted in on every core, uploaded from the mapping. candle's own mapped
/// VarBuilder when a file is one [`TensorFile`] does not read (a dtype it has no name for), so a
/// load never fails for the faster route's sake.
pub fn var_builder<P: AsRef<Path>>(files: &[P], dt: DType, dev: &Device, keep: impl Fn(&str) -> bool) -> Result<VarBuilder<'static>> {
    let opened: std::result::Result<Vec<TensorFile>, _> = files.iter().map(|p| TensorFile::open(p.as_ref())).collect();
    let files_vb = match opened {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(error = %e, "mapped loader: not a plain safetensors set; candle's VarBuilder");
            let paths: Vec<&Path> = files.iter().map(|p| p.as_ref()).collect();
            return Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dt, dev)? });
        }
    };
    let threads = fault_threads();
    for f in &files_vb {
        f.prefault_where(&keep, threads);
    }
    let mut index = HashMap::new();
    for (i, f) in files_vb.iter().enumerate() {
        for n in f.names() {
            index.insert(n, i);
        }
    }
    Ok(VarBuilder::from_backend(Box::new(MappedSet { files: files_vb, index }), dt, dev.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, tensors: &[(&str, Tensor)]) {
        let map: HashMap<String, Tensor> = tensors.iter().map(|(n, t)| (n.to_string(), t.clone())).collect();
        candle_core::safetensors::save(&map, path).unwrap();
    }

    #[test]
    fn the_tensors_are_candles() {
        let d = std::env::temp_dir().join(format!("qir-mapped-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (a, b) = (d.join("a.safetensors"), d.join("b.safetensors"));
        let x = Tensor::arange(0f32, 24.0, &Device::Cpu).unwrap().reshape((4, 6)).unwrap().to_dtype(DType::BF16).unwrap();
        let y = Tensor::arange(0f32, 5.0, &Device::Cpu).unwrap();
        write(&a, &[("m.x", x)]);
        write(&b, &[("m.y", y)]);
        let ours = var_builder(&[&a, &b], DType::F32, &Device::Cpu, |n| n.starts_with("m.")).unwrap();
        let theirs = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&a, &b], DType::F32, &Device::Cpu).unwrap() };
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(ours.get((4, 6), "m.x").unwrap()), flat(theirs.get((4, 6), "m.x").unwrap()));
        assert_eq!(flat(ours.get(5, "m.y").unwrap()), flat(theirs.get(5, "m.y").unwrap()));
        assert!(ours.contains_tensor("m.y") && !ours.contains_tensor("m.z"));
        assert!(ours.get((4, 6), "m.z").is_err());
        assert!(ours.get((6, 4), "m.x").is_err(), "a wrong shape is refused");
    }
}
