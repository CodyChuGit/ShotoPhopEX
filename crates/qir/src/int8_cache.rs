//! On-disk INT8 copies of the Layered-family DiTs (Split's Qwen-Image-Layered and 2511's edit
//! model): the ConvRot INT8 linears a resident load converts from the fp8 file (with 2511's LoRAs
//! merged first), saved once and read back as they are, so a later load skips the fp8 read, the
//! LoRA materialize and the conversion. The entry is keyed by the checkpoint and every LoRA file
//! (path, size, mtime, strength) and [`FORMAT`]; a few entries per model are kept ([`KEEP`]).

use std::path::{Path, PathBuf};

use candle_core::Tensor;

use crate::lora::LoraStack;
use crate::Result;

/// Bumped when the cached tensors' meaning changes (the rotation, the quantizer, the names).
pub const FORMAT: &str = "qir-layered-int8-v1";
/// Entries kept per checkpoint (one per LoRA set); the oldest go first.
pub const KEEP: usize = 3;

/// Whether the cache is used (`QIR_INT8_CACHE=0` turns it off).
pub fn enabled() -> bool {
    std::env::var("QIR_INT8_CACHE").map_or(true, |v| v.trim() != "0")
}

/// The cache folder for `models_root`, as the worker's ConvRot cache: under
/// `%LOCALAPPDATA%\genai-native\qir-cache` where that exists (a ComfyUI `models` folder is not
/// written into), else `<models_root>/.qir-cache`.
pub fn dir(models_root: &Path) -> PathBuf {
    match std::env::var_os("LOCALAPPDATA") {
        Some(d) => PathBuf::from(d).join("genai-native").join("qir-cache"),
        None => models_root.join(".qir-cache"),
    }
}

fn stamp(path: &Path) -> Result<String> {
    let md = std::fs::metadata(path)?;
    let mtime = md.modified()?.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    Ok(format!("{}|{}|{mtime}", path.display(), md.len()))
}

/// The entry for `checkpoint` with `loras` merged: `<dir>/<stem>-int8-<key>.safetensors`.
pub fn entry(dir: &Path, checkpoint: &Path, loras: &LoraStack) -> Result<PathBuf> {
    let mut key = format!("{FORMAT}\n{}", stamp(checkpoint)?);
    for (file, strength) in &loras.layers {
        key.push_str(&format!("\n{}|{:08x}", stamp(&file.path)?, strength.to_bits()));
    }
    let stem = checkpoint.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "dit".into());
    let stem: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(64).collect();
    Ok(dir.join(format!("{stem}-int8-{:016x}.safetensors", crate::convrot_cache::fnv1a64(key.as_bytes()))))
}

/// Save `weights` (CPU tensors) at `path` atomically, then drop all but the newest [`KEEP`]
/// entries of the same checkpoint.
pub fn write(path: &Path, weights: Vec<(String, Tensor)>) -> Result<()> {
    let map: std::collections::HashMap<String, Tensor> = weights.into_iter().collect();
    crate::convrot_cache::write_atomic(path, |tmp| Ok(candle_core::safetensors::save(&map, tmp)?))?;
    prune(path);
    Ok(())
}

/// [`write`] on its own thread, so the request that converted does not wait for the disk; a
/// failure is logged and the next load converts again.
pub fn write_in_background(path: &Path, weights: Vec<(String, Tensor)>) {
    if weights.is_empty() {
        return;
    }
    let path = path.to_path_buf();
    let spawned = std::thread::Builder::new().name("int8-cache-write".into()).spawn(move || {
        let t0 = std::time::Instant::now();
        match write(&path, weights) {
            Ok(()) => tracing::info!(path = %path.display(), write_s = t0.elapsed().as_secs_f64(), "INT8 cache entry written"),
            Err(e) => tracing::warn!(path = %path.display(), error = %e, "INT8 cache write failed"),
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "INT8 cache writer did not start");
    }
}

fn prune(path: &Path) {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name().map(|n| n.to_string_lossy().into_owned())) else { return };
    let Some(prefix) = name.rsplit_once("-int8-").map(|(p, _)| format!("{p}-int8-")) else { return };
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix) && e.file_name().to_string_lossy().ends_with(".safetensors"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.0));
    for (_, old) in entries.into_iter().skip(KEEP) {
        tracing::info!(path = %old.display(), "removing an old INT8 cache entry");
        let _ = std::fs::remove_file(old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("qir-int8-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_entry_follows_the_checkpoint_and_is_stable() {
        let d = tmpdir("entry");
        let ckpt = d.join("model one.safetensors");
        std::fs::write(&ckpt, b"a").unwrap();
        let a = entry(&d, &ckpt, &LoraStack::default()).unwrap();
        assert_eq!(a, entry(&d, &ckpt, &LoraStack::default()).unwrap());
        assert!(a.file_name().unwrap().to_string_lossy().starts_with("model_one-int8-"));
        std::fs::write(&ckpt, b"ab").unwrap();
        assert_ne!(a, entry(&d, &ckpt, &LoraStack::default()).unwrap());
    }

    #[test]
    fn writing_keeps_the_newest_entries_of_one_checkpoint() {
        let d = tmpdir("prune");
        let other = d.join("other-int8-0.safetensors");
        std::fs::write(&other, b"x").unwrap();
        let t = Tensor::zeros(4, candle_core::DType::U8, &candle_core::Device::Cpu).unwrap();
        for i in 0..(KEEP + 2) {
            write(&d.join(format!("dit-int8-{i}.safetensors")), vec![("w".into(), t.clone())]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let left: Vec<_> = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("dit-int8-")).collect();
        assert_eq!(left.len(), KEEP);
        assert!(!left.contains(&"dit-int8-0.safetensors".to_string()));
        assert!(other.exists());
    }
}
