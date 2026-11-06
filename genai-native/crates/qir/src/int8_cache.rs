//! On-disk INT8 copies of the Layered-family DiTs (Split's Qwen-Image-Layered and 2511's edit
//! model): the ConvRot INT8 linears a resident load converts from the fp8 file (with 2511's LoRAs
//! merged first), saved once and read back as they are, so a later load skips the fp8 read, the
//! LoRA materialize and the conversion. The entry is keyed by the checkpoint and every LoRA file
//! (path, size, mtime, strength) and [`FORMAT`]. An entry is ~19 GB: one per model is kept
//! ([`KEEP`]), all of them within [`MAX_GB`], and none is written that would leave the disk with
//! less than [`MIN_FREE_GB`] free.

use std::path::{Path, PathBuf};

use candle_core::Tensor;

use crate::lora::LoraStack;
use crate::Result;

/// Bumped when the cached tensors' meaning changes (the rotation, the quantizer, the names).
pub const FORMAT: &str = "qir-layered-int8-v1";
/// Entries kept per checkpoint, one per LoRA set (`QIR_INT8_CACHE_KEEP`).
pub const KEEP: usize = 1;
/// Total size of all entries in GiB; the least recently used go first (`QIR_INT8_CACHE_MAX_GB`).
pub const MAX_GB: u64 = 60;
/// Free space in GiB a write must leave on the disk (`QIR_INT8_CACHE_MIN_FREE_GB`).
pub const MIN_FREE_GB: u64 = 30;

/// Whether the cache is used (`QIR_INT8_CACHE=0` turns it off).
pub fn enabled() -> bool {
    std::env::var("QIR_INT8_CACHE").map_or(true, |v| v.trim() != "0")
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
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

pub(crate) fn stamp(path: &Path) -> Result<String> {
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

/// Save `weights` (CPU tensors) at `path` atomically and evict ([`evictions`]); `false` when it
/// was not written because the disk would be left with less than [`MIN_FREE_GB`] free.
pub fn write(path: &Path, weights: Vec<(String, Tensor)>) -> Result<bool> {
    let size: u64 = weights.iter().map(|(_, t)| (t.elem_count() * t.dtype().size_in_bytes()) as u64).sum();
    let dir = path.parent().ok_or_else(|| anyhow::anyhow!("INT8 cache entry has no folder"))?;
    std::fs::create_dir_all(dir)?;
    let free = fs2::available_space(dir)?;
    let min_free = env_or("QIR_INT8_CACHE_MIN_FREE_GB", MIN_FREE_GB) << 30;
    if free.saturating_sub(size) < min_free {
        tracing::info!(path = %path.display(), free_gib = free >> 30, entry_gib = size >> 30, min_free_gib = min_free >> 30, "INT8 cache entry not written: too little free space");
        return Ok(false);
    }
    let map: std::collections::HashMap<String, Tensor> = weights.into_iter().collect();
    crate::convrot_cache::write_atomic(path, |tmp| Ok(candle_core::safetensors::save(&map, tmp)?))?;
    prune(path);
    Ok(true)
}


/// Mark an entry as just used, so eviction takes the least recently used first.
pub fn touch(path: &Path) {
    if let Ok(f) = std::fs::File::options().write(true).open(path) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

fn checkpoint_of(path: &Path) -> Option<String> {
    path.file_name()?.to_string_lossy().rsplit_once("-int8-").map(|(stem, _)| stem.to_string())
}

/// The entries to delete once `newest` is written, from `(last used, path, bytes)`: past `keep`
/// of its checkpoint, then the least recently used of the rest until all fit in `cap` bytes.
/// `newest` always stays.
fn evictions(mut entries: Vec<(std::time::SystemTime, PathBuf, u64)>, newest: &Path, keep: usize, cap: u64) -> Vec<PathBuf> {
    let own = checkpoint_of(newest);
    entries.sort_by_key(|e| std::cmp::Reverse(e.0));
    let mut total = entries.iter().find(|e| e.1 == newest).map_or(0, |e| e.2);
    let mut kept_own = 1;
    let mut out = Vec::new();
    for (_, path, len) in entries {
        if path == newest {
            continue;
        }
        let same = own.is_some() && checkpoint_of(&path) == own;
        if (same && kept_own >= keep) || total + len > cap {
            out.push(path);
            continue;
        }
        kept_own += usize::from(same);
        total += len;
    }
    out
}

fn prune(newest: &Path) {
    let Some(dir) = newest.parent() else { return };
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let entries = rd
        .flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.contains("-int8-") && name.ends_with(".safetensors")
        })
        .filter_map(|e| {
            let md = e.metadata().ok()?;
            Some((md.modified().ok()?, e.path(), md.len()))
        })
        .collect();
    let keep = env_or("QIR_INT8_CACHE_KEEP", KEEP).max(1);
    let cap = env_or("QIR_INT8_CACHE_MAX_GB", MAX_GB) << 30;
    for old in evictions(entries, newest, keep, cap) {
        tracing::info!(path = %old.display(), "removing an INT8 cache entry");
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
    fn eviction_keeps_per_checkpoint_then_fits_the_cap_by_last_use() {
        let at = |s: u64| std::time::UNIX_EPOCH + std::time::Duration::from_secs(s);
        let p = |n: &str| PathBuf::from(format!("{n}.safetensors"));
        let entries = vec![
            (at(1), p("a-int8-1"), 19),
            (at(5), p("a-int8-2"), 19),
            (at(4), p("b-int8-1"), 19),
            (at(2), p("c-int8-1"), 19),
            (at(3), p("d-int8-1"), 19),
        ];
        // a-int8-2 is new: the older a goes (one per checkpoint), then 60 fits a-2, b and d.
        let mut out = evictions(entries.clone(), &p("a-int8-2"), 1, 60);
        out.sort();
        assert_eq!(out, vec![p("a-int8-1"), p("c-int8-1")]);
        // Two per checkpoint and room for all: nothing goes.
        assert!(evictions(entries.clone(), &p("a-int8-2"), 2, 1000).is_empty());
        // The new entry stays even when it alone is over the cap.
        assert_eq!(evictions(vec![(at(1), p("a-int8-1"), 99)], &p("a-int8-1"), 1, 60), Vec::<PathBuf>::new());
    }

    #[test]
    fn writing_replaces_the_checkpoints_older_entry() {
        let d = tmpdir("prune");
        let other = d.join("other-int8-0.safetensors");
        std::fs::write(&other, b"x").unwrap();
        let t = Tensor::zeros(4, candle_core::DType::U8, &candle_core::Device::Cpu).unwrap();
        // The free-space guard applies to the test folder's disk too.
        let free = fs2::available_space(&d).unwrap();
        for i in 0..3 {
            let written = write(&d.join(format!("dit-int8-{i}.safetensors")), vec![("w".into(), t.clone())]).unwrap();
            assert_eq!(written, free >= MIN_FREE_GB << 30);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let left: Vec<_> = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("dit-int8-")).collect();
        if free >= MIN_FREE_GB << 30 {
            assert_eq!(left, vec!["dit-int8-2.safetensors".to_string()]);
        }
        assert!(other.exists());
    }
}
