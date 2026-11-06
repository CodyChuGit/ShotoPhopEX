//! Q8_0 quantization on every CPU core. candle's `QTensor::quantize` of a CUDA tensor copies it
//! to the host as f32 and quantizes it on one thread: the chat head (151,936 × 4,096, the text
//! encoder's `lm_head`) took 10.2 s to load that way, on every first Remove after another model
//! family had the card (measured in PhotoshopEX 2026-11-01). Here the weight is read on the CPU,
//! split by rows across the cores, each chunk quantized with candle's own `BlockQ8_0::from_float`
//! (so the blocks are the bytes candle makes), and uploaded once.
//!
//! [`cached_q8_0`] keeps those blocks on disk next to the INT8 cache, so a later load reads
//! 0.66 GB instead of reading and quantizing the 1.2 GB bf16 head (1.35 s a swap-back Remove).

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::quantized::k_quants::BlockQ8_0;
use candle_core::quantized::{GgmlDType, GgmlType, QStorage, QTensor};
use candle_core::{DType, Device, Tensor};

use crate::Result;

/// Bumped when the cached blocks' meaning changes (the quantizer, the layout).
pub const FORMAT: &str = "qir-q8_0-v1";

/// Whether the head cache is used (`QIR_Q8_CACHE=0` turns it off).
pub fn cache_enabled() -> bool {
    std::env::var("QIR_Q8_CACHE").map_or(true, |v| v.trim() != "0")
}

/// The entry for `name` of `files` at `shape`: `<dir>/<stem>-q8-<key>.bin`, keyed by every file
/// (path, size, mtime), the name, the shape and [`FORMAT`].
pub fn cache_entry(dir: &Path, files: &[PathBuf], name: &str, shape: (usize, usize)) -> Result<PathBuf> {
    let mut key = format!("{FORMAT}\n{name}\n{}x{}", shape.0, shape.1);
    for f in files {
        key.push('\n');
        key.push_str(&crate::int8_cache::stamp(f)?);
    }
    let stem = files.first().and_then(|f| f.file_stem()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "head".into());
    let stem: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(64).collect();
    Ok(dir.join(format!("{stem}-q8-{:016x}.bin", crate::convrot_cache::fnv1a64(key.as_bytes()))))
}

/// The Q8_0 bytes of a `(rows, cols)` matrix.
fn q8_len(shape: (usize, usize)) -> Option<usize> {
    shape.0.checked_mul(shape.1 / BlockQ8_0::BLCK_SIZE)?.checked_mul(std::mem::size_of::<BlockQ8_0>())
}

/// `name` (a `(rows, cols)` matrix) of the safetensors `files` as Q8_0 on `dev`: the blocks
/// from the on-disk entry when there is one ([`cache_entry`], under
/// [`crate::int8_cache::dir`]), else read and quantized on every core ([`quantize_q8_0`]) and
/// saved for next time. The blocks are the same either way; a cache that cannot be read or
/// written only costs the quantize.
pub fn cached_q8_0(files: &[PathBuf], name: &str, shape: (usize, usize), dtype: DType, dev: &Device) -> Result<QTensor> {
    let dir = files.first().and_then(|f| f.parent()).filter(|_| cache_enabled()).map(crate::int8_cache::dir);
    let min_free = std::env::var("QIR_INT8_CACHE_MIN_FREE_GB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(crate::int8_cache::MIN_FREE_GB) << 30;
    cached_q8_0_in(dir.as_deref(), min_free, files, name, shape, dtype, dev)
}

/// [`cached_q8_0`] with its entries in `dir` (`None`: no cache), a write leaving `min_free` bytes
/// free.
fn cached_q8_0_in(dir: Option<&Path>, min_free: u64, files: &[PathBuf], name: &str, shape: (usize, usize), dtype: DType, dev: &Device) -> Result<QTensor> {
    let entry = dir.and_then(|d| cache_entry(d, files, name, shape).ok());
    if let Some(entry) = &entry {
        match read_entry(entry, shape, dev) {
            Ok(Some(q)) => return Ok(q),
            Ok(None) => {}
            Err(e) => tracing::warn!(path = %entry.display(), error = %format!("{e:#}"), "Q8_0 cache entry unreadable; quantizing"),
        }
    }
    let host = crate::mapped::var_builder(files, dtype, &Device::Cpu, |n| n == name)?;
    let w = host.get(shape, name).with_context(|| format!("{name} not found"))?;
    let cpu = quantize_q8_0(&w, &Device::Cpu)?;
    let bytes = cpu.data()?;
    if let Some(entry) = &entry {
        if let Err(e) = write_entry(entry, &bytes, min_free) {
            tracing::warn!(path = %entry.display(), error = %format!("{e:#}"), "Q8_0 cache entry not written");
        }
    }
    if dev.is_cpu() {
        drop(bytes);
        return Ok(cpu);
    }
    Ok(QTensor::new(QStorage::from_data(bytes, dev, GgmlDType::Q8_0)?, shape)?)
}

fn read_entry(entry: &Path, shape: (usize, usize), dev: &Device) -> Result<Option<QTensor>> {
    if !entry.is_file() {
        return Ok(None);
    }
    let want = q8_len(shape).context("Q8_0 size overflow")?;
    let bytes = std::fs::read(entry)?;
    anyhow::ensure!(bytes.len() == want, "{} bytes, want {want}", bytes.len());
    tracing::info!(path = %entry.display(), "Q8_0 cache entry read");
    Ok(Some(QTensor::new(QStorage::from_data(Cow::Owned(bytes), dev, GgmlDType::Q8_0)?, shape)?))
}

/// Write `bytes` at `entry` atomically, leaving the disk `min_free` bytes free (the INT8 cache's
/// floor, `QIR_INT8_CACHE_MIN_FREE_GB`), and remove the other entries of the same file stem.
fn write_entry(entry: &Path, bytes: &[u8], min_free: u64) -> Result<()> {
    let dir = entry.parent().context("Q8_0 cache entry has no folder")?;
    std::fs::create_dir_all(dir)?;
    let free = fs2::available_space(dir)?;
    anyhow::ensure!(free.saturating_sub(bytes.len() as u64) >= min_free, "too little free space ({} GiB)", free >> 30);
    crate::convrot_cache::write_atomic(entry, |tmp| Ok(std::fs::write(tmp, bytes)?))?;
    let name = entry.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let (Some((stem, _)), Ok(rd)) = (name.rsplit_once("-q8-"), std::fs::read_dir(dir)) {
        for e in rd.flatten() {
            let other = e.file_name().to_string_lossy().into_owned();
            if other != name && other.ends_with(".bin") && other.rsplit_once("-q8-").is_some_and(|(s, _)| s == stem) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    tracing::info!(path = %entry.display(), bytes = bytes.len(), "Q8_0 cache entry written");
    Ok(())
}

/// `w` (a 2-D tensor on the CPU, any float dtype) as a Q8_0 [`QTensor`] on `dev`, identical to
/// `QTensor::quantize(w, GgmlDType::Q8_0)`.
pub fn quantize_q8_0(w: &Tensor, dev: &Device) -> Result<QTensor> {
    let (rows, cols) = w.dims2()?;
    anyhow::ensure!(cols.is_multiple_of(BlockQ8_0::BLCK_SIZE), "Q8_0 needs rows of whole {}-value blocks, got {cols}", BlockQ8_0::BLCK_SIZE);
    let w = w.to_device(&Device::Cpu)?;
    let per_row = cols / BlockQ8_0::BLCK_SIZE;
    let mut blocks = vec![BlockQ8_0::zeros(); rows * per_row];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32);
    let rows_per = rows.div_ceil(threads).max(1);
    if w.dtype() == DType::BF16 {
        // The checkpoint's bf16 as it is; each worker widens its own rows (exactly, as the
        // f32 cast does) one at a time.
        let src = w.flatten_all()?.to_vec1::<half::bf16>()?;
        std::thread::scope(|s| {
            for (from, to) in src.chunks(rows_per * cols).zip(blocks.chunks_mut(rows_per * per_row)) {
                s.spawn(move || {
                    let mut row = vec![0f32; cols];
                    for (r, b) in from.chunks_exact(cols).zip(to.chunks_exact_mut(per_row)) {
                        for (o, v) in row.iter_mut().zip(r) {
                            *o = v.to_f32();
                        }
                        BlockQ8_0::from_float(&row, b);
                    }
                });
            }
        });
    } else {
        let src = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        std::thread::scope(|s| {
            for (from, to) in src.chunks(rows_per * cols).zip(blocks.chunks_mut(rows_per * per_row)) {
                s.spawn(move || BlockQ8_0::from_float(from, to));
            }
        });
    }
    let cpu = QTensor::new(QStorage::Cpu(Box::new(blocks)), (rows, cols))?;
    if dev.is_cpu() {
        return Ok(cpu);
    }
    Ok(QTensor::new(QStorage::from_data(cpu.data()?, dev, GgmlDType::Q8_0)?, (rows, cols))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blocks_are_candles() {
        // Rows of several blocks, more rows than threads, values over a wide range.
        let (rows, cols) = (97, 160);
        let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 13) as f32)).collect();
        let w = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap();
        let ours = quantize_q8_0(&w, &Device::Cpu).unwrap();
        let theirs = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        assert_eq!(ours.shape(), theirs.shape());
        assert_eq!(ours.data().unwrap(), theirs.data().unwrap());
    }

    #[test]
    fn f32_weights_take_the_same_blocks() {
        let (rows, cols) = (5, 64);
        let v: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 150.0) / 7.0).collect();
        let w = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap();
        let ours = quantize_q8_0(&w, &Device::Cpu).unwrap();
        assert_eq!(ours.data().unwrap(), QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap());
    }

    #[test]
    fn the_cached_blocks_are_the_quantized_ones() {
        let d = std::env::temp_dir().join(format!("qir-q8-cache-{}", std::process::id()));
        let cache = d.join("cache");
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("te.safetensors");
        let (rows, cols) = (9, 64);
        let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.21).cos() * (1.0 + (i % 7) as f32)).collect();
        let w = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap();
        let save = |w: &Tensor| candle_core::safetensors::save(&std::collections::HashMap::from([("h.weight".to_string(), w.clone())]), &file).unwrap();
        save(&w);
        let files = vec![file.clone()];
        let want = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap().into_owned();
        let load = || cached_q8_0_in(Some(&cache), 0, &files, "h.weight", (rows, cols), DType::BF16, &Device::Cpu).unwrap().data().unwrap().into_owned();
        // A miss quantizes and writes the entry; a hit reads it back as it is.
        assert_eq!(load(), want);
        let entry = cache_entry(&cache, &files, "h.weight", (rows, cols)).unwrap();
        assert_eq!(std::fs::read(&entry).unwrap(), want);
        let marked: Vec<u8> = want.iter().map(|b| b ^ 1).collect();
        std::fs::write(&entry, &marked).unwrap();
        assert_eq!(load(), marked, "a hit is the entry's bytes");
        // A short entry is quantized again (and rewritten); so is a changed file, its old entry removed.
        std::fs::write(&entry, &want[1..]).unwrap();
        assert_eq!(load(), want);
        save(&(w * 2.0).unwrap());
        let again = cache_entry(&cache, &files, "h.weight", (rows, cols)).unwrap();
        assert_ne!(again, entry);
        let _ = load();
        assert!(again.is_file() && !entry.is_file());
        // No cache, a missing tensor: an error, no panic.
        assert!(cached_q8_0_in(None, 0, &files, "nope", (rows, cols), DType::BF16, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_ragged_row_is_refused() {
        let w = Tensor::zeros((4, 33), DType::F32, &Device::Cpu).unwrap();
        assert!(quantize_q8_0(&w, &Device::Cpu).is_err());
    }
}
