//! ComfyUI's `fp8_scaled` repacks (`comfy_quant` format `float8_e4m3fn`, one `weight_scale`
//! F32 scalar per linear) rewritten as bf16 safetensors, tensor by tensor, so the ordinary
//! loaders read them: Krea 2 Turbo and the Qwen3-VL-4B encoder exist on this machine only in
//! that form (plan Phase 4). Nothing else in the file changes: bf16 and f32 tensors are copied
//! as they are, the `comfy_quant` marker tensors and the consumed scales are dropped.
//!
//! The writer streams: the header is laid out first from the shapes, then each tensor's bytes
//! follow in order, so a 12 GB file needs the memory of one tensor at a time.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::json;

use crate::Result;

/// A converted tensor's place in the output.
struct OutTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: usize,
}

/// What [`dequantize_fp8_file`] did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub dequantized: usize,
    pub copied: usize,
    pub dropped: usize,
}

/// One fp8 tensor's bytes as bf16, `scale` applied (`weight_scale`, ComfyUI's per-tensor
/// dequantisation: `w = fp8.to(bf16) * scale`).
fn fp8_to_bf16(src: &[u8], scale: f32) -> Vec<u8> {
    let mut out = vec![0u8; src.len() * 2];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
    let chunk = src.len().div_ceil(threads).max(1 << 16);
    std::thread::scope(|s| {
        for (i, o) in src.chunks(chunk).zip(out.chunks_mut(chunk * 2)) {
            s.spawn(move || {
                for (b, d) in i.iter().zip(o.as_chunks_mut::<2>().0.iter_mut()) {
                    let v = float8::F8E4M3::from_bits(*b).to_f32() * scale;
                    *d = bf16::from_f32(v).to_le_bytes();
                }
            });
        }
    });
    out
}

/// A scalar F32 tensor's value.
fn scalar_f32(t: &TensorView<'_>) -> Option<f32> {
    if t.dtype() != Dtype::F32 || t.data().len() != 4 {
        return None;
    }
    let b: [u8; 4] = t.data().try_into().ok()?;
    Some(f32::from_le_bytes(b))
}

/// Rewrite `src` as `dst`: fp8 linears dequantised to bf16, the rest copied.
pub fn dequantize_fp8_file(src: &Path, dst: &Path) -> Result<Report> {
    let file = std::fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    // SAFETY: the file is read-only and not modified while mapped (it is a weight file the user
    // placed here); a changed file could only produce a wrong conversion, never memory unsafety
    // beyond what memmap2 documents.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("map {}", src.display()))?;
    let st = safetensors::SafeTensors::deserialize(&mmap).with_context(|| format!("read {}", src.display()))?;
    let names: BTreeMap<String, ()> = st.names().into_iter().map(|n| (n.to_string(), ())).collect();
    let scale_of = |name: &str| -> Option<f32> {
        let base = name.strip_suffix(".weight")?;
        let t = st.tensor(&format!("{base}.weight_scale")).ok()?;
        scalar_f32(&t)
    };
    // Lay the output out: which tensors go, with what dtype and size.
    let mut plan: Vec<OutTensor> = Vec::new();
    let mut report = Report::default();
    for name in names.keys() {
        let t = st.tensor(name).with_context(|| format!("tensor {name}"))?;
        if name.ends_with(".comfy_quant") || name.ends_with(".weight_scale") || name.ends_with(".input_scale") {
            report.dropped += 1;
            continue;
        }
        match t.dtype() {
            Dtype::F8_E4M3 => {
                let Some(_) = scale_of(name) else { anyhow::bail!("{name}: an fp8 tensor without its weight_scale") };
                plan.push(OutTensor { name: name.clone(), dtype: Dtype::BF16, shape: t.shape().to_vec(), bytes: t.data().len() * 2 });
                report.dequantized += 1;
            }
            Dtype::BF16 | Dtype::F32 | Dtype::F16 | Dtype::I64 | Dtype::I32 | Dtype::U8 => {
                plan.push(OutTensor { name: name.clone(), dtype: t.dtype(), shape: t.shape().to_vec(), bytes: t.data().len() });
                report.copied += 1;
            }
            other => anyhow::bail!("{name}: {other:?} is not a dtype this converter copies"),
        }
    }
    // The header: offsets in plan order.
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({"format": "pt", "converted_from": "comfy fp8_scaled", "by": "qwen-image-rs convert-fp8"}));
    let mut offset = 0usize;
    for t in &plan {
        let dtype = match t.dtype {
            Dtype::BF16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::U8 => "U8",
            other => anyhow::bail!("{}: {other:?} is not written", t.name),
        };
        header.insert(t.name.clone(), json!({"dtype": dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes]}));
        offset += t.bytes;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
    // safetensors pads the header to 8 bytes with spaces.
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let out = std::fs::File::create(dst).with_context(|| format!("create {}", dst.display()))?;
    let mut w = BufWriter::with_capacity(1 << 22, out);
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;
    for t in &plan {
        let src_t = st.tensor(&t.name).with_context(|| format!("tensor {}", t.name))?;
        match src_t.dtype() {
            Dtype::F8_E4M3 => {
                let scale = scale_of(&t.name).unwrap_or(1.0);
                w.write_all(&fp8_to_bf16(src_t.data(), scale))?;
            }
            _ => w.write_all(src_t.data())?,
        }
    }
    w.flush()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_bytes_become_scaled_bf16() {
        // e4m3 (bias 7): 1.0 is 0x38 (exponent 0111), 2.0 is 0x40, -1.5 is 0xBC (mantissa 100).
        let out = fp8_to_bf16(&[0x38, 0x40, 0xBC], 0.5);
        let v: Vec<f32> = out.as_chunks::<2>().0.iter().map(|c| bf16::from_le_bytes(*c).to_f32()).collect();
        assert_eq!(v, vec![0.5, 1.0, -0.75]);
    }

    #[test]
    fn a_repack_converts_and_the_markers_go() {
        let d = std::env::temp_dir().join(format!("qir-convert-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        // A tiny "repack": one fp8 linear with its scale and marker, one bf16 bias.
        let w: Vec<u8> = vec![0x38, 0x40, 0xBC, 0x00];
        let scale = 2.0f32.to_le_bytes().to_vec();
        let marker = vec![0u8; 64];
        let bias: Vec<u8> = [bf16::from_f32(3.0), bf16::from_f32(-1.0)].iter().flat_map(|b| b.to_le_bytes()).collect();
        let tensors: Vec<(&str, TensorView<'_>)> = vec![
            ("blocks.0.w.weight", TensorView::new(Dtype::F8_E4M3, vec![2, 2], &w).unwrap_or_else(|e| panic!("{e}"))),
            ("blocks.0.w.weight_scale", TensorView::new(Dtype::F32, vec![], &scale).unwrap_or_else(|e| panic!("{e}"))),
            ("blocks.0.w.comfy_quant", TensorView::new(Dtype::U8, vec![64], &marker).unwrap_or_else(|e| panic!("{e}"))),
            ("blocks.0.w.bias", TensorView::new(Dtype::BF16, vec![2], &bias).unwrap_or_else(|e| panic!("{e}"))),
        ];
        let src = d.join("in.safetensors");
        safetensors::serialize_to_file(tensors, None, &src).unwrap_or_else(|e| panic!("{e}"));
        let dst = d.join("out.safetensors");
        let r = dequantize_fp8_file(&src, &dst).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r, Report { dequantized: 1, copied: 1, dropped: 2 });
        let bytes = std::fs::read(&dst).unwrap_or_else(|e| panic!("{e}"));
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let mut names = st.names();
        names.sort();
        assert_eq!(names, vec!["blocks.0.w.bias", "blocks.0.w.weight"]);
        let t = st.tensor("blocks.0.w.weight").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((t.dtype(), t.shape()), (Dtype::BF16, &[2usize, 2][..]));
        let v: Vec<f32> = t.data().as_chunks::<2>().0.iter().map(|c| bf16::from_le_bytes(*c).to_f32()).collect();
        assert_eq!(v, vec![2.0, 4.0, -3.0, 0.0]);
        let b = st.tensor("blocks.0.w.bias").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(b.data(), &bias[..]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
