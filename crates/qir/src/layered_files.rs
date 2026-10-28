//! Layered component inventory and biased ComfyUI FP8 linears.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

use crate::fp8::{Fp8Linear, Fp8Plan};
use crate::Result;

/// Explicit filenames avoid accidentally selecting a Qwen-Image-2.1 component.
#[derive(Debug, Clone)]
pub struct LayeredFiles {
    pub diffusion: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

impl LayeredFiles {
    pub fn resolve(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let files = Self {
            diffusion: root.join("diffusion_models/qwen_image_layered_fp8mixed.safetensors"),
            text_encoder: root.join("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            vae: root.join("vae/qwen_image_layered_vae.safetensors"),
        };
        files.validate()?;
        Ok(files)
    }

    /// Check the installed repack's inventory without reading its multi-GB payload.
    pub fn validate(&self) -> Result<()> {
        let dit = TensorFile::open(&self.diffusion)?;
        dit.inventory(&[("BF16", 1095), ("F32", 839), ("F8_E4M3", 839), ("U8", 839)])?;
        for (name, shape) in [
            ("img_in.weight", vec![3072, 64]),
            ("proj_out.weight", vec![64, 3072]),
            ("txt_norm.weight", vec![3584]),
            ("time_text_embed.addition_t_embedding.weight", vec![2, 3072]),
        ] {
            dit.check(name, &shape, "BF16")?;
        }
        for li in 0..60 {
            for stream in ["img", "txt"] {
                dit.check(
                    &format!("transformer_blocks.{li}.{stream}_mod.1.weight"),
                    &[18432, 3072],
                    if stream == "img" && li == 0 { "BF16" } else { "F8_E4M3" },
                )?;
            }
        }
        dit.scales("weight_scale", 839)?;
        let text = TensorFile::open(&self.text_encoder)?;
        text.inventory(&[("BF16", 371), ("F32", 716), ("F8_E4M3", 359)])?;
        text.check("model.embed_tokens.weight", &[152064, 3584], "BF16")?;
        text.check("model.norm.weight", &[3584], "BF16")?;
        text.check("scaled_fp8", &[0], "F8_E4M3")?;
        for li in 0..28 {
            let p = format!("model.layers.{li}");
            for (name, shape) in [
                ("self_attn.q_proj", [3584, 3584]),
                ("self_attn.k_proj", [512, 3584]),
                ("self_attn.v_proj", [512, 3584]),
                ("self_attn.o_proj", [3584, 3584]),
                ("mlp.gate_proj", [18944, 3584]),
                ("mlp.up_proj", [18944, 3584]),
                ("mlp.down_proj", [3584, 18944]),
            ] {
                text.check(&format!("{p}.{name}.weight"), &shape, "F8_E4M3")?;
            }
            for (name, width) in [("q_proj", 3584), ("k_proj", 512), ("v_proj", 512)] {
                text.check(&format!("{p}.self_attn.{name}.bias"), &[width], "BF16")?;
            }
            for name in ["input_layernorm", "post_attention_layernorm"] {
                text.check(&format!("{p}.{name}.weight"), &[3584], "BF16")?;
            }
        }
        text.scales("scale_weight", 358)?;
        let vae = TensorFile::open(&self.vae)?;
        vae.inventory(&[("BF16", 194)])?;
        for (name, shape) in
            [("encoder.conv1.weight", vec![96, 4, 3, 3, 3]), ("decoder.conv1.weight", vec![384, 16, 3, 3, 3]), ("decoder.head.2.weight", vec![4, 96, 3, 3, 3])]
        {
            vae.check(name, &shape, "BF16")?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// A validated read-only map avoids per-tensor opens and intermediate payload buffers.
pub struct TensorFile {
    path: PathBuf,
    start: u64,
    tensors: BTreeMap<String, TensorInfo>,
    map: memmap2::Mmap,
}

impl TensorFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        anyhow::ensure!(len < 256 << 20, "{}: oversized header", path.display());
        let mut bytes = vec![0; len as usize];
        file.read_exact(&mut bytes)?;
        let mut header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
        header.remove("__metadata__");
        let tensors: BTreeMap<String, TensorInfo> = serde_json::from_value(header.into())?;
        let size = file.metadata()?.len();
        let mut ranges = Vec::new();
        for (name, info) in &tensors {
            let [a, b] = info.data_offsets;
            let width = dtype(&info.dtype)?.size_in_bytes();
            let elements = info.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d).ok_or_else(|| anyhow::anyhow!("{name}: shape overflow")))?;
            let payload = elements.checked_mul(width).ok_or_else(|| anyhow::anyhow!("{name}: size overflow"))? as u64;
            anyhow::ensure!(b >= a && b - a == payload && b <= size.saturating_sub(8 + len), "{name}: invalid offsets");
            ranges.push((a, b));
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (a, b) in ranges {
            anyhow::ensure!(a == end, "{}: overlapping or noncontiguous payload", path.display());
            end = b;
        }
        anyhow::ensure!(end == size.saturating_sub(8 + len), "{}: trailing payload", path.display());
        // SAFETY: installed checkpoints must remain immutable for the lifetime of this reader.
        let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(Self { path: path.to_path_buf(), start: 8 + len, tensors, map })
    }

    /// Resolve mapping faults while the VAE and text encoder use the GPU.
    pub(crate) fn prefetch(&self, stop: &std::sync::atomic::AtomicBool) {
        let mut touched = 0u8;
        for page in self.map.chunks(4096) {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            touched ^= page[0];
        }
        std::hint::black_box(touched);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        let a = usize::try_from(self.start + info.data_offsets[0])?;
        let b = usize::try_from(self.start + info.data_offsets[1])?;
        let bytes = self.map.get(a..b).ok_or_else(|| anyhow::anyhow!("{name}: payload outside mapping"))?;
        Ok(Tensor::from_raw_buffer(bytes, dtype(&info.dtype)?, &info.shape, device)?)
    }

    /// A block's raw tensors plus ComfyUI's legacy scale aliases and marker metadata.
    pub fn builder(&self, prefix: &str, dt: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let legacy_full = self.tensors.get("scaled_fp8").is_some_and(|v| v.shape.iter().product::<usize>() == 2);
        for name in self.tensors.keys().filter(|n| n.starts_with(prefix)) {
            let short = name.strip_prefix(prefix).ok_or_else(|| anyhow::anyhow!("prefix mismatch"))?;
            let host = short.ends_with(".scale_input") || (short.ends_with(".scale_weight") && self.tensors[name].shape.iter().product::<usize>() == 1);
            let value = self.tensor(name, if host { &Device::Cpu } else { device })?;
            let mapped = short.replace(".scale_weight", ".weight_scale").replace(".scale_input", ".input_scale");
            if short.ends_with(".scale_weight") {
                let layer = short.trim_end_matches(".scale_weight");
                let record = serde_json::to_vec(&serde_json::json!({"format":"float8_e4m3fn", "full_precision_matrix_mult":legacy_full}))?;
                tensors.insert(format!("{layer}.comfy_quant"), Tensor::from_vec(record.clone(), record.len(), &Device::Cpu)?);
            }
            if short.ends_with(".scale_input") && value.flatten_all()?.to_vec1::<f32>()? == [1.0] {
                continue;
            }
            tensors.insert(mapped, value);
        }
        Ok(VarBuilder::from_tensors(tensors, dt, device))
    }

    /// Upload a contiguous FP8 region once; individual matrices become zero-copy device views.
    pub fn resident_builder(self, dt: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let mut ranges: Vec<_> = self.tensors.values().filter(|info| info.dtype == "F8_E4M3").map(|info| info.data_offsets).collect();
        ranges.sort_unstable();
        let packed = match (ranges.first(), ranges.last()) {
            (Some(first), Some(last)) if ranges.windows(2).all(|pair| pair[0][1] == pair[1][0]) => {
                let start = first[0];
                let a = usize::try_from(self.start + start)?;
                let b = usize::try_from(self.start + last[1])?;
                let bytes = self.map.get(a..b).ok_or_else(|| anyhow::anyhow!("FP8 payload outside mapping"))?;
                Some((start, Tensor::from_raw_buffer(bytes, DType::F8E4M3, &[bytes.len()], device)?))
            }
            _ => None,
        };
        Ok(VarBuilder::from_backend(Box::new(ResidentFile { file: self, packed }), dt, device.clone()))
    }

    fn check(&self, name: &str, shape: &[usize], dt: &str) -> Result<()> {
        let t = self.tensors.get(name).ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        anyhow::ensure!(t.shape == shape && t.dtype == dt, "{name}: expected {dt} {shape:?}, got {} {:?}", t.dtype, t.shape);
        Ok(())
    }

    fn inventory(&self, expected: &[(&str, usize)]) -> Result<()> {
        let mut counts = BTreeMap::new();
        for t in self.tensors.values() {
            *counts.entry(t.dtype.as_str()).or_insert(0usize) += 1;
        }
        anyhow::ensure!(counts == expected.iter().copied().collect(), "{}: unexpected dtype inventory {counts:?}", self.path.display());
        Ok(())
    }

    fn scales(&self, suffix: &str, count: usize) -> Result<()> {
        let mut found = 0;
        for (name, info) in &self.tensors {
            if let Some(layer) = name.strip_suffix(&format!(".{suffix}")) {
                let weight = self.tensors.get(&format!("{layer}.weight")).ok_or_else(|| anyhow::anyhow!("{name}: missing weight"))?;
                anyhow::ensure!(weight.dtype == "F8_E4M3" && weight.shape.len() == 2, "{name}: expected E4M3 matrix");
                anyhow::ensure!(
                    info.dtype == "F32"
                        && (info.shape.is_empty() || info.shape == [1] || info.shape == [weight.shape[0]] || info.shape == [weight.shape[0], 1]),
                    "{name}: invalid scale"
                );
                if suffix == "weight_scale" {
                    let record = self.tensor(&format!("{layer}.comfy_quant"), &Device::Cpu)?.to_vec1::<u8>()?;
                    let record: serde_json::Value = serde_json::from_slice(&record)?;
                    anyhow::ensure!(
                        record["format"] == "float8_e4m3fn" && record["full_precision_matrix_mult"] == true,
                        "{layer}: unexpected quantization record"
                    );
                }
                found += 1;
            }
        }
        anyhow::ensure!(found == count, "{}: expected {count} scales, got {found}", self.path.display());
        Ok(())
    }
}

struct ResidentFile {
    file: TensorFile,
    packed: Option<(u64, Tensor)>,
}

impl candle_nn::var_builder::SimpleBackend for ResidentFile {
    fn get(&self, shape: candle_core::Shape, name: &str, _: candle_nn::Init, dt: DType, device: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dt, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("{name}: expected {shape:?}, got {:?}", tensor.shape());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dt: DType, device: &Device) -> candle_core::Result<Tensor> {
        let load = || -> Result<Tensor> {
            let info = self.file.tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing checkpoint tensor {name}"))?;
            if info.dtype == "F8_E4M3" {
                if let Some((start, packed)) = &self.packed {
                    let offset = usize::try_from(info.data_offsets[0] - start)?;
                    let len = usize::try_from(info.data_offsets[1] - info.data_offsets[0])?;
                    return Ok(packed.narrow(0, offset, len)?.reshape(info.shape.as_slice())?.to_dtype(dt)?);
                }
            }
            let host = name.ends_with(".comfy_quant") || name.ends_with(".weight_scale");
            Ok(self.file.tensor(name, if host { &Device::Cpu } else { device })?.to_dtype(dt)?)
        };
        load().map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.file.contains(name)
    }
}

fn dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F8_E4M3" => DType::F8E4M3,
        "U8" => DType::U8,
        _ => anyhow::bail!("unsupported Layered dtype {name}"),
    })
}

enum Weight {
    Full(Tensor),
    Scalar(Fp8Linear),
    Rows { weight: Tensor, scale: Tensor },
}

/// Bias-capable FP8 linear; encoder calls force dequantized multiplication.
pub struct LayeredLinear {
    weight: Weight,
    bias: Option<Tensor>,
}

impl LayeredLinear {
    pub fn load(input: usize, output: usize, bias: bool, full: bool, plan: Option<&Fp8Plan>, vb: VarBuilder) -> Result<Self> {
        // qir's FP8 GEMM assumes activation scale one; dequantize for other scales.
        let full = full || vb.contains_tensor("input_scale") || vb.contains_tensor("scale_input");
        let weight = if vb.contains_tensor("weight_scale") || vb.contains_tensor("scale_weight") {
            let key = if vb.contains_tensor("weight_scale") { "weight_scale" } else { "scale_weight" };
            let scale = vb.get_unchecked_dtype(key, DType::F32)?;
            let w = vb.get_unchecked_dtype("weight", DType::F8E4M3)?;
            anyhow::ensure!(w.dims() == [output, input], "{}: incorrect linear shape", vb.prefix());
            if scale.elem_count() == 1 {
                let fp8 = if key == "weight_scale" {
                    crate::fp8::load_linear(input, output, plan, &vb)?.ok_or_else(|| anyhow::anyhow!("missing FP8 layer"))?
                } else {
                    Fp8Linear::new_full_precision(w, &scale)?
                };
                Weight::Scalar(if full { fp8.full_precision() } else { fp8 })
            } else {
                anyhow::ensure!(scale.dims() == [output, 1] || scale.dims() == [output], "{}: invalid row scales", vb.prefix());
                Weight::Rows { weight: w, scale: scale.reshape((output, 1))? }
            }
        } else {
            Weight::Full(vb.get((output, input), "weight")?)
        };
        Ok(Self { weight, bias: if bias { Some(vb.get(output, "bias")?) } else { None } })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = match &self.weight {
            Weight::Scalar(w) => w.forward(x)?,
            Weight::Full(w) => candle_nn::Linear::new(w.to_dtype(x.dtype())?, None).forward(x)?,
            Weight::Rows { weight, scale } => {
                let w = weight.to_dtype(DType::F32)?.to_dtype(DType::BF16)?.broadcast_mul(&scale.to_dtype(DType::BF16)?)?.to_dtype(x.dtype())?;
                candle_nn::Linear::new(w, None).forward(x)?
            }
        };
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(x.dtype())?)?,
            None => y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_checkpoint_views_and_noncontiguous_fallback() -> Result<()> {
        let device = Device::Cpu;
        for interleaved in [false, true] {
            let path = std::env::temp_dir().join(format!("qir-layered-bulk-{}-{interleaved}.bin", std::process::id()));
            let (header, payload) = if interleaved {
                (
                    serde_json::json!({
                        "a": {"dtype":"F8_E4M3", "shape":[2], "data_offsets":[0,2]},
                        "b": {"dtype":"BF16", "shape":[1], "data_offsets":[2,4]},
                        "c": {"dtype":"F8_E4M3", "shape":[2], "data_offsets":[4,6]}
                    }),
                    vec![0x38u8, 0x40, 0x80, 0x3f, 0x44, 0x48],
                )
            } else {
                (
                    serde_json::json!({
                        "a": {"dtype":"F8_E4M3", "shape":[2], "data_offsets":[0,2]},
                        "c": {"dtype":"F8_E4M3", "shape":[2], "data_offsets":[2,4]},
                        "b": {"dtype":"BF16", "shape":[1], "data_offsets":[4,6]}
                    }),
                    vec![0x38u8, 0x40, 0x44, 0x48, 0x80, 0x3f],
                )
            };
            let header = serde_json::to_vec(&header)?;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(payload);
            std::fs::write(&path, bytes)?;
            let file = TensorFile::open(&path)?;
            let vb = TensorFile::open(&path)?.resident_builder(DType::F32, &device)?;
            for (name, dt) in [("a", DType::F8E4M3), ("b", DType::BF16), ("c", DType::F8E4M3)] {
                let expected = file.tensor(name, &device)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                assert_eq!(vb.get_unchecked_dtype(name, dt)?.to_dtype(DType::F32)?.to_vec1::<f32>()?, expected);
            }
            assert!(vb.get(3, "a").is_err());
            assert!(vb.get_unchecked("missing").is_err());
            drop(vb);
            drop(file);
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}
