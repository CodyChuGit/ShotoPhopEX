//! Resolve unmodified ComfyUI 2511 checkpoints without loading model payloads.
use crate::{layered_files::TensorFile, Result};
use candle_core::{DType, Device};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

pub const ADAPTER_NAMES: [&str; 4] = [
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    "Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors",
    "qwen-image-edit-2511-multiple-angles-lora.safetensors",
];

/// Optional adapters remain absent until installed; required base files are validated.
#[derive(Debug, Clone)]
pub struct Edit2511Files {
    pub diffusion: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    pub lightning4: Option<PathBuf>,
    pub lightning8: Option<PathBuf>,
    pub remover: Option<PathBuf>,
    pub angles: Option<PathBuf>,
}

impl Edit2511Files {
    pub fn resolve(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let optional = |name| {
            let path = root.join("loras").join(name);
            path.exists().then_some(path)
        };
        let files = Self {
            diffusion: root.join("diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
            text_encoder: root.join("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            vae: root.join("vae/qwen_image_vae.safetensors"),
            lightning4: optional(ADAPTER_NAMES[0]),
            lightning8: optional(ADAPTER_NAMES[1]),
            remover: optional(ADAPTER_NAMES[2]),
            angles: optional(ADAPTER_NAMES[3]),
        };
        files.validate()?;
        Ok(files)
    }

    /// Validate inventories and quantization contracts with only headers and small tensors.
    pub fn validate(&self) -> Result<()> {
        let dit = Header::open(&self.diffusion)?;
        dit.inventory(&[("BF16", 1094), ("F32", 1679), ("F8_E4M3", 839), ("U8", 839)])?;
        dit.check("__index_timestep_zero__", &[0], "F32")?;
        anyhow::ensure!(!dit.entries.contains_key("time_text_embed.addition_t_embedding.weight"), "Layered addition is invalid for 2511");
        for (key, shape) in [("img_in.weight", vec![3072, 64]), ("proj_out.weight", vec![64, 3072]), ("txt_norm.weight", vec![3584])] {
            dit.check(key, &shape, "BF16")?;
        }
        let file = TensorFile::open(&self.diffusion)?;
        let mut quantized = 0;
        for (name, info) in &dit.entries {
            if let Some(stem) = name.strip_suffix(".weight_scale") {
                anyhow::ensure!(info.shape.is_empty() && info.dtype == "F32", "{name}: expected scalar F32");
                let weight = dit.get(&format!("{stem}.weight"))?;
                anyhow::ensure!(weight.dtype == "F8_E4M3" && weight.shape.len() == 2, "{stem}: expected FP8 matrix");
                let input = format!("{stem}.input_scale");
                dit.check(&input, &[], "F32")?;
                anyhow::ensure!(file.tensor(&input, &Device::Cpu)?.to_scalar::<f32>()? == 1., "{input}: unsupported activation scale");
                let bytes = file.tensor(&format!("{stem}.comfy_quant"), &Device::Cpu)?.to_vec1::<u8>()?;
                let record: serde_json::Value = serde_json::from_slice(&bytes)?;
                anyhow::ensure!(record["format"] == "float8_e4m3fn" && record["full_precision_matrix_mult"] == true, "{stem}: unsupported quantization record");
                quantized += 1;
            }
        }
        anyhow::ensure!(quantized == 839, "expected 839 quantized matrices, got {quantized}");
        for i in 0..60 {
            for stream in ["img", "txt"] {
                dit.check(
                    &format!("transformer_blocks.{i}.{stream}_mod.1.weight"),
                    &[18432, 3072],
                    if i == 0 && stream == "img" { "BF16" } else { "F8_E4M3" },
                )?;
            }
        }
        drop(file);
        let text = Header::open(&self.text_encoder)?;
        text.inventory(&[("BF16", 371), ("F32", 716), ("F8_E4M3", 359)])?;
        text.check("model.embed_tokens.weight", &[152064, 3584], "BF16")?;
        text.check("model.norm.weight", &[3584], "BF16")?;
        text.check("scaled_fp8", &[0], "F8_E4M3")?;
        for i in 0..28 {
            for (name, shape) in [
                ("self_attn.q_proj", [3584, 3584]),
                ("self_attn.k_proj", [512, 3584]),
                ("self_attn.v_proj", [512, 3584]),
                ("self_attn.o_proj", [3584, 3584]),
                ("mlp.gate_proj", [18944, 3584]),
                ("mlp.up_proj", [18944, 3584]),
                ("mlp.down_proj", [3584, 18944]),
            ] {
                text.check(&format!("model.layers.{i}.{name}.weight"), &shape, "F8_E4M3")?;
            }
            for (name, width) in [("q_proj", 3584), ("k_proj", 512), ("v_proj", 512)] {
                text.check(&format!("model.layers.{i}.self_attn.{name}.bias"), &[width], "BF16")?;
            }
            for name in ["input_layernorm", "post_attention_layernorm"] {
                text.check(&format!("model.layers.{i}.{name}.weight"), &[3584], "BF16")?;
            }
        }
        text.check("visual.patch_embed.proj.weight", &[1280, 3, 2, 14, 14], "BF16")?;
        anyhow::ensure!(text.entries.keys().filter(|k| k.starts_with("visual.blocks.")).count() == 704, "expected 704 vision block entries");
        for i in 0..32 {
            for (name, shape) in [
                ("attn.qkv", [3840, 1280]),
                ("attn.proj", [1280, 1280]),
                ("mlp.gate_proj", [3420, 1280]),
                ("mlp.up_proj", [3420, 1280]),
                ("mlp.down_proj", [1280, 3420]),
            ] {
                text.check(&format!("visual.blocks.{i}.{name}.weight"), &shape, "F8_E4M3")?;
                text.check(&format!("visual.blocks.{i}.{name}.bias"), &[shape[0]], "BF16")?;
            }
            for norm in ["norm1", "norm2"] {
                text.check(&format!("visual.blocks.{i}.{norm}.weight"), &[1280], "BF16")?;
            }
        }
        text.check("visual.merger.ln_q.weight", &[1280], "BF16")?;
        for (name, shape) in [("0", [5120, 5120]), ("2", [3584, 5120])] {
            text.check(&format!("visual.merger.mlp.{name}.weight"), &shape, "F8_E4M3")?;
            text.check(&format!("visual.merger.mlp.{name}.bias"), &[shape[0]], "BF16")?;
        }
        anyhow::ensure!(text.entries.keys().filter(|k| k.starts_with("visual.merger.")).count() == 9, "expected nine merger entries");
        let mut scales = 0;
        for (name, info) in &text.entries {
            if let Some(stem) = name.strip_suffix(".scale_weight") {
                let weight = text.get(&format!("{stem}.weight"))?;
                anyhow::ensure!(weight.dtype == "F8_E4M3" && weight.shape.len() == 2, "{name}: expected FP8 matrix");
                anyhow::ensure!(
                    info.dtype == "F32"
                        && (info.shape.is_empty() || info.shape == [1] || info.shape == [weight.shape[0]] || info.shape == [weight.shape[0], 1]),
                    "{name}: invalid weight scale"
                );
                let input = text.get(&format!("{stem}.scale_input"))?;
                anyhow::ensure!(input.dtype == "F32" && (input.shape.is_empty() || input.shape == [1]), "{stem}: invalid input scale");
                scales += 1;
            }
        }
        anyhow::ensure!(scales == 358, "expected 358 text/vision scale pairs, got {scales}");
        // TensorFile also checks all ranges, dtypes and payload lengths.
        drop(TensorFile::open(&self.text_encoder)?);
        let vae = Header::open(&self.vae)?;
        vae.inventory(&[("BF16", 194)])?;
        for (name, shape) in [
            ("encoder.conv1.weight", vec![96, 3, 3, 3, 3]),
            ("decoder.conv1.weight", vec![384, 16, 3, 3, 3]),
            ("decoder.head.2.weight", vec![3, 96, 3, 3, 3]),
            ("decoder.head.2.bias", vec![3]),
        ] {
            vae.check(name, &shape, "BF16")?;
        }
        drop(TensorFile::open(&self.vae)?);
        for path in [&self.lightning4, &self.lightning8, &self.remover, &self.angles].into_iter().flatten() {
            let modules = adapter_inventory(path)?;
            // Unlike the spec's missing copy, the installed Angles adapter has 840 rank-16 modules.
            let count = if Some(path) == self.angles.as_ref() { 840 } else { 720 };
            anyhow::ensure!(modules.len() == count, "{}: expected {count} adapter modules, got {}", path.display(), modules.len());
            let expected =
                if Some(path) == self.lightning4.as_ref() || Some(path) == self.lightning8.as_ref() { Some((64, Some(8.))) } else { Some((16, None)) };
            if let Some((rank, alpha)) = expected {
                anyhow::ensure!(modules.values().all(|v| v.rank == rank && v.alpha == alpha), "{}: unexpected rank/alpha", path.display());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
}
struct Header {
    entries: BTreeMap<String, Info>,
}
impl Header {
    fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = [0u8; 8];
        file.read_exact(&mut bytes)?;
        let len = u64::from_le_bytes(bytes);
        anyhow::ensure!(len < 256 << 20 && len <= file.metadata()?.len().saturating_sub(8), "{}: invalid header length", path.display());
        let mut bytes = vec![0; len as usize];
        file.read_exact(&mut bytes)?;
        let mut header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
        header.remove("__metadata__");
        Ok(Self { entries: serde_json::from_value(header.into())? })
    }
    fn get(&self, key: &str) -> Result<&Info> {
        self.entries.get(key).ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))
    }
    fn check(&self, key: &str, shape: &[usize], dtype: &str) -> Result<()> {
        let info = self.get(key)?;
        anyhow::ensure!(info.shape == shape && info.dtype == dtype, "{key}: expected {dtype} {shape:?}, got {} {:?}", info.dtype, info.shape);
        Ok(())
    }
    fn inventory(&self, expected: &[(&str, usize)]) -> Result<()> {
        let mut counts = BTreeMap::new();
        for info in self.entries.values() {
            *counts.entry(info.dtype.as_str()).or_insert(0usize) += 1;
        }
        anyhow::ensure!(counts == expected.iter().copied().collect(), "unexpected dtype inventory {counts:?}");
        Ok(())
    }
}

/// Header-derived LoRA map; an absent file alpha yields unit effective scale.
#[derive(Debug, Clone, serde::Serialize, Deserialize, PartialEq)]
pub struct AdapterModule {
    pub rank: usize,
    pub alpha: Option<f32>,
    pub scale: f32,
}

/// Inspect all adapter pairs, loading only scalar alphas, so verification stays CPU and small.
pub fn adapter_inventory(path: &Path) -> Result<BTreeMap<String, AdapterModule>> {
    let header = Header::open(path)?;
    let file = TensorFile::open(path)?;
    let mut out = BTreeMap::new();
    let mut used = std::collections::BTreeSet::new();
    for (key, info) in &header.entries {
        let pair = [(".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"), (".lora_A.default.weight", ".lora_B.default.weight")]
            .into_iter()
            .find_map(|(a, b)| key.strip_suffix(a).map(|stem| (stem, format!("{stem}{b}"))));
        let Some((stem, b)) = pair else { continue };
        let up = header.get(&b)?;
        anyhow::ensure!(
            info.dtype == "BF16" && up.dtype == "BF16" && info.shape.len() == 2 && up.shape.len() == 2 && info.shape[0] > 0 && info.shape[0] == up.shape[1],
            "{stem}: invalid adapter matrices"
        );
        let alpha_key = format!("{stem}.alpha");
        let alpha = if header.entries.contains_key(&alpha_key) {
            used.insert(alpha_key.clone());
            Some(
                file.tensor(&alpha_key, &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("empty alpha"))?,
            )
        } else {
            None
        };
        used.insert(key.clone());
        used.insert(b);
        let module =
            stem.strip_prefix("diffusion_model.").or_else(|| stem.strip_prefix("transformer.")).or_else(|| stem.strip_prefix("lora_unet_")).unwrap_or(stem);
        anyhow::ensure!(
            out.insert(module.to_owned(), AdapterModule { rank: info.shape[0], alpha, scale: alpha.map_or(1., |a| a / info.shape[0] as f32) }).is_none(),
            "duplicate module {module}"
        );
    }
    anyhow::ensure!(!out.is_empty() && used.len() == header.entries.len(), "{}: unmatched adapter tensors", path.display());
    Ok(out)
}
