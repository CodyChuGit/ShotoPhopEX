//! What a file holds and what a folder has. A [`Fingerprint`] is what the header says (tensor
//! count, type histogram, naming prefixes, the shapes and metadata that tell families apart);
//! [`identify`] turns it into a [`Family`] and a [`Precision`] from the conventions read off the
//! real files (`docs/MODELS-ON-DISK.md`), never from the file name; [`ModelIndex::scan`] walks a
//! folder in ComfyUI's layout (`diffusion_models/`, `unet/`, `text_encoders/`, `clip/`, `vae/`,
//! `loras/`, `checkpoints/`) or a flat folder and lists what is installed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::gguf::Gguf;
use crate::safetensors::SafeTensors;

/// Which model a file holds (the architecture, not the checkpoint's name).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Family {
    /// Krea 2 (Raw or Turbo) transformer: `blocks.N.attn.{wq,wk,wv,wo,gate}`, `txtfusion`,
    /// `first`, `last`.
    Krea2,
    /// Qwen-Image / Qwen-Image-Edit (the 20B transformer, hidden 3072, 60 blocks; 2509, 2511 and
    /// Layered share it): `transformer_blocks`, `img_in [3072, 64]`, `txt_norm`.
    QwenImage,
    /// Qwen-Image-2.1 (hidden 4096, 32 blocks): `transformer_blocks`, `modulation`,
    /// `img_in [4096, 64]`.
    QwenImage21,
    /// Qwen2.5-VL 7B as a text encoder (the 2511 family's): `model.embed_tokens [152064, 3584]`,
    /// a `visual` tower.
    Qwen25Vl7B,
    /// Qwen3-VL 4B as a text encoder (Krea 2's): `model.embed_tokens [151936, 2560]`.
    Qwen3Vl4B,
    /// Qwen3-VL 8B as a text encoder (Qwen-Image-2.1's): `model.embed_tokens [151936, 4096]`.
    Qwen3Vl8B,
    /// The Wan 2.1 VAE (16 latent channels, RGB out): Qwen-Image's and Krea 2's VAE.
    WanVae,
    /// The Wan-architecture VAE retrained for RGBA (Qwen-Image-Layered): `decoder.head.2 [4, …]`.
    QwenImageLayeredVae,
    /// Qwen-Image-2.1's own VAE (64 latent channels, RGBA; `modelspec.architecture =
    /// qwen_image_2.1_vae`).
    QwenImage21Vae,
    /// SAM 3 / 3.1 (`detector.*`, `tracker.*`).
    Sam3,
    /// A LoRA (`*.lora_down`/`lora_up`, `lora_A`/`lora_B`, or `lora_*` names).
    Lora,
    Unknown,
}

/// How the weights are stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Precision {
    Bf16,
    Fp16,
    Fp32,
    /// ComfyUI's `float8_e4m3fn`: `weight` F8E4M3 with a per-tensor `weight_scale` F32 (and an
    /// `input_scale` in the "fp8mixed" files).
    Fp8Scaled,
    /// ComfyUI's `int8_tensorwise` with `convrot` (a Hadamard rotation, group size 256):
    /// `weight` I8 with a per-row `weight_scale` F32 `[out, 1]`.
    Int8ConvRot,
    /// ComfyUI's `nvfp4`: `weight` U8 `[out, in/2]` (two 4-bit values per byte),
    /// `weight_scale` F8E4M3 `[out, in/16]` block scales, `weight_scale_2` F32 global scale.
    Nvfp4,
    /// A GGUF quantisation, by its dominant block type (`Q8_0`, `Q4_K`, …).
    Gguf(String),
    Mixed,
    Unknown,
}

/// What the header says, format-agnostic.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fingerprint {
    pub format: String,
    pub tensors: usize,
    /// Bytes of tensor data.
    pub data_bytes: u64,
    /// Tensor count per element type name.
    pub dtypes: BTreeMap<String, usize>,
    /// The first few tensor names (sorted), enough to see the naming scheme.
    pub sample_names: Vec<String>,
    /// Name prefixes before the first `.` and how many tensors carry each.
    pub prefixes: BTreeMap<String, usize>,
    /// Metadata that identifies a file (`format`, `modelspec.*`, `general.architecture`, …),
    /// values shortened.
    pub metadata: BTreeMap<String, String>,
    /// Quantisation formats named by ComfyUI's `_quantization_metadata` (`float8_e4m3fn`,
    /// `int8_tensorwise`, `nvfp4`).
    pub quant_formats: BTreeSet<String>,
    /// `comfy_quant` descriptor tensors (one per quantised layer).
    pub comfy_quant_tensors: usize,
    /// Tensors that belong to a LoRA.
    pub lora_tensors: usize,
    /// `model.embed_tokens.weight`'s shape (a language model's vocabulary and hidden size).
    pub embed_shape: Option<Vec<u64>>,
    /// `img_in.weight`'s shape (a Qwen-Image transformer's hidden size).
    pub img_in_shape: Option<Vec<u64>>,
    /// `decoder.head.2.weight`'s output channels (a VAE's colour channels).
    pub decoder_out: Option<u64>,
    /// Any `*.weight_scale_2` (the NVFP4 global scale).
    pub has_scale_2: bool,
}

fn quant_formats(metadata: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if let Some(text) = metadata.get("_quantization_metadata")
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(text)
        && let Some(layers) = v.get("layers").and_then(|l| l.as_object())
    {
        for layer in layers.values() {
            if let Some(f) = layer.get("format").and_then(|f| f.as_str()) {
                out.insert(f.to_string());
            }
        }
    }
    out
}

fn is_lora_name(n: &str) -> bool {
    n.starts_with("lora_") || n.contains(".lora_down.") || n.contains(".lora_up.") || n.contains(".lora_A.") || n.contains(".lora_B.")
}

impl Fingerprint {
    pub fn of_safetensors(st: &SafeTensors) -> Self {
        let mut fp = Self::empty("safetensors", st.data_len());
        let mut names: Vec<&str> = Vec::with_capacity(st.tensors().len());
        for t in st.tensors() {
            fp.note(&t.name, &t.dtype.to_string(), &t.shape);
            names.push(&t.name);
        }
        fp.finish(names);
        fp.quant_formats = quant_formats(st.metadata());
        fp.metadata = st.metadata().iter().filter(|(k, _)| k.len() < 64).map(|(k, v)| (k.clone(), v.chars().take(120).collect())).collect();
        fp
    }

    pub fn of_gguf(g: &Gguf) -> Self {
        let mut fp = Self::empty("gguf", 0);
        let mut names: Vec<&str> = Vec::with_capacity(g.tensors().len());
        for t in g.tensors() {
            // ggml dims are innermost first; shapes here are outermost first like safetensors.
            let shape: Vec<u64> = t.dims.iter().rev().copied().collect();
            fp.note(&t.name, &t.dtype.to_string(), &shape);
            fp.data_bytes = fp.data_bytes.saturating_add(t.bytes.unwrap_or(0));
            names.push(&t.name);
        }
        fp.finish(names);
        fp.metadata = g
            .metadata()
            .iter()
            .filter(|(k, _)| k.starts_with("general.") && !k.contains("tags"))
            .map(|(k, v)| (k.clone(), format!("{v:?}").chars().take(120).collect()))
            .collect();
        fp
    }

    fn empty(format: &str, data_bytes: u64) -> Self {
        Self {
            format: format.into(),
            tensors: 0,
            data_bytes,
            dtypes: BTreeMap::new(),
            sample_names: Vec::new(),
            prefixes: BTreeMap::new(),
            metadata: BTreeMap::new(),
            quant_formats: BTreeSet::new(),
            comfy_quant_tensors: 0,
            lora_tensors: 0,
            embed_shape: None,
            img_in_shape: None,
            decoder_out: None,
            has_scale_2: false,
        }
    }

    fn note(&mut self, name: &str, dtype: &str, shape: &[u64]) {
        self.tensors += 1;
        *self.dtypes.entry(dtype.to_string()).or_insert(0) += 1;
        *self.prefixes.entry(name.split('.').next().unwrap_or("").to_string()).or_insert(0) += 1;
        if name.ends_with(".comfy_quant") {
            self.comfy_quant_tensors += 1;
        }
        if is_lora_name(name) {
            self.lora_tensors += 1;
        }
        if name.ends_with(".weight_scale_2") {
            self.has_scale_2 = true;
        }
        match name {
            "model.embed_tokens.weight" => self.embed_shape = Some(shape.to_vec()),
            "img_in.weight" => self.img_in_shape = Some(shape.to_vec()),
            "decoder.head.2.weight" => self.decoder_out = shape.first().copied(),
            _ => {}
        }
    }

    fn finish(&mut self, mut names: Vec<&str>) {
        names.sort_unstable();
        self.sample_names = names.iter().take(12).map(|s| (*s).to_string()).collect();
    }

    fn count(&self, dtype: &str) -> usize {
        self.dtypes.get(dtype).copied().unwrap_or(0)
    }

    fn has_prefix(&self, p: &str) -> bool {
        self.prefixes.contains_key(p)
    }
}

/// A file the index recognised (or not).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Installed {
    pub path: PathBuf,
    /// The ComfyUI folder it sits in (`diffusion_models`, `vae`, …) or `""` for a flat folder.
    pub folder: String,
    pub size: u64,
    pub family: Family,
    pub precision: Precision,
    pub fingerprint: Option<Fingerprint>,
    /// Why a file could not be read, when it could not.
    pub error: Option<String>,
}

/// The ComfyUI folders the index looks in.
pub const FOLDERS: &[&str] = &["diffusion_models", "unet", "text_encoders", "clip", "vae", "loras", "checkpoints"];

/// Which [`Family`] and [`Precision`] a fingerprint describes, from its tensors alone.
pub fn identify(fp: &Fingerprint) -> (Family, Precision) {
    (family_of(fp), precision_of(fp))
}

fn family_of(fp: &Fingerprint) -> Family {
    if fp.lora_tensors > 0 && fp.lora_tensors * 2 >= fp.tensors {
        return Family::Lora;
    }
    if fp.has_prefix("detector") && fp.has_prefix("tracker") {
        return Family::Sam3;
    }
    if fp.has_prefix("blocks") && fp.has_prefix("txtfusion") && fp.has_prefix("first") && fp.has_prefix("last") {
        return Family::Krea2;
    }
    if fp.has_prefix("transformer_blocks") && fp.has_prefix("img_in") {
        let hidden = fp.img_in_shape.as_ref().and_then(|s| s.first().copied());
        return if fp.has_prefix("modulation") || hidden == Some(4096) { Family::QwenImage21 } else { Family::QwenImage };
    }
    if let Some(s) = &fp.embed_shape {
        return match (s.first().copied(), s.get(1).copied()) {
            (Some(151936), Some(2560)) => Family::Qwen3Vl4B,
            (Some(151936), Some(4096)) => Family::Qwen3Vl8B,
            (Some(152064), Some(3584)) => Family::Qwen25Vl7B,
            _ => Family::Unknown,
        };
    }
    if fp.has_prefix("decoder") && fp.has_prefix("encoder") {
        if fp.metadata.get("modelspec.architecture").is_some_and(|a| a == "qwen_image_2.1_vae") {
            return Family::QwenImage21Vae;
        }
        return match fp.decoder_out {
            Some(4) => Family::QwenImageLayeredVae,
            Some(3) => Family::WanVae,
            _ => Family::Unknown,
        };
    }
    Family::Unknown
}

fn precision_of(fp: &Fingerprint) -> Precision {
    if fp.format == "gguf" {
        let dominant = fp
            .dtypes
            .iter()
            .filter(|(k, _)| k.starts_with('Q') || k.starts_with("IQ") || k.starts_with("NVFP") || k.starts_with("MXFP") || k.starts_with("TQ"))
            .max_by_key(|(_, n)| **n)
            .map(|(k, _)| k.clone());
        return dominant.map_or(Precision::Unknown, Precision::Gguf);
    }
    // ComfyUI's own description first; the mixed files (an NVFP4 transformer with fp8 embeddings)
    // are named by their lowest precision.
    let fmt = |needle: &str| fp.quant_formats.iter().any(|f| f.contains(needle));
    if fmt("nvfp4") || fp.has_scale_2 {
        return Precision::Nvfp4;
    }
    if fmt("int8") || fp.count("I8") > fp.tensors / 8 {
        return Precision::Int8ConvRot;
    }
    if fmt("float8") || fp.count("F8E4M3") + fp.count("F8E5M2") > fp.tensors / 8 {
        return Precision::Fp8Scaled;
    }
    let total = fp.tensors.max(1);
    if fp.count("BF16") > total / 2 {
        Precision::Bf16
    } else if fp.count("F16") > total / 2 {
        Precision::Fp16
    } else if fp.count("F32") > total / 2 {
        Precision::Fp32
    } else if fp.tensors == 0 {
        Precision::Unknown
    } else {
        Precision::Mixed
    }
}

/// Read one file's header and identify it.
pub fn inspect(path: &Path, folder: &str) -> Installed {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let fp = match ext.as_str() {
        "safetensors" | "sft" => SafeTensors::open(path).map(|st| Fingerprint::of_safetensors(&st)),
        "gguf" => Gguf::open(path).map(|g| Fingerprint::of_gguf(&g)),
        "pt" | "pth" | "ckpt" | "bin" => Err(format!("{name}: pickle-based checkpoints are not read (convert to safetensors)")),
        _ => Err(format!("{name}: not a model file")),
    };
    match fp {
        Ok(fp) => {
            let (family, precision) = identify(&fp);
            Installed { path: path.to_path_buf(), folder: folder.to_string(), size, family, precision, fingerprint: Some(fp), error: None }
        }
        Err(e) => Installed {
            path: path.to_path_buf(),
            folder: folder.to_string(),
            size,
            family: Family::Unknown,
            precision: Precision::Unknown,
            fingerprint: None,
            error: Some(e),
        },
    }
}

/// What a models folder holds.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelIndex {
    pub root: PathBuf,
    pub files: Vec<Installed>,
}

fn is_model_file(p: &Path) -> bool {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| matches!(e.as_str(), "safetensors" | "sft" | "gguf" | "pt" | "pth" | "ckpt" | "bin"))
}

impl ModelIndex {
    /// Scan `root`: ComfyUI's subfolders when present (one level deep each), else the folder
    /// itself. Headers only; a 20 GB file costs a few milliseconds.
    pub fn scan(root: &Path) -> crate::Result<Self> {
        if !root.is_dir() {
            return Err(format!("{} is not a folder", root.display()));
        }
        let mut files = Vec::new();
        let mut any_folder = false;
        for folder in FOLDERS {
            let dir = root.join(folder);
            if dir.is_dir() {
                any_folder = true;
                for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten() {
                    let p = entry.path();
                    if p.is_file() && is_model_file(&p) {
                        files.push(inspect(&p, folder));
                    }
                }
            }
        }
        if !any_folder {
            for entry in std::fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?.flatten() {
                let p = entry.path();
                if p.is_file() && is_model_file(&p) {
                    files.push(inspect(&p, ""));
                }
            }
        }
        files.sort_by(|a, b| (a.folder.as_str(), a.path.as_path()).cmp(&(b.folder.as_str(), b.path.as_path())));
        Ok(Self { root: root.to_path_buf(), files })
    }

    /// The installed files of one family.
    pub fn of(&self, family: Family) -> impl Iterator<Item = &Installed> {
        self.files.iter().filter(move |f| f.family == family)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn st(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])], meta: &[(&str, &str)]) {
        let meta: BTreeMap<String, String> = meta.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect();
        crate::safetensors::write(path, tensors, &meta).unwrap();
    }

    #[test]
    fn families_and_precisions_come_from_the_tensors_not_the_names() {
        let d = std::env::temp_dir().join(format!("gn-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let b16 = vec![0u8; 16];
        let f32s = 1.0f32.to_le_bytes();
        let dm = d.join("diffusion_models");
        // Qwen-Image 20B, fp8mixed (named anything).
        st(
            &dm.join("a.safetensors"),
            &[
                ("img_in.weight", "BF16", &[3072, 64], &vec![0u8; 3072 * 64 * 2]),
                ("transformer_blocks.0.attn.to_k.weight", "F8_E4M3", &[16], &b16),
                ("transformer_blocks.0.attn.to_k.weight_scale", "F32", &[], &f32s),
                ("txt_norm.weight", "BF16", &[8], &b16),
            ],
            &[("_quantization_metadata", r#"{"layers":{"transformer_blocks.0.attn.to_k":{"format":"float8_e4m3fn"}}}"#)],
        );
        // Qwen-Image-2.1, int8 convrot.
        st(
            &dm.join("b.safetensors"),
            &[
                ("img_in.weight", "BF16", &[4096, 64], &vec![0u8; 4096 * 64 * 2]),
                ("modulation.1.weight", "BF16", &[8], &b16),
                ("transformer_blocks.0.attn.to_k.weight", "I8", &[16], &b16),
                ("transformer_blocks.0.attn.to_k.weight_scale", "F32", &[1, 1], &f32s),
                ("transformer_blocks.0.attn.to_k.comfy_quant", "U8", &[16], &b16),
            ],
            &[],
        );
        // Krea 2, NVFP4 (U8 pairs, F8 block scales, F32 global scale).
        st(
            &dm.join("c.safetensors"),
            &[
                ("blocks.0.attn.wk.weight", "U8", &[4, 4], &b16),
                ("blocks.0.attn.wk.weight_scale", "F8_E4M3", &[4, 1], &b16[..4]),
                ("blocks.0.attn.wk.weight_scale_2", "F32", &[], &f32s),
                ("txtfusion.x", "BF16", &[8], &b16),
                ("first.w", "BF16", &[8], &b16),
                ("last.w", "BF16", &[8], &b16),
            ],
            &[],
        );
        let te = d.join("text_encoders");
        // Encoders by their embedding shape (a tiny stand-in tensor with the real shape would
        // not match its byte count, so the shape is carried by a zero-sized second dimension).
        st(&te.join("e4.safetensors"), &[("model.embed_tokens.weight", "BF16", &[151936, 2560, 0], &[])], &[]);
        st(&te.join("e8.safetensors"), &[("model.embed_tokens.weight", "BF16", &[151936, 4096, 0], &[])], &[]);
        st(&te.join("e25.safetensors"), &[("model.embed_tokens.weight", "BF16", &[152064, 3584, 0], &[]), ("visual.blocks.0.w", "BF16", &[8], &b16)], &[]);
        let vae = d.join("vae");
        st(&vae.join("wan.safetensors"), &[("decoder.head.2.weight", "BF16", &[3, 1, 1, 1, 1], &b16[..6]), ("encoder.head.2.weight", "BF16", &[8], &b16)], &[]);
        st(
            &vae.join("layered.safetensors"),
            &[("decoder.head.2.weight", "BF16", &[4, 1, 1, 1, 1], &b16[..8]), ("encoder.head.2.weight", "BF16", &[8], &b16)],
            &[],
        );
        st(
            &vae.join("v21.safetensors"),
            &[("decoder.head.2.weight", "BF16", &[4, 1, 1, 1, 1], &b16[..8]), ("encoder.head.2.weight", "BF16", &[8], &b16)],
            &[("modelspec.architecture", "qwen_image_2.1_vae")],
        );
        st(
            &d.join("loras").join("l.safetensors"),
            &[
                ("transformer_blocks.0.attn.to_k.lora_down.weight", "BF16", &[8], &b16),
                ("transformer_blocks.0.attn.to_k.lora_up.weight", "BF16", &[8], &b16),
                ("transformer_blocks.0.attn.to_k.alpha", "BF16", &[], &b16[..2]),
            ],
            &[],
        );
        st(&d.join("checkpoints").join("sam.safetensors"), &[("detector.x", "F16", &[8], &b16), ("tracker.y", "F16", &[8], &b16)], &[]);
        std::fs::write(d.join("checkpoints").join("old.ckpt"), b"\x80\x02").unwrap();
        let ix = ModelIndex::scan(&d).unwrap();
        let kind = |name: &str| {
            let f = ix.files.iter().find(|f| f.path.ends_with(name)).unwrap();
            (f.family, f.precision.clone())
        };
        assert_eq!(kind("a.safetensors"), (Family::QwenImage, Precision::Fp8Scaled));
        assert_eq!(kind("b.safetensors"), (Family::QwenImage21, Precision::Int8ConvRot));
        assert_eq!(kind("c.safetensors"), (Family::Krea2, Precision::Nvfp4));
        assert_eq!(kind("e4.safetensors").0, Family::Qwen3Vl4B);
        assert_eq!(kind("e8.safetensors").0, Family::Qwen3Vl8B);
        assert_eq!(kind("e25.safetensors").0, Family::Qwen25Vl7B);
        assert_eq!(kind("wan.safetensors"), (Family::WanVae, Precision::Bf16));
        assert_eq!(kind("layered.safetensors").0, Family::QwenImageLayeredVae);
        assert_eq!(kind("v21.safetensors").0, Family::QwenImage21Vae);
        assert_eq!(kind("l.safetensors").0, Family::Lora);
        assert_eq!(kind("sam.safetensors"), (Family::Sam3, Precision::Fp16));
        assert!(ix.files.iter().find(|f| f.path.ends_with("old.ckpt")).unwrap().error.as_deref().unwrap_or("").contains("pickle"));
        assert_eq!(ix.of(Family::Lora).count(), 1);
        assert_eq!(ix.files.iter().find(|f| f.path.ends_with("c.safetensors")).unwrap().folder, "diffusion_models");
        let _ = std::fs::remove_dir_all(d);
    }

    /// The real files on this machine, when present (the owner's ComfyUI): every one identified,
    /// none unknown. Skipped elsewhere.
    #[test]
    fn the_local_comfyui_models_are_all_identified() {
        let root = Path::new(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models");
        if !root.is_dir() {
            return;
        }
        let ix = ModelIndex::scan(root).unwrap();
        let unknown: Vec<String> =
            ix.files.iter().filter(|f| f.family == Family::Unknown || f.precision == Precision::Unknown).map(|f| f.path.display().to_string()).collect();
        assert!(unknown.is_empty(), "{unknown:#?}");
        assert!(ix.of(Family::Krea2).count() >= 1 && ix.of(Family::Sam3).count() >= 1);
    }
}
