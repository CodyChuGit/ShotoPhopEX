//! Models in ComfyUI's own folder layout (Rust rebuild R4): every component found by its file
//! name under the user's `models` folder (`diffusion_models/`, `text_encoders/`, `vae/`) and read
//! as it is, with no snapshot, converted copy or tokenizer file of ours. The tensor names that
//! differ from Hugging Face's snapshot (ComfyUI's repacks) are adapted at load: the 2.1 DiT's
//! fused `img_mlp.gate_up` (read by `dit::SwiGlu` itself), the text encoder's `model.` root (the
//! loaders take the root) and the 2.1 VAE's Wan-style names ([`Renamed`] with
//! [`qwen21_vae_name`]). The Qwen tokenizer is assembled from the vocabulary, merges and added
//! tokens ComfyUI keeps in its code tree (`comfy/text_encoders/qwen25_tokenizer`), the same
//! pipeline as Qwen's `tokenizer.json`. Mapping: `docs/comfy/research/notes/rust-rebuild-r4-comfy-layout.md`
//! in PhotoshopEX.

use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use tokenizers::Tokenizer;

use crate::Result;

/// Qwen-Image-2.1's three files in a ComfyUI `models` folder.
#[derive(Clone, Debug)]
pub struct Qwen21Files {
    pub dit: PathBuf,
    pub text: PathBuf,
    pub vae: PathBuf,
    /// The folder holding `vocab.json`, `merges.txt` and `tokenizer_config.json`.
    pub tokenizer: PathBuf,
}

/// Krea 2 Turbo's three files in a ComfyUI `models` folder (its encoder is ComfyUI's
/// `qwen3vl_4b` repack, rooted at `model`).
#[derive(Clone, Debug)]
pub struct Krea2Files {
    pub dit: PathBuf,
    pub text: PathBuf,
    pub vae: PathBuf,
    pub tokenizer: PathBuf,
}

/// ComfyUI's file names, first found wins. The bf16 files only: ComfyUI's `int8_convrot`
/// repacks use another rotation basis and per-row scales (the spec's steps 5 and 6).
const QWEN21_DIT: &[&str] = &["qwen_image_2.1_bf16.safetensors"];
const QWEN21_TEXT: &[&str] = &["qwen3vl_8b_bf16.safetensors"];
const QWEN21_VAE: &[&str] = &["qwen_image_2.1_vae_bf16.safetensors"];
const KREA2_DIT: &[&str] = &["krea2_turbo_fp8_scaled.safetensors"];
const KREA2_TEXT: &[&str] = &["qwen3vl_4b_fp8_scaled.safetensors"];
/// The Wan 2.1 VAE the native Krea 2 route was measured at parity with.
const KREA2_VAE: &[&str] = &["wan_2.1_vae.safetensors"];

fn first(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// Qwen-Image-2.1 under `root` (a ComfyUI `models` folder), when all three files and a
/// tokenizer are there.
pub fn find_qwen21(root: &Path) -> Option<Qwen21Files> {
    Some(Qwen21Files {
        dit: first(&root.join("diffusion_models"), QWEN21_DIT)?,
        text: first(&root.join("text_encoders"), QWEN21_TEXT)?,
        vae: first(&root.join("vae"), QWEN21_VAE)?,
        tokenizer: tokenizer_dir(root)?,
    })
}

/// Krea 2 Turbo under `root`, as [`find_qwen21`].
pub fn find_krea2(root: &Path) -> Option<Krea2Files> {
    Some(Krea2Files {
        dit: first(&root.join("diffusion_models"), KREA2_DIT)?,
        text: first(&root.join("text_encoders"), KREA2_TEXT)?,
        vae: first(&root.join("vae"), KREA2_VAE)?,
        tokenizer: tokenizer_dir(root)?,
    })
}

/// The Qwen tokenizer's folder for a `models` root: a copy in `text_encoders/qwen25_tokenizer`,
/// else ComfyUI's own beside the models folder (`<ComfyUI>/comfy/text_encoders/qwen25_tokenizer`).
pub fn tokenizer_dir(root: &Path) -> Option<PathBuf> {
    let has = |d: &Path| ["vocab.json", "merges.txt", "tokenizer_config.json"].iter().all(|f| d.join(f).is_file());
    let own = root.join("text_encoders").join("qwen25_tokenizer");
    let comfy = root.parent().map(|p| p.join("comfy").join("text_encoders").join("qwen25_tokenizer"));
    std::iter::once(own).chain(comfy).find(|d| has(d))
}

/// Qwen's byte-level BPE tokenizer from `dir`'s `vocab.json`, `merges.txt` and the added tokens
/// of `tokenizer_config.json`, with the pipeline of Qwen's `tokenizer.json` (NFC, the Qwen split
/// pattern, byte-level pre-tokenizer, post-processor and decoder; no BOS/EOS added).
pub fn qwen_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).with_context(|| format!("read {}", dir.join(f).display()));
    let vocab: serde_json::Value = serde_json::from_str(&read("vocab.json")?).context("vocab.json")?;
    let merges: Vec<serde_json::Value> = read("merges.txt")?
        .lines()
        .filter(|l| !l.starts_with("#version") && !l.is_empty())
        .map(|l| l.split_once(' ').map(|(a, b)| serde_json::json!([a, b])).with_context(|| format!("merges.txt: no pair in {l:?}")))
        .collect::<Result<_>>()?;
    let config: serde_json::Value = serde_json::from_str(&read("tokenizer_config.json")?).context("tokenizer_config.json")?;
    let decoder = config["added_tokens_decoder"].as_object().context("tokenizer_config.json has no added_tokens_decoder")?;
    let mut added: Vec<(u64, serde_json::Value)> = decoder
        .iter()
        .map(|(id, t)| -> Result<(u64, serde_json::Value)> {
            let id: u64 = id.parse().with_context(|| format!("added token id {id:?}"))?;
            let flag = |k: &str| t[k].as_bool().unwrap_or(false);
            let content = t["content"].as_str().context("added token without content")?;
            let entry = serde_json::json!({"id": id, "content": content, "single_word": flag("single_word"), "lstrip": flag("lstrip"),
                "rstrip": flag("rstrip"), "normalized": flag("normalized"), "special": flag("special")});
            Ok((id, entry))
        })
        .collect::<Result<_>>()?;
    added.sort_by_key(|(id, _)| *id);
    let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added.into_iter().map(|(_, t)| t).collect::<Vec<_>>(),
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_SPLIT}, "behavior": "Isolated", "invert": false},
            byte_level,
        ]},
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges},
    });
    Tokenizer::from_bytes(serde_json::to_vec(&json)?).map_err(|e| anyhow::anyhow!("assemble the Qwen tokenizer from {}: {e}", dir.display()))
}

/// Qwen2's pre-tokenizer split (contractions, letters with one leading mark, single digits,
/// punctuation runs, newlines, spaces).
const QWEN_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen-Image-2.1's VAE latent statistics (its snapshot's `vae/config.json`, which ComfyUI
/// hard-codes in `latent_formats.py`). Measured values: -0.5236 is not π/6.
#[allow(clippy::approx_constant)]
pub const QWEN21_LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472,
    0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555,
    0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275,
    4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub const QWEN21_LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443,
    2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735,
    3.149, 3.7662, 3.6746, 3.4563, 3.8161,
];

/// The name ComfyUI's `qwen_image_2.1_vae` file gives a tensor of the snapshot's VAE (diffusers
/// names in, Wan-style names out; the spec's table). A name the table does not know comes back
/// as it is.
pub fn qwen21_vae_name(name: &str) -> String {
    let Some((side, rest)) = name.split_once('.') else { return name.to_string() };
    match (side, rest) {
        ("quant_conv", p) => return format!("conv1.{p}"),
        ("post_quant_conv", p) => return format!("conv2.{p}"),
        ("encoder" | "decoder", _) => {}
        _ => return name.to_string(),
    }
    let parts: Vec<&str> = rest.split('.').collect();
    let tail = |n: usize| parts[n.min(parts.len())..].join(".");
    let mapped = match parts.as_slice() {
        ["conv_in", ..] => format!("conv1.{}", tail(1)),
        ["conv_out", ..] => format!("head.2.{}", tail(1)),
        ["norm_out", ..] => format!("head.0.{}", tail(1)),
        ["mid_block", "resnets", "0", ..] => format!("middle.0.{}", residual(&tail(3))),
        ["mid_block", "resnets", "1", ..] => format!("middle.2.{}", residual(&tail(3))),
        ["mid_block", "attentions", "0", ..] => format!("middle.1.{}", tail(3)),
        ["down_blocks", i, "resnets", j, ..] => format!("downsamples.{i}.downsamples.{j}.{}", residual(&tail(4))),
        ["down_blocks", i, "downsampler", ..] => format!("downsamples.{i}.downsamples.2.{}", tail(3)),
        ["up_blocks", i, "resnets", j, ..] => format!("upsamples.{i}.upsamples.{j}.{}", residual(&tail(4))),
        ["up_blocks", i, "upsampler", ..] => format!("upsamples.{i}.upsamples.3.{}", tail(3)),
        _ => rest.to_string(),
    };
    format!("{side}.{mapped}")
}

/// A residual block's member: diffusers' names to the Wan `residual` sequence's indices.
fn residual(member: &str) -> String {
    let (head, p) = member.split_once('.').unwrap_or((member, ""));
    let to = match head {
        "norm1" => "residual.0",
        "conv1" => "residual.2",
        "norm2" => "residual.3",
        "conv2" => "residual.6",
        "conv_shortcut" => "shortcut",
        _ => return member.to_string(),
    };
    if p.is_empty() {
        to.to_string()
    } else {
        format!("{to}.{p}")
    }
}

/// The metadata key a cached text GGUF keeps its source's fingerprint under.
const GGUF_SOURCE: &str = "qir.source";

/// A Q8_0 GGUF of the language model in `text` (ComfyUI's single file, the language model at
/// `lm_root`), the way `prequantize-text` makes one beside a snapshot: linears Q8_0, the
/// embedding and norms F16, the vision tower and head left out, names in the snapshot's layout
/// (`model.language_model.*`, which `QwenTextEncoder::load_gguf` reads). Cached under
/// `cache_root` beside the prequantized DiT and rebuilt when the source file changes: Q8_0 on
/// every load took 119 s for the 8B (2026-10-27), the cached file loads in seconds.
pub fn text_q8_cache(text: &Path, lm_root: &str, cache_root: &Path) -> Result<PathBuf> {
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    let entry = crate::convrot_cache::entry_path(cache_root, text)?.with_file_name("text_encoder.q8_0.gguf");
    let source = crate::convrot_cache::source_fingerprint(text, std::slice::from_ref(&text.to_path_buf()))?;
    if gguf_source(&entry).as_deref() == Some(source.as_str()) {
        return Ok(entry);
    }
    let t0 = std::time::Instant::now();
    tracing::info!(path = %entry.display(), "building the Q8_0 text encoder cache (one-time)");
    let st = unsafe { MmapedSafetensors::new(text)? };
    let (embed, layers) = (format!("{lm_root}.embed_tokens"), format!("{lm_root}.layers."));
    let mut names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).filter(|n| n.starts_with(&embed) || n.starts_with(&layers)).collect();
    names.sort();
    anyhow::ensure!(!names.is_empty(), "{} has no language model at {lm_root}", text.display());
    // Quantizing is CPU-bound: one name in N per thread.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(names.len());
    let parts: Vec<Result<Vec<(String, QTensor)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let (st, names) = (&st, &names);
                scope.spawn(move || -> Result<Vec<(String, QTensor)>> {
                    names
                        .iter()
                        .skip(k)
                        .step_by(threads)
                        .map(|n| {
                            let t = st.load(n, &Device::Cpu)?;
                            let dt = if is_text_linear(n) { GgmlDType::Q8_0 } else { GgmlDType::F16 };
                            let to = format!("model.language_model{}", &n[lm_root.len()..]);
                            Ok((to, QTensor::quantize(&t, dt)?))
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a quantizing thread failed")))).collect()
    });
    let mut q: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    for p in parts {
        q.extend(p?);
    }
    q.sort_by(|a, b| a.0.cmp(&b.0));
    let refs: Vec<(&str, &QTensor)> = q.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let meta = gguf_file::Value::String(source);
    crate::convrot_cache::write_atomic(&entry, |tmp| {
        let mut f = std::fs::File::create(tmp).with_context(|| format!("create {}", tmp.display()))?;
        gguf_file::write(&mut f, &[(GGUF_SOURCE, &meta)], &refs)?;
        Ok(())
    })?;
    tracing::info!(path = %entry.display(), tensors = refs.len(), elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()), "text encoder cache built");
    Ok(entry)
}

/// The source fingerprint a cached text GGUF was built from, if it is readable.
fn gguf_source(path: &Path) -> Option<String> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let content = candle_core::quantized::gguf_file::Content::read(&mut f).ok()?;
    match content.metadata.get(GGUF_SOURCE)? {
        candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// The language model's seven decoder linears (Q8_0 in the cache).
fn is_text_linear(name: &str) -> bool {
    const SUFFIXES: [&str; 7] =
        [".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight", ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"];
    name.contains(".layers.") && SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// A safetensors backend that answers the names a loader asks for from a file that stores them
/// under other names (`map`), reshaping a tensor whose stored shape differs from the asked one
/// only by size-1 axes (ComfyUI's VAE keeps the Wan convolutions' singleton time axis:
/// `[out, in, 1, h, w]` for the loader's `[out, in, h, w]`).
pub struct Renamed {
    st: MmapedSafetensors,
    map: fn(&str) -> String,
}

impl Renamed {
    /// Memory-map `files`.
    ///
    /// # Safety
    /// As `MmapedSafetensors::multi`: the files must not change while mapped.
    pub unsafe fn new(files: &[PathBuf], map: fn(&str) -> String) -> Result<Self> {
        Ok(Self { st: unsafe { MmapedSafetensors::multi(files)? }, map })
    }

    /// A `VarBuilder` over this backend.
    pub fn var_builder(self, dtype: DType, dev: &Device) -> candle_nn::VarBuilder<'static> {
        candle_nn::VarBuilder::from_backend(Box::new(self), dtype, dev.clone())
    }
}

/// The two shapes hold the same values in the same order, apart from size-1 axes.
fn same_but_singletons(a: &[usize], b: &[usize]) -> bool {
    let strip = |s: &[usize]| s.iter().copied().filter(|&d| d != 1).collect::<Vec<_>>();
    strip(a) == strip(b)
}

impl candle_nn::var_builder::SimpleBackend for Renamed {
    fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let stored = (self.map)(name);
        let t = self.st.load(&stored, dev)?;
        let t = if t.dims() == s.dims() {
            t
        } else if same_but_singletons(t.dims(), s.dims()) {
            t.reshape(s)?
        } else {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name} (stored as {stored})"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt());
        };
        t.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.st.load(&(self.map)(name), dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.st.get(&(self.map)(name)).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This PC's ComfyUI `models` folder, when there.
    fn comfy_models() -> Option<PathBuf> {
        let p = PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models");
        p.is_dir().then_some(p)
    }

    #[test]
    fn residual_members_and_stage_names() {
        assert_eq!(qwen21_vae_name("quant_conv.weight"), "conv1.weight");
        assert_eq!(qwen21_vae_name("decoder.conv_in.bias"), "decoder.conv1.bias");
        assert_eq!(qwen21_vae_name("decoder.norm_out.gamma"), "decoder.head.0.gamma");
        assert_eq!(qwen21_vae_name("encoder.mid_block.resnets.1.conv2.weight"), "encoder.middle.2.residual.6.weight");
        assert_eq!(qwen21_vae_name("encoder.mid_block.attentions.0.to_qkv.bias"), "encoder.middle.1.to_qkv.bias");
        assert_eq!(qwen21_vae_name("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"), "encoder.downsamples.1.downsamples.0.shortcut.weight");
        assert_eq!(qwen21_vae_name("encoder.down_blocks.2.downsampler.time_conv.bias"), "encoder.downsamples.2.downsamples.2.time_conv.bias");
        assert_eq!(qwen21_vae_name("decoder.up_blocks.0.upsampler.resample.1.weight"), "decoder.upsamples.0.upsamples.3.resample.1.weight");
        assert_eq!(qwen21_vae_name("decoder.up_blocks.4.resnets.2.norm1.gamma"), "decoder.upsamples.4.upsamples.2.residual.0.gamma");
        assert!(same_but_singletons(&[1152, 64, 1, 3, 3], &[1152, 64, 3, 3]));
        assert!(!same_but_singletons(&[64, 1152, 3, 3], &[1152, 64, 3, 3]));
    }

    /// Every tensor of the snapshot's VAE has its ComfyUI counterpart with the same values'
    /// shape, and nothing in ComfyUI's file is left over (a bijection, on the real headers).
    #[test]
    fn the_vae_map_is_a_bijection_on_the_real_files() -> Result<()> {
        let snap = PathBuf::from(r"C:\Users\5090\Documents\_eval\weights\qwen-image-2.1\vae\diffusion_pytorch_model.safetensors");
        let Some(models) = comfy_models().filter(|_| snap.is_file()) else { return Ok(()) };
        let comfy = models.join("vae").join("qwen_image_2.1_vae_bf16.safetensors");
        let (a, b) = unsafe { (MmapedSafetensors::new(&snap)?, MmapedSafetensors::new(&comfy)?) };
        let mut used = std::collections::HashSet::new();
        for (name, view) in a.tensors() {
            let to = qwen21_vae_name(&name);
            let other = b.get(&to).with_context(|| format!("{name} -> {to} is not in ComfyUI's file"))?;
            assert!(same_but_singletons(view.shape(), other.shape()), "{name} {:?} vs {to} {:?}", view.shape(), other.shape());
            used.insert(to);
        }
        assert_eq!(used.len(), b.tensors().len(), "every ComfyUI tensor is asked for exactly once");
        Ok(())
    }

    /// The assembled tokenizer gives Qwen's `tokenizer.json`'s ids, special tokens included.
    #[test]
    fn the_assembled_tokenizer_matches_qwens() -> Result<()> {
        let reference = PathBuf::from(r"C:\Users\5090\Documents\_eval\weights\qwen-image-2.1\processor\tokenizer.json");
        let Some(dir) = comfy_models().and_then(|m| tokenizer_dir(&m)).filter(|_| reference.is_file()) else { return Ok(()) };
        let ours = qwen_tokenizer(&dir)?;
        let theirs = Tokenizer::from_file(&reference).map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompts = [
            "a red fox in the snow, 35mm photo",
            "<|im_start|>system\nDescribe the image.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Replace the cup with a glass of wine.<|im_end|>\n",
            "Café crème — naïve résumé, 東京の夜景, 3.14159 and 2026-10-27!!",
            "  leading spaces\n\n\ttabs and\r\nnewlines   ",
            "e\u{301}clair (combining accent) and emoji 🦊🔥",
            "<think>it's they're we've I'm you'll he'd</think>",
        ];
        for p in prompts {
            let a = ours.encode(p, false).map_err(|e| anyhow::anyhow!("{e}"))?;
            let b = theirs.encode(p, false).map_err(|e| anyhow::anyhow!("{e}"))?;
            assert_eq!(a.get_ids(), b.get_ids(), "{p:?}");
            let back = ours.decode(a.get_ids(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
            let back_ref = theirs.decode(b.get_ids(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
            assert_eq!(back, back_ref);
        }
        Ok(())
    }

    #[test]
    fn files_are_found_by_name_and_need_all_three_and_a_tokenizer() -> Result<()> {
        let d = std::env::temp_dir().join(format!("qir-comfy-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let models = d.join("models");
        for (dir, file) in [("diffusion_models", "qwen_image_2.1_bf16.safetensors"), ("text_encoders", "qwen3vl_8b_bf16.safetensors")] {
            std::fs::create_dir_all(models.join(dir))?;
            std::fs::write(models.join(dir).join(file), b"x")?;
        }
        assert!(find_qwen21(&models).is_none(), "no VAE yet");
        std::fs::create_dir_all(models.join("vae"))?;
        std::fs::write(models.join("vae").join("qwen_image_2.1_vae_bf16.safetensors"), b"x")?;
        assert!(find_qwen21(&models).is_none(), "no tokenizer yet");
        let tok = d.join("comfy").join("text_encoders").join("qwen25_tokenizer");
        std::fs::create_dir_all(&tok)?;
        for f in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
            std::fs::write(tok.join(f), b"{}")?;
        }
        let found = find_qwen21(&models).context("all there")?;
        assert_eq!(found.tokenizer, tok);
        assert!(find_krea2(&models).is_none());
        let _ = std::fs::remove_dir_all(&d);
        Ok(())
    }
}
