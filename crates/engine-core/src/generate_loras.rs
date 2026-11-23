//! `generate.loras`: the LoRA adapters in the native engine's models folder (`<modelsPath>/
//! loras`), each with the model family its tensors are for, read from the file's safetensors
//! header alone (no server, no weights loaded). The Generate panel lists the ones for the
//! template it will run, when that template runs on the native engine, whose folder this is
//! (the ComfyUI route applies `loras` too, from its own `models/loras`, which this list does
//! not read).

use serde_json::{Value, json};

use super::opt_str;
use crate::{Result, Session};

pub const LORAS: &str = "generate.loras";

/// A header larger than this is not a LoRA's (a LoRA's names and shapes fit in a few hundred
/// kilobytes; a whole model's in a few megabytes).
const MAX_HEADER: u64 = 32 << 20;

/// The model family a LoRA's tensor names are for: Krea 2's single-stream `blocks.N.attn.gate`,
/// Qwen-Image's double-stream `transformer_blocks.N`, 32 of them in Qwen-Image-2.1 (width 4096)
/// and 60 in Qwen-Image-Edit-2511 (width 3072), MOSS-Image's `layers.N.proj.{q,k,v,qkv,o,
/// ffn_up_gate,ffn_down}` (its turbo, Edit and quality LoRAs). `None` for anything else.
pub(crate) fn lora_family<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    let (mut krea, mut qwen, mut moss, mut last_block) = (false, false, false, 0usize);
    for n in names {
        let block = |marker: &str| n.split_once(marker).and_then(|(_, rest)| rest.split('.').next()).and_then(|i| i.parse::<usize>().ok());
        if let Some(i) = block("transformer_blocks.") {
            qwen = true;
            last_block = last_block.max(i);
        } else if n.contains(".attn.gate") && block("blocks.").is_some() {
            krea = true;
        } else if is_moss(n) {
            moss = true;
        }
    }
    match (krea, qwen, moss) {
        (true, false, false) => Some("krea2"),
        (false, true, false) if last_block < 40 => Some("qwen-image-2.1"),
        (false, true, false) => Some("qwen-image-edit-2511"),
        (false, false, true) => Some("moss-image"),
        _ => None,
    }
}

/// A MOSS-Image adapter tensor: `layers.N.proj.<part>.…`, with `<part>` one of the block's
/// projections (under any prefix: `diffusion_model.`, `transformer.`, none).
fn is_moss(n: &str) -> bool {
    const PARTS: [&str; 7] = ["q", "k", "v", "qkv", "o", "ffn_up_gate", "ffn_down"];
    let Some((_, rest)) = n.split_once("layers.") else { return false };
    let mut it = rest.split('.');
    it.next().is_some_and(|i| i.parse::<usize>().is_ok()) && it.next() == Some("proj") && it.next().is_some_and(|p| PARTS.contains(&p))
}

/// The tensor names in a safetensors file's header, or `None` when it is not one.
fn tensor_names(path: &std::path::Path) -> Option<Vec<String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > MAX_HEADER {
        return None;
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).ok()?;
    let v: Value = serde_json::from_slice(&header).ok()?;
    Some(v.as_object()?.keys().filter(|k| *k != "__metadata__").cloned().collect())
}

/// Whether `template` would run on the native engine: `engine` native, or `auto` with the
/// native folders set and the template's family one `auto` sends there.
pub(crate) fn runs_native(s: &Session, _template: &str) -> bool {
    s.prefs().integrations.native_configured()
}

pub(crate) fn run_loras(s: &mut Session, p: &Value) -> Result<Value> {
    let template = opt_str(LORAS, p, "template", 200)?.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    let family = template.as_deref().and_then(|t| photocraft_genai::template::find(t).ok()).map(|t| t.meta.family.clone());
    let models = s.prefs().integrations.models_root();
    let folder = if models.is_empty() { String::new() } else { std::path::Path::new(&models).join("loras").to_string_lossy().into_owned() };
    let mut loras: Vec<Value> = Vec::new();
    if !folder.is_empty()
        && let Ok(dir) = std::fs::read_dir(&folder)
    {
        let mut files: Vec<std::path::PathBuf> =
            dir.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))).collect();
        files.sort();
        for path in files {
            let Some(file) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
            let found = tensor_names(&path).and_then(|names| lora_family(names.iter().map(String::as_str)));
            if family.as_deref().is_none_or(|f| found == Some(f)) {
                loras.push(json!({"file": file, "family": found}));
            }
        }
    }
    Ok(json!({
        "folder": folder,
        "family": family,
        "native": template.as_deref().is_some_and(|t| runs_native(s, t)),
        "loras": loras,
    }))
}

#[cfg(test)]
mod tests {
    use super::lora_family;

    #[test]
    fn a_lora_is_told_by_its_tensor_names() {
        let krea = ["diffusion_model.blocks.0.attn.gate.lora_A.weight", "diffusion_model.blocks.27.mlp.down.lora_B.weight"];
        assert_eq!(lora_family(krea), Some("krea2"));
        let q21 = ["diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight", "diffusion_model.transformer_blocks.31.img_mlp.net.2.lora_B.weight"];
        assert_eq!(lora_family(q21), Some("qwen-image-2.1"));
        let e2511 = ["transformer_blocks.0.attn.add_k_proj.alpha", "transformer_blocks.59.attn.to_out.0.lora_up.weight"];
        assert_eq!(lora_family(e2511), Some("qwen-image-edit-2511"));
        let moss = ["diffusion_model.layers.0.proj.q.lora_A.weight", "diffusion_model.layers.31.proj.ffn_down.lora_B.weight"];
        assert_eq!(lora_family(moss), Some("moss-image"));
        let kohya = ["diffusion_model.layers.5.proj.qkv.lora_down.weight", "diffusion_model.layers.5.proj.qkv.alpha"];
        assert_eq!(lora_family(kohya), Some("moss-image"));
        assert_eq!(lora_family(["layers.0.attention.q.lora_A.weight"]), None, "another model's layers.N is not MOSS");
        assert_eq!(lora_family(["lora_unet_down_blocks_0.weight"]), None, "an SD-style file is none of ours");
        assert_eq!(lora_family(Vec::<&str>::new()), None);
    }
}
