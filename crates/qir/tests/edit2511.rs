//! ComfyUI oracle fixtures: tools/edit2511_fixtures.py (portable Python, CPU).
use candle_core::{DType, Device, Tensor};
use qwen_image_rs::{
    edit2511_files::{adapter_inventory, AdapterModule, Edit2511Files, ADAPTER_NAMES},
    edit2511_prep::{crop_vae_target, latent_noise_mask, resize_kontext, resize_vae_reference},
    lora::LoraFile,
    model::qwen25_vision::resize_editplus,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
};

fn compare(key: &str, actual: &Tensor, expected: &Tensor, exact: bool) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{key}");
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.flatten_all()?.to_vec1::<f32>()?;
    let max = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let sum: f64 = a.iter().zip(&b).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
    let norm: f64 = b.iter().map(|b| (*b as f64).powi(2)).sum();
    let rms = (sum / a.len() as f64).sqrt() / (norm / b.len() as f64).sqrt().max(1e-12);
    eprintln!("{key}: max abs {max:.9e}, relative RMS {rms:.9e}");
    assert!(max <= if exact { 0. } else { 1e-6 }, "{key}: {max}");
    Ok(())
}
fn rgb(pixels: &Tensor) -> anyhow::Result<image::RgbImage> {
    let (h, w, _) = pixels.dims3()?;
    let raw = pixels.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| (v * 255.).round() as u8).collect();
    image::RgbImage::from_raw(w as u32, h as u32, raw).ok_or_else(|| anyhow::anyhow!("RGB shape"))
}
fn stages(pixels: &Tensor) -> anyhow::Result<Vec<(&'static str, Tensor)>> {
    let png = rgb(pixels)?;
    Ok(vec![
        ("vl", resize_editplus(&png)?),
        ("reference", resize_vae_reference(pixels)?),
        ("kontext", resize_kontext(pixels)?),
        ("crop", crop_vae_target(pixels)?),
        ("mask4", latent_noise_mask(&png, &[2, 16, 9, 13])?),
        ("mask5", latent_noise_mask(&png, &[2, 16, 1, 9, 13])?),
    ])
}
#[test]
fn small_comfy_preparation() -> anyhow::Result<()> {
    let f = candle_core::safetensors::load_buffer(include_bytes!("fixtures/edit2511_prep_small.safetensors"), &Device::Cpu)?;
    for case in ["portrait", "landscape", "odd"] {
        for (stage, value) in stages(&f[&format!("{case}.pixels")])? {
            let key = format!("{case}.{stage}");
            let shape = f[&format!("{key}.shape")].to_vec1::<i64>()?;
            assert_eq!(value.dims().iter().map(|n| *n as i64).collect::<Vec<_>>(), shape);
            let flat = value.flatten_all()?;
            let actual = if stage == "reference" || stage == "kontext" || stage == "vl" {
                let src = flat.to_vec1::<f32>()?;
                let n = 4096;
                let data = (0..n).map(|i| src[i * (src.len() - 1) / (n - 1)]).collect::<Vec<_>>();
                Tensor::from_vec(data, n, &Device::Cpu)?
            } else {
                flat
            };
            compare(&key, &actual, &f[&key], stage != "reference" && stage != "vl")?;
        }
    }
    Ok(())
}
#[test]
#[ignore = "regenerate full CPU ComfyUI fixtures with tools/edit2511_fixtures.py"]
fn full_comfy_preparation() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target-edit2511/fixtures");
    for case in ["portrait", "landscape", "odd", "large_portrait", "large_landscape"] {
        let f = candle_core::safetensors::load(root.join(format!("{case}.safetensors")), &Device::Cpu)?;
        for (key, actual) in stages(&f["pixels"])? {
            compare(&format!("{case}.{key}"), &actual, &f[key], key != "reference" && key != "vl")?;
        }
    }
    Ok(())
}
fn models() -> PathBuf {
    std::env::var_os("QIR_COMFY_MODELS").map(PathBuf::from).unwrap_or_else(|| r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models".into())
}
#[test]
#[ignore = "requires installed original 2511 model headers and optional adapter files; CPU only"]
fn installed_inventory_and_comfy_adapter_maps() -> anyhow::Result<()> {
    let files = Edit2511Files::resolve(models())?;
    let maps: BTreeMap<String, Option<BTreeMap<String, AdapterModule>>> = serde_json::from_str(include_str!("fixtures/edit2511_lora_maps.json"))?;
    for (name, path) in ADAPTER_NAMES.into_iter().zip([files.lightning4, files.lightning8, files.remover, files.angles]) {
        let Some(path) = path else {
            eprintln!("{name}: absent");
            continue;
        };
        let expected = maps[name].as_ref().ok_or_else(|| anyhow::anyhow!("regenerate ComfyUI map for newly installed {name}"))?;
        assert_eq!(&adapter_inventory(&path)?, expected, "{name}: header map");
        let file = LoraFile::load(&path, &Device::Cpu, DType::BF16)?;
        let actual: BTreeMap<String, AdapterModule> = file
            .entries
            .iter()
            .map(|(name, e)| {
                let rank = e.rank();
                let alpha = expected[name].alpha.map(|_| e.alpha);
                (name.clone(), AdapterModule { rank, alpha, scale: e.alpha / rank as f32 })
            })
            .collect();
        assert_eq!(&actual, expected, "{name}: actual loader map");
        eprintln!("{name}: {} modules, exact ranks/alphas/scales", actual.len());
    }
    Ok(())
}
#[test]
fn default_suffix_matches_plain_and_down_layouts() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!("qir-edit2511-lora-{}.safetensors", std::process::id()));
    let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    let b = Tensor::from_vec(vec![5f32, 6., 7., 8.], (2, 2), &Device::Cpu)?;
    let mut raw = HashMap::new();
    for (stem, down, up) in [("default", "lora_A.default", "lora_B.default"), ("plain", "lora_A", "lora_B"), ("down", "lora_down", "lora_up")] {
        raw.insert(format!("{stem}.{down}.weight"), a.clone());
        raw.insert(format!("{stem}.{up}.weight"), b.clone());
    }
    candle_core::safetensors::save(&raw, &path)?;
    let file = LoraFile::load(&path, &Device::Cpu, DType::F32)?;
    std::fs::remove_file(path)?;
    assert_eq!(file.entries.len(), 3);
    for e in file.entries.values() {
        assert_eq!(e.rank(), 2);
        assert_eq!(e.alpha, 2.);
        assert_eq!(e.a.to_vec2::<f32>()?, a.to_vec2::<f32>()?);
        assert_eq!(e.b.to_vec2::<f32>()?, b.to_vec2::<f32>()?);
    }
    Ok(())
}
