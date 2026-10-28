//! Full fixtures: portable Python -I tools/qir_qwen25_vision_fixtures.py.
use candle_core::{DType, Device, Tensor};
use qwen_image_rs::model::qwen25_vision::{preprocess, resize_editplus, window_index, Qwen25Vision};
use std::path::PathBuf;

fn compare(key: &str, x: &Tensor, y: &Tensor, tolerance: f32) -> anyhow::Result<()> {
    assert_eq!(x.dims(), y.dims(), "{key}");
    let x = x.to_dtype(DType::F32)?;
    let y = y.to_dtype(DType::F32)?;
    let delta = (&x - &y)?;
    let abs = delta.abs()?.max_all()?.to_scalar::<f32>()?;
    let rms = delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() / y.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt().max(1e-12);
    eprintln!("{key}: max abs {abs:.9}, relative RMS {rms:.9}");
    if key.ends_with("patches") || key.ends_with("resized") {
        assert!(abs <= 1e-6, "{key} abs {abs}");
    }
    assert!(rms <= tolerance, "{key} relative RMS {rms}");
    Ok(())
}

#[test]
fn comfy_small_preprocessing() -> anyhow::Result<()> {
    let f = candle_core::safetensors::load_buffer(include_bytes!("fixtures/qwen25_vision_small.safetensors"), &Device::Cpu)?;
    let (patches, grid) = preprocess(&f["pixels"])?;
    assert_eq!(grid, [1, 4, 6]);
    compare("small.patches", &patches, &f["patches"], 1e-6)?;
    let (idx, cu) = window_index(grid)?;
    assert_eq!(idx.iter().map(|i| *i as i64).collect::<Vec<_>>(), f["window_index"].to_vec1::<i64>()?);
    assert_eq!(cu.iter().map(|i| *i as i64).collect::<Vec<_>>(), f["cu"].to_vec1::<i64>()?);
    for (h, w) in [(18, 26), (16, 16)] {
        let (idx, cu) = window_index([1, h, w])?;
        assert_eq!(idx.iter().map(|i| *i as i64).collect::<Vec<_>>(), f[&format!("index_{h}_{w}")].to_vec1::<i64>()?);
        assert_eq!(cu.iter().map(|i| *i as i64).collect::<Vec<_>>(), f[&format!("cu_{h}_{w}")].to_vec1::<i64>()?);
    }
    let raw: Vec<u8> = (0..57 * 85 * 3).map(|i| ((i * 37 + 19) % 256) as u8).collect();
    let rgb = image::RgbImage::from_raw(85, 57, raw).ok_or_else(|| anyhow::anyhow!("image shape"))?;
    compare("small.resized", &resize_editplus(&rgb)?, &f["resized"], 1e-6)?;
    assert!(window_index([1, 3, 4]).is_err());
    assert!(preprocess(&Tensor::zeros((0, 4, 3), DType::F32, &Device::Cpu)?).is_err());
    Ok(())
}

#[test]
#[ignore = "installed scaled-fp8 model and regenerated full ComfyUI fixtures; CPU only"]
fn comfy_cpu_vision_parity() -> anyhow::Result<()> {
    let models = std::env::var_os("QIR_COMFY_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"));
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target-qwen25-vision/fixtures");
    let model = Qwen25Vision::from_file(models.join("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"))?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("metadata.json"))?)?;
    for case in ["portrait", "landscape", "square"] {
        let f = candle_core::safetensors::load(root.join(format!("{case}.safetensors")), &Device::Cpu)?;
        let path = meta["cases"][case]["path"].as_str().ok_or_else(|| anyhow::anyhow!("missing image path"))?;
        let rgb = image::open(path)?.to_rgb8();
        let pixels = resize_editplus(&rgb)?;
        compare(&format!("{case}.resized"), &pixels, &f["pixels"], 1e-6)?;
        let (patches, reference_grid) = preprocess(&f["pixels"])?;
        compare(&format!("{case}.same_pixels.patches"), &patches, &f["patches"], 1e-6)?;
        let (_, cu) = window_index(reference_grid)?;
        assert_eq!(cu.iter().map(|i| *i as i64).collect::<Vec<_>>(), f["cu"].to_vec1::<i64>()?);
        let required = ["patches", "patch_embed", "window_index", "block0", "block7", "block31", "merged"];
        for key in required {
            assert!(f.contains_key(key), "missing fixture {case}.{key}");
        }
        let mut seen = Vec::new();
        let (out, grid) = model.encode_with_taps(&pixels, |key, x| {
            if let Some(reference) = f.get(key) {
                seen.push(key.to_owned());
                if key == "window_index" {
                    assert_eq!(x.to_vec1::<u32>()?.iter().map(|i| *i as i64).collect::<Vec<_>>(), reference.to_vec1::<i64>()?);
                } else {
                    compare(&format!("{case}.{key}"), x, reference, 1e-4)?;
                }
            }
            Ok(())
        })?;
        assert_eq!(seen, required);
        assert_eq!(grid.iter().map(|i| *i as i64).collect::<Vec<_>>(), f["grid"].flatten_all()?.to_vec1::<i64>()?);
        assert_eq!(out.dims(), [grid[0] * grid[1] * grid[2] / 4, 3584]);
    }
    Ok(())
}
