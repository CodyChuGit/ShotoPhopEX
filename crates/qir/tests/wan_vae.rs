//! Regenerate with tools/qir_wan_vae_fixtures.py; checkpoint paths can be overridden
//! with QIR_LAYERED_VAE and QIR_RGB_VAE. Missing checkpoints skip parity only.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use qwen_image_rs::model::wan_vae::{rgba_to_rgba_u8, to_rgba_u8, WanVae};
use std::path::{Path, PathBuf};

fn compare(label: &str, actual: &Tensor, expected: &Tensor) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    let error = (actual - expected)?;
    let max = error.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let rms = error.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    let reference_rms = expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    eprintln!("{label}: max abs {max:.9}, relative RMS {:.9}", rms / reference_rms.max(1e-12));
    assert!(max < 1e-3, "{label}: max abs {max} >= 1e-3");
    Ok(())
}

fn parity(name: &str, variable: &str, channels: usize) -> anyhow::Result<()> {
    let path = std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\vae").join(format!("{name}.safetensors")));
    if !path.is_file() {
        eprintln!("Skipping {name}: checkpoint absent: {}", path.display());
        return Ok(());
    }
    let device = Device::Cpu;
    let weights = candle_core::safetensors::load(&path, &device)?;
    let model = WanVae::load(VarBuilder::from_tensors(weights, DType::F32, &device))?;
    assert_eq!(model.image_channels(), channels);
    assert!(model.can_encode());
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.safetensors"));
    let fixtures = candle_core::safetensors::load(fixture_path, &device)?;
    let get = |key: &str| fixtures.get(key).ok_or_else(|| anyhow::anyhow!("missing fixture {key}"));
    for case in ["soft", "opaque"] {
        let encoded = model.encode(get(&format!("{case}.input"))?)?;
        compare(&format!("{name}/{case}/normalized"), &encoded, get(&format!("{case}.encoded"))?)?;
        compare(&format!("{name}/{case}/posterior"), &model.denormalize(&encoded)?, get(&format!("{case}.mean"))?)?;
    }
    if channels == 4 {
        let rgb = get("opaque.input")?;
        let rgba = Tensor::cat(&[rgb, &Tensor::ones((1, 1, 64, 64), DType::F32, &device)?], 1)?;
        compare("RGB padding equals explicit opaque alpha", &model.encode(rgb)?, &model.encode(&rgba)?)?;
    }
    let decoded = model.decode(get("latent")?)?;
    let expected = get("decoded")?;
    compare(&format!("{name}/decode RGB"), &decoded.narrow(1, 0, 3)?, &expected.narrow(1, 0, 3)?)?;
    if channels == 4 {
        compare("layered/decode alpha", &decoded.narrow(1, 3, 1)?, &expected.narrow(1, 3, 1)?)?;
    }
    Ok(())
}

#[test]
fn layered_cpu_f32() -> anyhow::Result<()> {
    parity("qwen_image_layered_vae", "QIR_LAYERED_VAE", 4)
}

#[test]
fn krea_rgb_cpu_f32() -> anyhow::Result<()> {
    parity("wan_2.1_vae", "QIR_RGB_VAE", 3)
}

#[test]
fn rgba_bytes_preserve_alpha_and_rgb_bytes_stay_opaque() -> anyhow::Result<()> {
    let image = Tensor::from_slice(&[-2f32, 2., 0., 0.5, 1., -1., -1., 1.], (1, 4, 1, 2), &Device::Cpu)?;
    let (w, h, bytes) = rgba_to_rgba_u8(&image)?;
    assert_eq!((w, h), (2, 1));
    assert_eq!(bytes, [0, 128, 255, 0, 255, 191, 0, 255]);
    let (_, _, rgb) = to_rgba_u8(&image.narrow(1, 0, 3)?)?;
    assert_eq!(rgb, [0, 128, 255, 255, 255, 191, 0, 255]);
    assert!(rgba_to_rgba_u8(&image.narrow(1, 0, 3)?).is_err());
    Ok(())
}
