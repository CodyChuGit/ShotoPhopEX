use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use gn_segment::{geometry::Geometry, position::image_position};
use std::path::{Path, PathBuf};

fn compare(name: &str, actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual.iter().all(|x| x.is_finite()));
    let abs = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
    let rel = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs() / b.abs().max(1e-6)).fold(0_f32, f32::max);
    let scale = expected.iter().map(|x| x.abs()).fold(0_f32, f32::max).max(1e-6);
    eprintln!("{name}: max_abs={abs:.8e}, max_relative(floor 1e-6)={rel:.8e}, relative_inf={:.8e}", abs / scale);
    for (a, b) in actual.iter().zip(&expected) {
        assert!((a - b).abs() <= 1e-4 + 1e-4 * b.abs(), "{name}: actual {a}, expected {b}, max_abs {abs}");
    }
    Ok(())
}

#[test]
fn geometry_matches_comfyui_f32() -> Result<()> {
    let path = std::env::var_os("GN_SAM31_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"));
    if !path.exists() {
        eprintln!("skipping geometry parity: checkpoint absent at {}", path.display());
        return Ok(());
    }
    let device = Device::Cpu;
    let model = Geometry::load(path, &device)?;
    let fixtures = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/geometry.safetensors"), &device)?;
    let get = |key: &str| fixtures.get(key).with_context(|| format!("missing fixture {key}"));
    // Shared f32 formula: fpn[c,y,x] = 0.5*sin((0.013*c + 0.071*y) + 0.031*x).
    // Level 2 is 72x72, with no tracker no-memory embedding. Image positions are
    // the real DETR y,x sine features, not synthetic position values.
    let values: Vec<f32> = (0..256)
        .flat_map(|c| (0..72).flat_map(move |y| (0..72).map(move |x| 0.5 * ((0.013 * c as f32 + 0.071 * y as f32) + 0.031 * x as f32).sin())))
        .collect();
    let fpn = Tensor::from_vec(values, (1, 256, 72, 72), &device)?;
    let pos = image_position(72, 72, &device)?;
    for name in ["middle", "tiny", "full", "outside", "three"] {
        let boxes = get(&format!("{name}.boxes"))?;
        compare(name, &model.boxes(&fpn, &pos, boxes)?, get(&format!("{name}.tokens"))?)?;
    }
    compare("cls", &model.cls(&fpn, &pos)?, get("cls")?)?;
    // Box attention ignores positions, whereas CLS uses them as image keys.
    let zeros = pos.zeros_like()?;
    let boxes = get("middle.boxes")?;
    compare("boxes ignore positions", &model.boxes(&fpn, &zeros, boxes)?, &model.boxes(&fpn, &pos, boxes)?)?;
    let delta = (model.cls(&fpn, &zeros)? - model.cls(&fpn, &pos)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(delta > 1e-4);
    assert!(model.boxes(&fpn, &pos, &Tensor::zeros((1, 0, 4), candle_core::DType::F32, &device)?).is_err());
    assert!(model.boxes(&fpn, &pos, &Tensor::full(f32::NAN, (1, 1, 4), &device)?).is_err());
    assert!(model.cls(&fpn.narrow(2, 0, 71)?, &pos).is_err());
    Ok(())
}
