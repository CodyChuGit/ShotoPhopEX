use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use gn_segment::ops::{channel_layer_norm, resize_bilinear_antialias, roi_align};
use std::{collections::HashMap, path::Path};

fn fixture(name: &str) -> Result<HashMap<String, Tensor>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    Ok(candle_core::safetensors::load(path, &Device::Cpu)?)
}

fn tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors.get(name).with_context(|| format!("fixture has no {name}"))
}

fn compare(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual.iter().all(|v| v.is_finite()));
    let error = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(error < 1e-5, "max absolute error {error}");
    Ok(())
}

#[test]
fn roi_align_matches_torchvision() -> Result<()> {
    let f = fixture("roi_align.safetensors")?;
    let boxes: Vec<[f32; 4]> =
        tensor(&f, "boxes")?.to_vec2::<f32>()?.into_iter().map(|b| <[f32; 4]>::try_from(b).map_err(|_| anyhow::anyhow!("box shape"))).collect::<Result<_>>()?;
    compare(&roi_align(tensor(&f, "features")?, &boxes)?, tensor(&f, "expected")?)?;
    let empty = roi_align(tensor(&f, "features")?, &[])?;
    assert_eq!(empty.dims(), &[0, 3, 7, 7]);
    Ok(())
}

#[test]
fn resize_matches_pytorch() -> Result<()> {
    let f = fixture("resize.safetensors")?;
    for name in ["down35", "down4", "up2", "nonsquare", "mixed", "singleton", "identity"] {
        let expected = tensor(&f, &format!("{name}.expected"))?;
        let (_, _, h, w) = expected.dims4()?;
        compare(&resize_bilinear_antialias(tensor(&f, &format!("{name}.input"))?, h, w)?, expected).with_context(|| name.to_string())?;
    }
    Ok(())
}

#[test]
fn channel_layer_norm_matches_pytorch() -> Result<()> {
    let f = fixture("channel_layer_norm.safetensors")?;
    compare(&channel_layer_norm(tensor(&f, "input")?, tensor(&f, "weight")?, tensor(&f, "bias")?, 1e-5)?, tensor(&f, "expected")?)
}

#[test]
fn invalid_inputs_are_errors() -> Result<()> {
    let x = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    assert!(resize_bilinear_antialias(&x, 0, 4).is_err());
    assert!(resize_bilinear_antialias(&x, usize::MAX, 4).is_err());
    assert!(roi_align(&x, &[[f32::NAN, 0.0, 2.0, 2.0]]).is_err());
    assert!(roi_align(&x, &[[-f32::MAX, 0.0, f32::MAX, 2.0]]).is_err());
    assert!(channel_layer_norm(&x, &x, &x, 1e-5).is_err());
    let affine = Tensor::ones(2, candle_core::DType::F32, &Device::Cpu)?;
    assert!(channel_layer_norm(&x, &affine, &affine, -1.0).is_err());
    assert!(resize_bilinear_antialias(&x.to_dtype(candle_core::DType::F64)?, 2, 2).is_err());
    Ok(())
}
