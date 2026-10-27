//! CPU oracle: tools/qwen_image_dit_fixtures.py (ComfyUI embedded Python, -I).
//! Two generated layers plus composite: [1,16,3,32,32], sin(i*0.017)*0.7;
//! reference [1,16,1,32,32], cos(i*0.013)*0.4; text [1,12,3584], sin(i*0.019)*0.3.
//! Independent blocks use image sin(i*0.007)*0.2, text cos(i*0.011)*0.2,
//! and temb sin(i*0.003)*0.1. Slow checks load only one checkpoint block at a time.

use candle_core::{DType, Device, Tensor};
use qwen_image_rs::model::qwen_image_dit::{
    checkpoint_builder, image_positions, patchify, text_positions, timestep_projection, unpatchify, LayeredBlock, LayeredDit, LayeredInput, LayeredOutput,
    LayeredRope,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
struct Sample {
    shape: Vec<usize>,
    values: Vec<f32>,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

#[derive(Deserialize)]
struct Fixtures {
    tensors: HashMap<String, Sample>,
}

fn fixture() -> anyhow::Result<Fixtures> {
    Ok(serde_json::from_slice(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen_image_dit.json"))?)?)
}

fn checkpoint() -> Option<PathBuf> {
    let path = std::env::var_os("QIR_LAYERED_DIT").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\qwen_image_layered_fp8mixed.safetensors")
    });
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("Skipping absent Layered checkpoint: {}", path.display());
        None
    }
}

fn formula(shape: &[usize], frequency: f32, amplitude: f32, cosine: bool) -> anyhow::Result<Tensor> {
    let data: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| {
            let phase = i as f32 * frequency;
            (if cosine { phase.cos() } else { phase.sin() }) * amplitude
        })
        .collect();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

fn compare(fixtures: &Fixtures, name: &str, actual: &Tensor, tolerance: f64) -> anyhow::Result<()> {
    let expected = fixtures.tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing fixture {name}"))?;
    assert_eq!(actual.dims(), expected.shape, "{name}");
    let flat = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f64;
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    let mut max_relative = 0f64;
    for (i, &reference) in expected.values.iter().enumerate() {
        let index = i * (flat.len() - 1) / (expected.values.len() - 1).max(1);
        let error = f64::from(flat[index]) - f64::from(reference);
        max_abs = max_abs.max(error.abs());
        max_relative = max_relative.max(error.abs() / f64::from(reference).abs().max(1e-6));
        squared_error += error * error;
        squared_reference += f64::from(reference).powi(2);
    }
    let n = flat.len() as f64;
    let mean = flat.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let std = (flat.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min = flat.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let relative_rms = (squared_error / squared_reference.max(1e-20)).sqrt();
    eprintln!("{name}: max abs {max_abs:0.9}, max relative {max_relative:0.9}, relative RMS {relative_rms:0.9}; mean {mean:0.9}, std {std:0.9}, min {min:0.9}, max {max:0.9}");
    assert!(max_abs < tolerance, "{name}: max abs {max_abs} >= {tolerance}");
    // Sixty blocks accumulate CPU GEMM/softmax ordering differences: the observed
    // full-forward baseline is 3.02e-4; individual modules remain below 4.9e-6.
    let relative_limit = if name == "full_f32_velocity" { 5e-4 } else { 1e-4 };
    assert!(relative_rms < relative_limit, "{name}: relative RMS {relative_rms} >= {relative_limit}");
    for (label, actual, reference) in [("mean", mean, expected.mean), ("std", std, expected.std), ("min", min, expected.min), ("max", max, expected.max)] {
        assert!((actual - reference).abs() < tolerance, "{name}/{label}: {actual} vs {reference}");
    }
    Ok(())
}

fn positions() -> anyhow::Result<(Tensor, Tensor, LayeredRope)> {
    let img = Tensor::cat(&[image_positions(1, 3, 16, 16, 0, &Device::Cpu)?, image_positions(1, 1, 16, 16, -1, &Device::Cpu)?], 1)?;
    let txt = text_positions(1, 12, 16, 16, &Device::Cpu)?;
    let rope = LayeredRope::new(&Tensor::cat(&[&txt, &img], 1)?, DType::F32)?;
    Ok((img, txt, rope))
}

fn inputs() -> anyhow::Result<(Tensor, Tensor)> {
    let latent = formula(&[1, 16, 3, 32, 32], 0.017, 0.7, false)?;
    let reference = formula(&[1, 16, 1, 32, 32], 0.013, 0.4, true)?;
    Ok((Tensor::cat(&[patchify(&latent)?, patchify(&reference)?], 1)?, formula(&[1, 12, 3584], 0.019, 0.3, false)?))
}

#[test]
fn patches_positions_rope_and_time_projection() -> anyhow::Result<()> {
    let fixtures = fixture()?;
    let latent = formula(&[1, 16, 3, 32, 32], 0.017, 0.7, false)?;
    let patches = patchify(&latent)?;
    compare(&fixtures, "patches", &patches, 2e-7)?;
    let inverse = unpatchify(&patches, 3, 32, 32)?;
    assert_eq!(inverse.flatten_all()?.to_vec1::<f32>()?, latent.flatten_all()?.to_vec1::<f32>()?);
    compare(&fixtures, "reference_patches", &patchify(&formula(&[1, 16, 1, 32, 32], 0.013, 0.4, true)?)?, 2e-7)?;
    let (img, txt, rope) = positions()?;
    compare(&fixtures, "image_ids", &img, 1e-7)?;
    compare(&fixtures, "text_ids", &txt, 1e-7)?;
    compare(&fixtures, "rope", &rope.matrices()?, 2e-7)?;
    // libm and PyTorch's vector exp differ by one ulp before the 1000x phase scale.
    compare(&fixtures, "time_projection", &timestep_projection(&Tensor::new(&[0.625f32], &Device::Cpu)?)?, 4e-5)?;
    // Odd token grids use integer division for both centering and the text start.
    assert_eq!(text_positions(1, 1, 7, 5, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?, [3., 3., 3.]);
    assert!(patchify(&Tensor::zeros((1, 16, 1, 3, 4), DType::F32, &Device::Cpu)?).is_err());
    Ok(())
}

#[test]
#[ignore = "loads selected checkpoint modules, CPU only"]
fn selected_checkpoint_modules() -> anyhow::Result<()> {
    let Some(path) = checkpoint() else { return Ok(()) };
    let fixtures = fixture()?;
    let vb = checkpoint_builder(path, &Device::Cpu)?;
    let start = Instant::now();
    let input = LayeredInput::load(vb.clone())?;
    let time = Tensor::new(&[0.625f32], &Device::Cpu)?;
    compare(&fixtures, "time_embedding", &input.timestep(&time, None)?, 1e-3)?;
    compare(&fixtures, "time_embedding_one", &input.timestep(&time, Some(&Tensor::new(&[1u32], &Device::Cpu)?))?, 1e-3)?;
    let (img, txt) = inputs()?;
    let (img, txt) = input.project(&img, &txt)?;
    compare(&fixtures, "image_projection", &img, 1e-3)?;
    compare(&fixtures, "text_projection", &txt, 1e-3)?;
    drop((input, img, txt));
    eprintln!("input modules: {:0.3}s", start.elapsed().as_secs_f64());
    let (_, _, rope) = positions()?;
    let img = formula(&[1, 1024, 3072], 0.007, 0.2, false)?;
    let txt = formula(&[1, 12, 3072], 0.011, 0.2, true)?;
    let temb = formula(&[1, 3072], 0.003, 0.1, false)?;
    for i in [0, 29, 59] {
        let start = Instant::now();
        let block = LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))?;
        let (image, text) = block.forward(&img, &txt, &temb, &rope)?;
        compare(&fixtures, &format!("block_{i}_image"), &image, 5e-3)?;
        compare(&fixtures, &format!("block_{i}_text"), &text, 5e-3)?;
        eprintln!("block {i}: {:0.3}s", start.elapsed().as_secs_f64());
    }
    let start = Instant::now();
    let output = LayeredOutput::load(vb)?;
    compare(&fixtures, "norm_out", &output.normalize(&img, &temb)?, 1e-3)?;
    compare(&fixtures, "proj_out", &output.forward(&img, &temb)?, 1e-3)?;
    eprintln!("output modules: {:0.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

#[test]
#[ignore = "streams all 60 blocks on CPU"]
fn full_streamed_velocity() -> anyhow::Result<()> {
    let Some(path) = checkpoint() else { return Ok(()) };
    let start = Instant::now();
    let model = LayeredDit::open_streamed(path, &Device::Cpu)?;
    let (img, txt) = inputs()?;
    let (ip, tp, _) = positions()?;
    let result = model.forward(&img, &txt, &Tensor::new(&[0.625f32], &Device::Cpu)?, &ip, &tp, None)?.narrow(1, 0, 768)?;
    compare(&fixture()?, "full_f32_velocity", &result, 2e-3)?;
    eprintln!("full streamed CPU f32: {:0.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
