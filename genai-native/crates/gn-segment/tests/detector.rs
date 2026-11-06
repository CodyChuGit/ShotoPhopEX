use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use gn_segment::{detector::Detector, position::image_position};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|x| x.is_finite()));
    let abs = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
    let rel = actual.iter().zip(expected).map(|(a, b)| (a - b).abs() / b.abs().max(1e-6)).fold(0_f32, f32::max);
    eprintln!("{name}: max_abs={abs:.8e}, max_relative(floor 1e-6)={rel:.8e}");
    assert!(abs < tolerance, "{name} max absolute error {abs} >= {tolerance}");
}

#[test]
#[ignore = "full 5184-token encoder and 200 high-resolution masks; CPU parity"]
fn detector_matches_comfyui_f32() -> Result<()> {
    let path = std::env::var_os("GN_SAM31_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"));
    if !path.exists() {
        eprintln!("skipping detector parity: checkpoint absent at {}", path.display());
        return Ok(());
    }
    let start = Instant::now();
    let device = Device::Cpu;
    let model = Detector::load(path, &device)?;
    eprintln!("detector load: {:.3}s", start.elapsed().as_secs_f64());
    let fixtures = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/detector.safetensors"), &device)?;
    let get = |key: &str| fixtures.get(key).with_context(|| format!("missing fixture {key}"));
    // Shared f32 formula: fpn_k[c,y,x] = 0.5*sin(((.013*c+.071*y)+.031*x)+k).
    // Detector FPN has no tracker no-memory embedding. Prompt p[t,c]=.3*cos(.05*c+.7*t).
    let mut fpn = Vec::new();
    for (k, size) in [288, 144, 72].into_iter().enumerate() {
        let values: Vec<f32> = (0..256)
            .flat_map(|c| {
                (0..size).flat_map(move |y| (0..size).map(move |x| 0.5 * (((0.013 * c as f32 + 0.071 * y as f32) + 0.031 * x as f32) + k as f32).sin()))
            })
            .collect();
        fpn.push(Tensor::from_vec(values, (1, 256, size, size), &device)?);
    }
    let prompt: Vec<f32> = (0..8).flat_map(|t| (0..256).map(move |c| 0.3 * (0.05 * c as f32 + 0.7 * t as f32).cos())).collect();
    let prompt = Tensor::from_vec(prompt, (1, 8, 256), &device)?;
    let valid = Tensor::from_slice(&[1_u8, 1, 1, 1, 1, 1, 0, 0], (1, 8), &device)?;
    let pos = image_position(72, 72, &device)?;
    let forward = Instant::now();
    let (out, refs) = model.forward_with_references([&fpn[0], &fpn[1], &fpn[2]], &pos, &prompt, &valid)?;
    eprintln!("detector CPU forward: {:.3}s", forward.elapsed().as_secs_f64());
    assert_eq!(out.boxes.dims(), [1, 200, 4]);
    assert_eq!(out.logits.dims(), [1, 200]);
    assert_eq!(out.presence.dims(), [1, 1]);
    assert_eq!(out.masks.dims(), [1, 200, 288, 288]);
    for (name, tensor, tolerance) in [("boxes", &out.boxes, 1e-4), ("logits", &out.logits, 1e-3), ("presence", &out.presence, 1e-3)] {
        compare(name, &tensor.flatten_all()?.to_vec1::<f32>()?, &get(name)?.flatten_all()?.to_vec1::<f32>()?, tolerance);
    }
    let references = get("references")?.reshape((6, 800))?.to_vec2::<f32>()?;
    for (i, reference) in refs.iter().enumerate() {
        let expected = &references[i];
        compare(&format!("reference layer {i}"), &reference.flatten_all()?.to_vec1::<f32>()?, expected, 1e-4);
    }
    let masks = out.masks.flatten_all()?.to_vec1::<f32>()?;
    let samples: Vec<f32> = (0..4096).map(|i| masks[i * (masks.len() - 1) / 4095]).collect();
    compare("mask samples", &samples, &get("samples")?.to_vec1::<f32>()?, 1e-3);
    let mean = masks.iter().map(|&v| f64::from(v)).sum::<f64>() / masks.len() as f64;
    let std = (masks.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / masks.len() as f64).sqrt();
    let stats = [mean as f32, std as f32, masks.iter().copied().fold(f32::INFINITY, f32::min), masks.iter().copied().fold(f32::NEG_INFINITY, f32::max)];
    compare("mask stats", &stats, &get("stats")?.to_vec1::<f32>()?, 1e-3);
    let logits = out.logits.flatten_all()?.to_vec1::<f32>()?;
    let mut ranked: Vec<usize> = (0..200).collect();
    ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top = get("top_indices")?.to_vec1::<i64>()?;
    let expected = get("top_masks")?.to_vec3::<f32>()?;
    for (i, &index) in top.iter().enumerate() {
        let index = usize::try_from(index)?;
        assert_eq!(ranked[i], index, "top query ranking");
        let expected: Vec<f32> = expected[i].iter().flatten().copied().collect();
        compare(&format!("full mask query {index}"), &masks[index * 288 * 288..(index + 1) * 288 * 288], &expected, 1e-3);
    }
    // Padding must exclude invalid tokens from attention and scoring, even for large values.
    // Malformed dimensions are rejected before expensive transformer work.
    assert!(model.forward([&fpn[0], &fpn[1], &fpn[2]], &pos, &prompt, &valid.narrow(1, 0, 7)?).is_err());
    eprintln!("detector total parity time: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
