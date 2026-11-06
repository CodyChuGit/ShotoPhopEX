use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use gn_segment::vision::Vision;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn picture(device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(3 * 1008 * 1008);
    for ch in 0..3 {
        for y in 0i32..1008 {
            for x in 0i32..1008 {
                let mut rgb = if y >= 650 {
                    [60 + x * 30 / 1008, 100 + x * 70 / 1008, 35 + y * 30 / 1008]
                } else {
                    [80 + x * 80 / 1008, 140 + y * 60 / 1008, 220 - y * 70 / 1008]
                };
                if (x - 760).pow(2) + (y - 220).pow(2) < 110 * 110 {
                    rgb = [255, 225, 90];
                }
                if (180..480).contains(&x) && (360..820).contains(&y) {
                    rgb = [170, 75, 50];
                }
                if (250..330).contains(&x) && (440..560).contains(&y) {
                    rgb = [35, 60, 85];
                }
                values.push(rgb[ch] as f32 / 255.0);
            }
        }
    }
    Ok(Tensor::from_vec(values, (1, 3, 1008, 1008), device)?)
}

#[test]
#[ignore = "full SAM 3.1 vision is slow on CPU and requires the real checkpoint"]
fn vision_matches_comfyui_cpu_f32() -> Result<()> {
    let path = std::env::var_os("GN_SAM31_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"));
    if !path.exists() {
        eprintln!("skipping vision parity: checkpoint absent at {}", path.display());
        return Ok(());
    }
    let start = Instant::now();
    let device = Device::Cpu;
    let fixtures = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision.safetensors"), &device)?;
    let get = |key: &str| fixtures.get(key).with_context(|| format!("missing fixture {key}"));
    let model = Vision::load(path, &device)?;
    eprintln!("vision weights loaded in {:.3}s", start.elapsed().as_secs_f64());
    let compare = |name: &str, tensor: &Tensor| -> Result<()> {
        // Per-boundary absolute gates retain headroom over measured CPU f32 drift.
        let tolerance = match name {
            "patch" => 2e-6,
            "block0" => 1e-5,
            "block7" => 1e-4,
            "block31" => 5e-4,
            "detector0" => 4e-6,
            "detector1" => 4e-4,
            "detector2" => 2e-4,
            "interactive0" => 3e-5,
            "interactive1" | "interactive2" => 8e-5,
            "image_pos" => 1e-6,
            _ => anyhow::bail!("unexpected vision boundary {name}"),
        };
        let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|v| v.is_finite()), "{name} contains nonfinite values");
        let mean = flat.iter().map(|&v| v as f64).sum::<f64>() / flat.len() as f64;
        let std = (flat.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / flat.len() as f64).sqrt();
        let stats = [mean, std, flat.iter().copied().fold(f32::INFINITY, f32::min) as f64, flat.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64];
        let expected = get(&format!("{name}.stats"))?.to_vec1::<f32>()?;
        let stats_error = stats.iter().zip(&expected).map(|(a, b)| (a - *b as f64).abs()).fold(0.0, f64::max);
        eprintln!("{name}: stats max_abs={stats_error:.8e}; elapsed={:.3}s", start.elapsed().as_secs_f64());
        for (a, b) in stats.iter().zip(expected) {
            assert!((a - b as f64).abs() < tolerance, "{name}: stats {a} vs {b}");
        }
        if let Some(indices) = fixtures.get(&format!("{name}.indices")) {
            let indices = indices.to_vec1::<i64>()?;
            let expected = get(&format!("{name}.samples"))?.to_vec1::<f32>()?;
            let mut abs = 0f32;
            let mut rel = 0f32;
            let mut scale = 0f32;
            for (&index, &e) in indices.iter().zip(&expected) {
                let a = flat[index as usize];
                let delta = (a - e).abs();
                abs = abs.max(delta);
                rel = rel.max(delta / e.abs().max(1e-6));
                scale = scale.max(e.abs());
            }
            eprintln!("{name}: samples max_abs={abs:.8e}, max_relative (floor 1e-6)={rel:.8e}, relative_inf={:.8e}", abs / scale);
            for (&index, &e) in indices.iter().zip(&expected) {
                assert!((flat[index as usize] - e).abs() < tolerance as f32, "{name}: sample {index}: {} vs {e}", flat[index as usize]);
            }
        }
        Ok(())
    };
    let out = model.forward_with_observer(&picture(&device)?, compare)?;
    for (detector, interactive) in out.detector_fpn.iter().zip(&out.interactive_fpn) {
        assert_eq!(detector.dims(), interactive.dims());
    }
    assert_eq!(out.detector_fpn[0].dims(), [1, 256, 288, 288]);
    assert_eq!(out.detector_fpn[1].dims(), [1, 256, 144, 144]);
    assert_eq!(out.detector_fpn[2].dims(), [1, 256, 72, 72]);
    compare("image_pos", &out.image_pos)?;
    eprintln!("vision parity including load: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
