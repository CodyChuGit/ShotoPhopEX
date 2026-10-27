//! CPU fixtures come from ComfyUI functions; the ignored test streams installed checkpoints.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use image::RgbaImage;
use qwen_image_rs::layered_pipeline::{euler_cfg, layered_noise, layered_schedule, layered_timestep, retained_slices, rgba_image, LayeredPipeline};
use qwen_image_rs::model::qwen_image_dit::{patchify, unpatchify};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, max_limit: f32, rms_limit: f32) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{name}");
    let actual = actual.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()), "{name}: non-finite tensor");
    let max = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let error: f64 = actual.iter().zip(&expected).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
    let energy: f64 = expected.iter().map(|b| (*b as f64).powi(2)).sum();
    let relative = (error / energy.max(1e-24)).sqrt();
    eprintln!("{name}: max abs {max:.9}, relative RMS {relative:.9}");
    assert!(max <= max_limit && relative <= rms_limit as f64, "{name}: max {max}, relative RMS {relative}");
    Ok(())
}

#[test]
fn comfy_noise_packing_schedule_timestep_and_euler() -> anyhow::Result<()> {
    let started = Instant::now();
    let dev = Device::Cpu;
    let fixtures = candle_core::safetensors::load(fixture("layered_pipeline.bin"), &dev)?;
    let get = |key: &str| fixtures.get(key).ok_or_else(|| anyhow::anyhow!("missing fixture {key}"));
    for seed in [7, 4_294_967_303] {
        let noise = layered_noise(seed, 2, 4, 6, &dev)?;
        compare("noise", &noise, get(&format!("noise_{seed}"))?, 2e-6, 2e-7)?;
        let packed = patchify(&noise)?;
        assert_eq!(packed.dims(), [1, 18, 64]);
        compare("packed", &packed, get(&format!("packed_{seed}"))?, 2e-6, 2e-7)?;
        let unpacked = unpatchify(&packed, 3, 4, 6)?;
        compare("unpacked", &unpacked, get(&format!("unpacked_{seed}"))?, 2e-6, 2e-7)?;
        compare("round trip", &unpacked, &noise, 0.0, 0.0)?;
    }
    let schedule = layered_schedule(20)?;
    compare("all 21 sigmas", &Tensor::from_slice(&schedule.sigmas, 21, &dev)?, get("sigmas")?, 0.0, 0.0)?;
    let times = schedule.sigmas.iter().map(|s| layered_timestep(*s, &dev)?.to_vec1::<f32>().map_err(Into::into)).collect::<anyhow::Result<Vec<_>>>()?;
    compare("all timestep inputs", &Tensor::from_vec(times.into_iter().flatten().collect::<Vec<_>>(), 21, &dev)?, get("timesteps")?, 0.0, 0.0)?;
    assert_ne!(get("timesteps")?.to_vec1::<f32>()?, get("qwen21_rounded")?.to_vec1::<f32>()?);
    let updated = euler_cfg(get("noise_7")?, get("positive")?, get("negative")?, schedule.sigmas[7], schedule.sigmas[8])?;
    compare("Euler with both CFG branches", &updated, get("euler")?, 5e-7, 2e-7)?;
    eprintln!("sampler fixtures: {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}

#[test]
fn output_slice_order_and_saveimage_quantization() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let values: Vec<f32> = (0..16 * 4 * 2 * 4).map(|i| i as f32).collect();
    let latent = Tensor::from_vec(values, (1, 16, 4, 2, 4), &dev)?;
    let retained = retained_slices(&latent)?;
    assert_eq!(retained.dims(), [3, 16, 2, 4]);
    for layer in 0..3 {
        compare("retained order", &retained.narrow(0, layer, 1)?, &latent.narrow(2, layer + 1, 1)?.squeeze(2)?, 0.0, 0.0)?;
    }
    let image = Tensor::from_slice(&[-2f32, 2., 0., 0.5, 1., -1., -1., 1.], (1, 4, 1, 2), &dev)?;
    let image = rgba_image(&image)?;
    assert_eq!(image.dimensions(), (2, 1));
    assert_eq!(image.as_raw(), &[0, 127, 255, 0, 255, 191, 0, 255]);
    Ok(())
}

#[test]
fn invalid_sampler_inputs_return_errors() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    for (layers, h, w) in [(0, 4, 4), (1, 0, 4), (1, 3, 4), (usize::MAX, 4, 4)] {
        assert!(layered_noise(0, layers, h, w, &dev).is_err());
    }
    assert!(layered_schedule(0).is_err());
    assert!(layered_timestep(f32::NAN, &dev).is_err());
    let x = Tensor::zeros((1, 16, 1, 2, 2), DType::F32, &dev)?;
    assert!(retained_slices(&x).is_err());
    assert!(euler_cfg(&x, &x, &x, 0.0, 0.0).is_err());
    assert!(euler_cfg(&x, &x, &x, 0.5, 0.6).is_err());
    Ok(())
}

/// QIR_LAYERED_* overrides also provide a full-size GPU capture harness without a CLI change.
#[test]
#[ignore = "installed 7B encoder and 60-block DiT; CPU one-step determinism/parity"]
fn installed_one_step_smoke_and_capture() -> anyhow::Result<()> {
    let started = Instant::now();
    let root = std::env::var_os("QIR_COMFY_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models"));
    let gpu = std::env::var("QIR_LAYERED_DEVICE").is_ok_and(|v| v == "cuda");
    let device = if gpu { Device::new_cuda(0)? } else { Device::Cpu };
    let option = |key: &str, default: usize| -> anyhow::Result<usize> { Ok(std::env::var(key).ok().map(|v| v.parse()).transpose()?.unwrap_or(default)) };
    let (w, h, layers, steps, seed) = (
        option("QIR_LAYERED_WIDTH", 64)?,
        option("QIR_LAYERED_HEIGHT", 64)?,
        option("QIR_LAYERED_LAYERS", 2)?,
        option("QIR_LAYERED_STEPS", 1)?,
        option("QIR_LAYERED_SEED", 7)? as u64,
    );
    let prompt = std::env::var("QIR_LAYERED_PROMPT").unwrap_or_else(|_| "A red cat.".into());
    let picture = match std::env::var_os("QIR_LAYERED_INPUT") {
        Some(path) => image::open(path)?.into_rgba8(),
        None => RgbaImage::from_raw(w as u32, h as u32, (0..w * h * 4).map(|i| (i % 256) as u8).collect()).unwrap(),
    };
    assert_eq!(picture.dimensions(), (w as u32, h as u32));
    let pipeline = LayeredPipeline::load_on_device(root, device)?;
    let cancel = AtomicBool::new(false);
    let mut captured = HashMap::new();
    let mut progress = Vec::new();
    let images = pipeline.split_observed(
        &picture,
        layers,
        &prompt,
        seed,
        steps,
        |done, total| progress.push((done, total)),
        &cancel,
        |name, tensor| {
            let tensor = tensor.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
            assert!(tensor.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()), "{name}");
            captured.insert(name.to_owned(), tensor);
            Ok(())
        },
    )?;
    assert_eq!(progress, (0..=steps).map(|i| (i, steps)).collect::<Vec<_>>());
    assert_eq!(images.len(), layers);
    assert!(images.iter().all(|im| im.dimensions() == picture.dimensions()));
    assert_eq!(captured["retained"].dims(), [layers, 16, h / 8, w / 8]);
    if let Some(path) = std::env::var_os("QIR_LAYERED_CAPTURE") {
        candle_core::safetensors::save(&captured, path)?;
    }
    if !gpu && (w, h, layers, steps, seed) == (64, 64, 2, 1, 7) && prompt == "A red cat." && std::env::var_os("QIR_LAYERED_INPUT").is_none() {
        let oracle = candle_core::safetensors::load(fixture("layered_pipeline_cpu.bin"), &Device::Cpu)?;
        assert_eq!(captured.len(), oracle.len());
        for (key, expected) in oracle {
            // Bounds include the measured CPU attention error and its propagation through decode.
            let (max, rms) = match key.as_str() {
                "noise" => (2e-6, 2e-7),
                "reference" => (1e-6, 5e-7),
                "positive_text" | "negative_text" => (0.001, 1e-5),
                "step_0.positive" | "step_0.negative" => (0.001, 2e-4),
                "step_0.latent" | "retained" => (0.002, 0.001),
                _ => (0.025, 0.002),
            };
            compare(&key, &captured[&key], &expected, max, rms)?;
        }
    }
    eprintln!("first split and comparisons: {:.3}s", started.elapsed().as_secs_f64());
    if std::env::var_os("QIR_LAYERED_ONCE").is_none() {
        let again = pipeline.split_observed(
            &picture,
            layers,
            &prompt,
            seed,
            steps,
            |_, _| {},
            &cancel,
            |name, tensor| compare(&format!("determinism/{name}"), tensor, &captured[name], 0.0, 0.0),
        )?;
        assert_eq!(images, again);
    }
    let cancelled = AtomicBool::new(true);
    assert!(pipeline.split(&picture, layers, &prompt, seed, steps, |_, _| {}, &cancelled).is_err());
    assert!(pipeline.split(&picture, 0, &prompt, seed, steps, |_, _| {}, &cancel).is_err());
    assert!(pipeline.split(&picture, layers, &prompt, seed, 0, |_, _| {}, &cancel).is_err());
    eprintln!("smoke total: {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}
