//! CPU fixtures come from ComfyUI functions; the ignored test streams installed checkpoints.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use image::RgbaImage;
use qwen_image_rs::layered_pipeline::{euler_cfg, layered_noise, layered_schedule, layered_timestep, retained_slices, rgba_image, LayeredPipeline};
use qwen_image_rs::model::qwen_image_dit::{patchify, unpatchify};

#[path = "support/layered_resize.rs"]
mod layered_resize;

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

/// Time the production path without copying intermediate tensors to the host.
#[test]
#[ignore = "installed Layered checkpoints and exclusive GPU"]
fn installed_benchmark() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let mut started = Instant::now();
    let root = std::env::var_os("QIR_COMFY_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models"));
    let device = Device::new_cuda(0)?;
    let pipeline = LayeredPipeline::load_on_device(root, device.clone())?;
    let input = std::env::var_os("QIR_LAYERED_INPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\swap\gen\gen-tshirt.png"));
    let picture = image::open(input)?.into_rgba8();
    let dimension = |key: &str, default: u32| -> anyhow::Result<u32> { Ok(std::env::var(key).ok().map(|v| v.parse()).transpose()?.unwrap_or(default)) };
    let picture = layered_resize::resize(&picture, dimension("QIR_LAYERED_WIDTH", 640)?, dimension("QIR_LAYERED_HEIGHT", 640)?);
    let export_width = dimension("QIR_LAYERED_EXPORT_WIDTH", 1024)?;
    let export_height = dimension("QIR_LAYERED_EXPORT_HEIGHT", 1024)?;
    let cancel = AtomicBool::new(false);
    let steps = std::env::var("QIR_LAYERED_STEPS").ok().map(|v| v.parse()).transpose()?.unwrap_or(20);
    let repeat = std::env::var("QIR_LAYERED_REPEAT").ok().map(|v| v.parse()).transpose()?.unwrap_or(1);
    let mut previous = None;
    for run in 0..repeat {
        if run > 0 {
            started = Instant::now();
        }
        eprintln!("benchmark run {}", run + 1);
        let sync_steps = std::env::var("QIR_LAYERED_SYNC_STEPS").is_ok_and(|v| v == "1");
        let mut last = Instant::now();
        let split_started = Instant::now();
        let images = pipeline.split(
            &picture,
            3,
            "separate this image into layers: the background and each foreground subject on its own layer",
            7,
            steps,
            |done, _| {
                if sync_steps {
                    device.synchronize().unwrap();
                    eprintln!("benchmark synchronized progress {done}: {:.3}s", last.elapsed().as_secs_f64());
                } else if done.is_multiple_of(5) {
                    eprintln!("benchmark progress {done}/{steps}");
                }
                last = Instant::now();
            },
            &cancel,
        )?;
        device.synchronize()?;
        eprintln!("benchmark pipeline split: {:.3}s", split_started.elapsed().as_secs_f64());
        eprintln!("benchmark split end-to-end: {:.3}s", started.elapsed().as_secs_f64());
        if let Some(previous) = previous {
            assert_eq!(images, previous, "repeat output bytes");
        }
        previous = Some(images.clone());
        let out = std::env::var_os("QIR_LAYERED_OUTPUT").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("layered-benchmark"));
        std::fs::create_dir_all(&out)?;
        for (i, image) in images.iter().enumerate() {
            image.save(out.join(format!("raw-{}.png", i + 1)))?;
            let mut image = layered_resize::resize(image, export_width, export_height);
            // Document export clears color at zero alpha; match the saved ComfyUI files.
            for pixel in image.pixels_mut() {
                if pixel[3] == 0 {
                    pixel.0[..3].fill(0);
                }
            }
            image.save(out.join(format!("native-{}.png", i + 1)))?;
        }
    }
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

/// Compare the actual Layered QKV GEMM shapes before changing weight residency.
#[test]
#[ignore = "exclusive RTX 5090 microbenchmark"]
fn layered_qkv_gemm_benchmark() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let w = Tensor::randn(0f32, 0.1, (9216, 3072), &device)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 0.1, (16000, 3072), &device)?.to_dtype(DType::BF16)?;
    for merged in [false, true, false, true] {
        device.synchronize()?;
        let t = Instant::now();
        for _ in 0..10 {
            if merged {
                let _ = x.matmul(&w.t()?)?;
            } else {
                for i in 0..3 {
                    let _ = x.matmul(&w.narrow(0, i * 3072, 3072)?.t()?)?;
                }
            }
        }
        device.synchronize()?;
        eprintln!("QKV merged={merged}: {:.4}s per call", t.elapsed().as_secs_f64() / 10.0);
    }
    #[cfg(feature = "cuda")]
    {
        use candle_core::{
            cuda_backend::cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig},
            Storage,
        };
        let Device::Cuda(d) = &device else { unreachable!() };
        let lt = CudaBlasLT::new(d.cuda_stream())?;
        let (xs, _) = x.storage_and_layout();
        let Storage::Cuda(xs) = &*xs else { unreachable!() };
        let xs = xs.as_cuda_slice::<half::bf16>()?;
        let (ws, _) = w.storage_and_layout();
        let Storage::Cuda(ws) = &*ws else { unreachable!() };
        let ws = ws.as_cuda_slice::<half::bf16>()?;
        let mut out = unsafe { d.alloc::<half::bf16>(16000 * 3072)? };
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: 3072,
            n: 16000,
            k: 3072,
            alpha: 1.,
            lda: 3072,
            ldb: 3072,
            beta: 0.,
            ldc: 3072,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        for _ in 0..2 {
            device.synchronize()?;
            let t = Instant::now();
            for _ in 0..10 {
                for i in 0..3 {
                    let wi = ws.slice(i * 3072 * 3072..(i + 1) * 3072 * 3072);
                    unsafe {
                        lt.matmul(cfg, &wi, &xs.slice(..), &mut out, None, None)?;
                    }
                }
            }
            device.synchronize()?;
            eprintln!("QKV cuBLASLt: {:.4}s per call", t.elapsed().as_secs_f64() / 10.);
        }
    }
    Ok(())
}
