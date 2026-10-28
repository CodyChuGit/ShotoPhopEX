//! Warm CPU/CUDA end-to-end parity and synchronized stage timings, with separate backend processes.
use anyhow::{Result, ensure};
use gn_perception::{
    runtime::{Device, Runtime, RuntimeConfig},
    sam31,
};
use gn_segment::sam::{Instance, Options, Sam31};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn save(dir: &Path, name: &str, masks: &[Instance], times: [f64; 3]) -> Result<()> {
    let entries: Vec<_> = masks
        .iter()
        .enumerate()
        .map(|(i, m)| {
            std::fs::write(dir.join(format!("{name}-{i}.bin")), m.mask.iter().map(|v| u8::from(*v > 0.5)).collect::<Vec<_>>())?;
            Ok(serde_json::json!({"score":m.score,"bbox":m.bbox,"bounds":m.bounds,"pixels":m.pixels}))
        })
        .collect::<Result<_>>()?;
    std::fs::write(dir.join(format!("{name}.json")), serde_json::to_vec_pretty(&serde_json::json!({"instances":entries,"seconds":times}))?)?;
    Ok(())
}
fn compare(reference: &Path, name: &str, masks: &[Instance]) -> Result<bool> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(reference.join(format!("{name}.json")))?)?;
    let entries = meta["instances"].as_array().ok_or_else(|| anyhow::anyhow!("missing instances"))?;
    println!("{name}: count candle={} onnx={}", masks.len(), entries.len());
    let mut pass = masks.len() == entries.len();
    for (i, (m, e)) in masks.iter().zip(entries).enumerate() {
        let other = std::fs::read(reference.join(format!("{name}-{i}.bin")))?;
        ensure!(other.len() == m.mask.len(), "mask sizes differ");
        let (mut intersection, mut union) = (0usize, 0usize);
        for (a, b) in m.mask.iter().zip(other) {
            let a = *a > 0.5;
            let b = b != 0;
            intersection += usize::from(a && b);
            union += usize::from(a || b);
        }
        let iou = if union == 0 { 1.0 } else { intersection as f64 / union as f64 };
        let difference = f64::from(m.score) - e["score"].as_f64().ok_or_else(|| anyhow::anyhow!("missing score"))?;
        println!("{name} instance={i} IoU={iou:.9} score_difference={difference:.9} bounds={:?} onnx_bounds={}", m.bounds, e["bounds"]);
        pass &= iou >= 0.99;
    }
    Ok(pass)
}
fn stages(prefix: &str, values: &mut std::collections::BTreeMap<String, f64>) {
    for (name, ms) in gn_segment::profile::finish() {
        values.insert(format!("{prefix}.{name}"), ms);
    }
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let backend = args.get(1).map(String::as_str).unwrap_or("candle");
    ensure!(["candle", "cuda", "cuda-f16", "onnx", "onnx-cuda"].contains(&backend), "backend must be candle, cuda, cuda-f16, onnx or onnx-cuda");
    let cuda = backend == "cuda" || backend == "cuda-f16";
    let onnx_cuda = backend == "onnx-cuda";
    let warm = usize::from(cuda || onnx_cuda);
    let profiling = cuda && std::env::var("GN_SAM31_NO_PROFILE").is_err();
    let device = if cuda { candle_core::Device::new_cuda(0)? } else { candle_core::Device::Cpu };
    let repetitions: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(if warm > 0 { 5 } else { 1 });
    ensure!(repetitions > 0, "repetitions must be positive");
    let model_dir = PathBuf::from(r"C:\Users\5090\Documents\_eval\native-models\sam3.1");
    let stock = PathBuf::from(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock");
    let root = PathBuf::from(std::env::var("GN_SAM31_OUTPUT_DIR").unwrap_or_else(|_| "crates/gn-segment/benchmark-output".into()));
    let reference = root.join(std::env::var("GN_SAM31_REFERENCE").unwrap_or_else(|_| if cuda { "onnx-cuda".into() } else { "onnx".into() }));
    let output = root.join(backend);
    let iterations = std::env::var("GN_SAM31_ITERATIONS").ok().map(|s| s.parse()).transpose()?.unwrap_or(2);
    let opts = Options { refine_iterations: iterations, ..Options::default() };
    let onnx_opts = sam31::Options { refine_iterations: iterations, ..sam31::Options::default() };
    std::fs::create_dir_all(&output)?;
    println!(
        "backend={backend} warmups={warm} repetitions={repetitions} refinement={iterations} synchronized_stages={profiling} reference={}",
        reference.display()
    );
    let mut candle = if backend == "candle" || cuda {
        let mut s = Sam31::load_with_dtype(
            r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors",
            &device,
            if backend == "cuda-f16" { candle_core::DType::F16 } else { candle_core::DType::F32 },
        )?;
        s.set_tokenizer(model_dir.join("tokenizer.json"))?;
        Some(s)
    } else {
        None
    };
    let mut onnx = if backend == "onnx" || onnx_cuda {
        let rt = Runtime::open(RuntimeConfig {
            ort_dir: PathBuf::from(r"C:\Users\5090\Documents\_eval\onnxruntime-cuda13\onnxruntime-win-x64-gpu_cuda13-1.31.0\lib"),
            library_dirs: vec![
                PathBuf::from(r"C:\Users\5090\scoop\apps\cuda\current\bin"),
                PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\lib"),
            ],
            device: if onnx_cuda { Device::Cuda { ordinal: 0 } } else { Device::Cpu },
            profile_dir: None,
        })?;
        Some(sam31::Sam31::open(&rt, &model_dir)?)
    } else {
        None
    };
    let mut passed = true;
    let mut processed = 0;
    for (file, prompt, point) in [("st-dog", "dog", [[930.0, 2140.0]]), ("st-bench", "bench", [[1100.0, 690.0]]), ("st-cup", "coffee cup", [[590.0, 870.0]])] {
        if std::env::var("GN_SAM31_PICTURE").is_ok_and(|s| s != file) {
            continue;
        }
        processed += 1;
        let image = image::open(stock.join(format!("{file}.jpg")))?.to_rgba8();
        let (w, h) = (image.width() as usize, image.height() as usize);
        let mut samples = Vec::new();
        let mut stage_samples: std::collections::BTreeMap<String, Vec<f64>> = std::collections::BTreeMap::new();
        for repetition in 0..repetitions + warm {
            let mut stage_times = std::collections::BTreeMap::new();
            if profiling {
                gn_segment::profile::start();
            }
            device.synchronize()?;
            let t = Instant::now();
            let (text, interactive, times) = if let Some(s) = candle.as_mut() {
                let frame = s.encode(image.as_raw(), w, h)?;
                device.synchronize()?;
                let encode = t.elapsed().as_secs_f64();
                if profiling {
                    stages("encode", &mut stage_times);
                    gn_segment::profile::start();
                }
                let t = Instant::now();
                let text = s.select_text(&frame, prompt, opts)?;
                device.synchronize()?;
                let select = t.elapsed().as_secs_f64();
                if profiling {
                    stages("select", &mut stage_times);
                    gn_segment::profile::start();
                }
                let t = Instant::now();
                let point = s.segment_points(&frame, &point, &[], opts)?.0;
                device.synchronize()?;
                let interactive = t.elapsed().as_secs_f64();
                if profiling {
                    stages("point", &mut stage_times);
                }
                (text, vec![point], [encode, select, interactive])
            } else {
                let s = onnx.as_mut().ok_or_else(|| anyhow::anyhow!("no backend"))?;
                let frame = s.embed(image.as_raw(), w, h)?;
                let encode = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let text = s.detect_text(&frame, prompt, &[], onnx_opts)?.0;
                let select = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let point = s.segment_points(&frame, &point, &[], onnx_opts)?.0;
                let interactive = t.elapsed().as_secs_f64();
                let mirror = |m: sam31::Instance| Instance { score: m.score, bbox: m.bbox, mask: m.mask, bounds: m.bounds, pixels: m.pixels };
                (text.into_iter().map(mirror).collect(), vec![mirror(point)], [encode, select, interactive])
            };
            println!("{file} {backend} repetition={repetition}: encode/select/interactive seconds={times:?} new_text={:.6}", times[0] + times[1]);
            if profiling {
                println!("stages_ms={stage_times:?}");
            }
            if repetition < warm {
                continue;
            }
            samples.push(times);
            for (name, time) in stage_times {
                stage_samples.entry(name).or_default().push(time);
            }
            save(&output, &format!("{file}-text"), &text, times)?;
            save(&output, &format!("{file}-point"), &interactive, times)?;
            if candle.is_some() {
                passed &= compare(&reference, &format!("{file}-text"), &text)?;
                passed &= compare(&reference, &format!("{file}-point"), &interactive)?;
            }
        }
        let mut new_text: Vec<_> = samples.iter().map(|t| (t[0] + t[1]) * 1000.0).collect();
        let mut point: Vec<_> = samples.iter().map(|t| t[2] * 1000.0).collect();
        let stages: std::collections::BTreeMap<_, _> = stage_samples.into_iter().map(|(k, mut v)| (k, median(&mut v))).collect();
        let summary = serde_json::json!({"backend":backend,"picture":file,"repetitions":repetitions,"warmups":warm,"refinement":iterations,"synchronized_stages":profiling,"new_text_median_ms":median(&mut new_text),"point_median_ms":median(&mut point),"stages_median_ms":stages,"samples_seconds":samples});
        println!("summary={summary}");
        std::fs::write(output.join(format!("{file}-timings.json")), serde_json::to_vec_pretty(&summary)?)?;
    }
    ensure!(processed > 0, "no matching benchmark picture");
    ensure!(passed, "end-to-end IoU/count/order gate failed");
    Ok(())
}
