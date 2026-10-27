//! CPU end-to-end parity. Run reference and Candle separately to bound peak resident memory.
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
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let backend = args.get(1).map(String::as_str).unwrap_or("candle");
    ensure!(std::env::var("CUDA_VISIBLE_DEVICES").map_or(true, |s| s.is_empty()), "set CUDA_VISIBLE_DEVICES to the empty string");
    let model_dir = PathBuf::from(r"C:\Users\5090\Documents\_eval\native-models\sam3.1");
    let stock = PathBuf::from(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock");
    let root = PathBuf::from("target/sam31-parity");
    let output = root.join(backend);
    std::fs::create_dir_all(&output)?;
    let mut candle = if backend == "candle" {
        let mut s = Sam31::load(
            r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors",
            &candle_core::Device::Cpu,
        )?;
        s.set_tokenizer(model_dir.join("tokenizer.json"))?;
        Some(s)
    } else {
        None
    };
    let mut onnx = if backend == "onnx" {
        let rt = Runtime::open(RuntimeConfig {
            ort_dir: PathBuf::from(r"C:\Users\5090\Documents\_eval\onnxruntime-cuda13\onnxruntime-win-x64-gpu_cuda13-1.31.0\lib"),
            library_dirs: vec![],
            device: Device::Cpu,
            profile_dir: None,
        })?;
        Some(sam31::Sam31::open(&rt, &model_dir)?)
    } else {
        None
    };
    ensure!(candle.is_some() || onnx.is_some(), "backend must be candle or onnx");
    let mut passed = true;
    for (file, prompt, point) in [("st-dog", "dog", [[930.0, 2140.0]]), ("st-bench", "bench", [[1100.0, 690.0]]), ("st-cup", "coffee cup", [[590.0, 870.0]])] {
        let image = image::open(stock.join(format!("{file}.jpg")))?.to_rgba8();
        let (w, h) = (image.width() as usize, image.height() as usize);
        let t = Instant::now();
        let (text, interactive, times) = if let Some(s) = candle.as_mut() {
            let frame = s.encode(image.as_raw(), w, h)?;
            let encode = t.elapsed().as_secs_f64();
            println!("{file} candle encode={encode:.3}s");
            let t = Instant::now();
            let text = s.select_text(&frame, prompt, Options::default())?;
            let select = t.elapsed().as_secs_f64();
            println!("{file} candle select={select:.3}s");
            let t = Instant::now();
            let point = s.segment_points(&frame, &point, &[], Options::default())?.0;
            let interactive = t.elapsed().as_secs_f64();
            (text, vec![point], [encode, select, interactive])
        } else {
            let s = onnx.as_mut().ok_or_else(|| anyhow::anyhow!("no backend"))?;
            let frame = s.embed(image.as_raw(), w, h)?;
            let encode = t.elapsed().as_secs_f64();
            println!("{file} onnx encode={encode:.3}s");
            let t = Instant::now();
            let text = s.detect_text(&frame, prompt, &[], sam31::Options::default())?.0;
            let select = t.elapsed().as_secs_f64();
            println!("{file} onnx select={select:.3}s");
            let t = Instant::now();
            let point = s.segment_points(&frame, &point, &[], sam31::Options::default())?.0;
            let interactive = t.elapsed().as_secs_f64();
            let mirror = |m: sam31::Instance| Instance { score: m.score, bbox: m.bbox, mask: m.mask, bounds: m.bounds, pixels: m.pixels };
            (text.into_iter().map(mirror).collect(), vec![mirror(point)], [encode, select, interactive])
        };
        println!("{file} {backend}: encode/select/interactive seconds={times:?}");
        save(&output, &format!("{file}-text"), &text, times)?;
        save(&output, &format!("{file}-point"), &interactive, times)?;
        if backend == "candle" {
            passed &= compare(&root.join("onnx"), &format!("{file}-text"), &text)?;
            passed &= compare(&root.join("onnx"), &format!("{file}-point"), &interactive)?;
        }
    }
    ensure!(passed, "end-to-end IoU/count/order gate failed");
    Ok(())
}
