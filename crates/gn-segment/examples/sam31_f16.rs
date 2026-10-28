//! Measurement only: shared decoded pixels, ComfyUI reference, explicit CUDA precision.
use anyhow::{Context, Result, ensure};
use gn_segment::sam::{Instance, Options, Sam31};
use std::{path::Path, time::Instant};

fn save(dir: &Path, name: &str, masks: &[Instance]) -> Result<serde_json::Value> {
    let mut entries = Vec::new();
    for (i, m) in masks.iter().enumerate() {
        std::fs::write(dir.join(format!("{name}-{i}.bin")), m.mask.iter().map(|x| u8::from(*x > 0.5)).collect::<Vec<_>>())?;
        entries.push(serde_json::json!({"score":m.score,"pixels":m.pixels,"bbox":m.bbox,"bounds":m.bounds}));
    }
    Ok(serde_json::json!(entries))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("decode");
    let root = Path::new("target/sam31-f16");
    std::fs::create_dir_all(root)?;
    if mode == "decode" {
        let mut pictures = Vec::new();
        for (name, prompt) in [
            ("dog", "dog"),
            ("bench", "bench"),
            ("cup", "coffee cup"),
            ("bicycle", "bicycle"),
            ("car", "car"),
            ("cat", "cat"),
            ("hat", "top hat"),
            ("umbrella", "umbrella"),
            ("bird", "bird"),
            ("boat", "boat"),
            ("glasses", "glasses"),
            ("watch", "watch"),
        ] {
            let file = format!("st-{name}");
            let im = image::open(format!(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\{file}.jpg"))?.to_rgba8();
            std::fs::write(root.join(format!("{file}.rgba")), im.as_raw())?;
            pictures.push(serde_json::json!({"picture":file,"prompt":prompt,"width":im.width(),"height":im.height()}));
        }
        std::fs::write(root.join("pictures.json"), serde_json::to_vec_pretty(&pictures)?)?;
        return Ok(());
    }
    ensure!(["cpu", "cuda-f32", "cuda-tf32", "cuda-f16"].contains(&mode), "unknown mode");
    let cuda = mode != "cpu";
    candle_core::cuda_backend::set_gemm_reduced_precision_f32(mode == "cuda-tf32");
    let device = if cuda { candle_core::Device::new_cuda(0)? } else { candle_core::Device::Cpu };
    let t = Instant::now();
    let mut sam = Sam31::load_with_dtype(
        r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors",
        &device,
        if mode == "cuda-f16" { candle_core::DType::F16 } else { candle_core::DType::F32 },
    )?;
    sam.set_tokenizer(r"C:\Users\5090\Documents\_eval\native-models\sam3.1\tokenizer.json")?;
    device.synchronize()?;
    let load_seconds = t.elapsed().as_secs_f64();
    let pictures: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(root.join("pictures.json"))?)?;
    let out = root.join(mode);
    std::fs::create_dir_all(&out)?;
    for p in pictures {
        let name = p["picture"].as_str().context("missing or invalid measurement metadata")?;
        if std::env::var("GN_F16_PICTURE").is_ok_and(|s| s != name) {
            continue;
        }
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("comfy").join(format!("{name}.json")))?)?;
        let point = if reference["point"].is_null() {
            Vec::new()
        } else {
            vec![[
                reference["point"][0].as_f64().context("missing or invalid measurement metadata")? as f32,
                reference["point"][1].as_f64().context("missing or invalid measurement metadata")? as f32,
            ]]
        };
        let pixels = std::fs::read(root.join(format!("{name}.rgba")))?;
        let mut samples = Vec::new();
        let mut text_meta = serde_json::Value::Null;
        let mut point_meta = serde_json::Value::Null;
        let warmups = usize::from(cuda);
        let reps = if cuda { 5 } else { 1 };
        for rep in 0..warmups + reps {
            device.synchronize()?;
            let t = Instant::now();
            let frame = sam.encode(
                &pixels,
                p["width"].as_u64().context("missing or invalid measurement metadata")? as usize,
                p["height"].as_u64().context("missing or invalid measurement metadata")? as usize,
            )?;
            device.synchronize()?;
            let encode = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let text = sam.select_text(&frame, p["prompt"].as_str().context("missing or invalid measurement metadata")?, Options::default())?;
            device.synchronize()?;
            let select = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let pt = if point.is_empty() { Vec::new() } else { vec![sam.segment_points(&frame, &point, &[], Options::default())?.0] };
            device.synchronize()?;
            let point_seconds = t.elapsed().as_secs_f64();
            println!("{mode} {name} rep={rep} encode={encode:.6} select={select:.6} point={point_seconds:.6}");
            if rep >= warmups {
                samples.push([encode, select, point_seconds]);
                text_meta = save(&out, &format!("{name}-text"), &text)?;
                point_meta = save(&out, &format!("{name}-point"), &pt)?;
            }
        }
        std::fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"mode":mode,"load_seconds":load_seconds,"warmups":warmups,"samples_seconds":samples,"text":text_meta,"point":point_meta,"tf32":candle_core::cuda_backend::gemm_reduced_precision_f32()}),
            )?,
        )?;
    }
    Ok(())
}
