//! Diagnose ONNX/Candle numerical boundaries separately from benchmark timings.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use gn_segment::{
    detector::Detector,
    geometry::Geometry,
    text::{ClipTokenizer, TextEncoder},
    vision::Vision,
    weights::Checkpoint,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
#[allow(dead_code)]
#[path = "../src/resample.rs"]
mod resample;
const MODEL: &str = r"C:\Users\5090\Documents\_eval\native-models\sam3.1";
const CHECKPOINT: &str = r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors";
fn cpu(t: &Tensor) -> Result<Tensor> {
    Ok(t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)
}
fn save(path: &Path, values: &HashMap<String, Tensor>) -> Result<()> {
    Ok(candle_core::safetensors::save(values, path)?)
}
fn ort_tensor(t: &Tensor) -> Result<ort::value::Tensor<f32>> {
    Ok(ort::value::Tensor::from_array((t.dims().to_vec(), cpu(t)?.flatten_all()?.to_vec1::<f32>()?))?)
}
fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "candle".into());
    let root = PathBuf::from("crates/gn-segment/benchmark-output/stages");
    std::fs::create_dir_all(&root)?;
    let (ids, valid) = ClipTokenizer::open(Path::new(MODEL).join("tokenizer.json"))?.encode("coffee cup")?;
    let mut values = HashMap::new();
    if mode == "onnx" {
        use gn_perception::runtime::{Device as OrtDevice, Runtime, RuntimeConfig};
        let rt = Runtime::open(RuntimeConfig {
            ort_dir: PathBuf::from(r"C:\Users\5090\Documents\_eval\onnxruntime-cuda13\onnxruntime-win-x64-gpu_cuda13-1.31.0\lib"),
            library_dirs: vec![
                PathBuf::from(r"C:\Users\5090\scoop\apps\cuda\current\bin"),
                PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\lib"),
            ],
            device: OrtDevice::Cuda { ordinal: 0 },
            profile_dir: None,
        })?;
        let image = image::open(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-cup.jpg")?.to_rgb8();
        let mut planes = Vec::new();
        for ch in 0..3 {
            let p: Vec<f32> = image.pixels().map(|p| f32::from(p[ch]) / 255.0).collect();
            planes.extend(resample::bilinear(&p, image.width() as usize, image.height() as usize, 1008, 1008));
        }
        let input = Tensor::from_vec(planes, (1, 3, 1008, 1008), &Device::Cpu)?;
        values.insert("input".into(), input.clone());
        let mut vision = rt.session(&Path::new(MODEL).join("vision.onnx"))?;
        let out = vision.run(ort::inputs!["images" => ort_tensor(&input)?])?;
        for (name, value) in out.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            values.insert(name.into(), Tensor::from_vec(data.to_vec(), shape.iter().map(|v| *v as usize).collect::<Vec<_>>(), &Device::Cpu)?);
        }
        drop(out);
        drop(vision);
        let mut text = rt.session(&Path::new(MODEL).join("text.onnx"))?;
        let ids = ort::value::Tensor::from_array(([1, 32], ids.map(i64::from).to_vec()))?;
        let am = ort::value::Tensor::from_array(([1, 32], valid.map(i64::from).to_vec()))?;
        let out = text.run(ort::inputs!["input_ids" => ids, "attention_mask" => am])?;
        let (_, features) = out["text_features"].try_extract_tensor::<f32>()?;
        let text_features = Tensor::from_slice(features, (1, 32, 256), &Device::Cpu)?;
        values.insert("prompt".into(), Tensor::cat(&[text_features, values["geo_cls"].clone()], 1)?);
        save(&root.join("inputs.safetensors"), &values)?;
        drop(out);
        drop(text);
        let mut decoder = rt.session(&std::env::var("GN_SAM31_DECODER").map(PathBuf::from).unwrap_or_else(|_| Path::new(MODEL).join("decoder.onnx")))?;
        let mut mask = valid.to_vec();
        mask.push(true);
        let out = decoder.run(ort::inputs![
            "fpn_feat_0" => ort_tensor(&values["fpn_feat_0"] )?, "fpn_feat_1" => ort_tensor(&values["fpn_feat_1"] )?,
            "fpn_feat_2" => ort_tensor(&values["fpn_feat_2"] )?, "fpn_pos_2" => ort_tensor(&values["fpn_pos_2"] )?,
            "prompt_features" => ort_tensor(&values["prompt"] )?, "prompt_mask" => ort::value::Tensor::from_array(([1,33], mask))?
        ])?;
        for (name, value) in out.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            values.insert(name.into(), Tensor::from_vec(data.to_vec(), shape.iter().map(|v| *v as usize).collect::<Vec<_>>(), &Device::Cpu)?);
        }
    } else {
        let device = Device::new_cuda(0)?;
        let ck = Checkpoint::open(CHECKPOINT)?.with_compute_dtype(DType::F16)?;
        if mode == "cross" {
            values = candle_core::safetensors::load(root.join("inputs.safetensors"), &device)?;
        } else {
            let input = candle_core::safetensors::load(root.join("inputs.safetensors"), &device)?.remove("input").context("input missing")?;
            let vision = Vision::from_checkpoint(&ck, &device)?.forward(&input)?;
            let geo = Geometry::from_checkpoint(&ck, &device)?.cls(&vision.detector_fpn[2], &vision.image_pos)?;
            let (features, _) = TextEncoder::from_checkpoint(&ck, &device)?.encode(&ids, &valid)?;
            values.insert("prompt".into(), Tensor::cat(&[features.to_dtype(geo.dtype())?, geo], 1)?);
            for (i, x) in vision.detector_fpn.into_iter().enumerate() {
                values.insert(format!("fpn_feat_{i}"), x);
            }
            values.insert("fpn_pos_2".into(), vision.image_pos);
        }
        let mut mask = valid.map(u8::from).to_vec();
        mask.push(1);
        let mut observations = HashMap::new();
        let out = Detector::from_checkpoint(&ck, &device)?.forward_with_observer(
            [&values["fpn_feat_0"], &values["fpn_feat_1"], &values["fpn_feat_2"]],
            &values["fpn_pos_2"],
            &values["prompt"],
            &Tensor::from_vec(mask, (1, 33), &device)?,
            |name, t| {
                observations.insert(name.into(), cpu(t)?);
                Ok(())
            },
        )?;
        values.extend(observations);
        values.insert("pred_masks".into(), out.masks);
        values.insert("pred_logits".into(), out.logits);
        values.insert("pred_boxes".into(), out.boxes);
        values = values.iter().map(|(k, v)| Ok((k.clone(), cpu(v)?))).collect::<Result<_>>()?;
    }
    let logits = values["pred_logits"].flatten_all()?.to_vec1::<f32>()?;
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).context("empty logits")?.0;
    println!("{mode} best={best} logit={}", logits[best]);
    values.insert("best_mask".into(), values["pred_masks"].narrow(1, best, 1)?);
    values.remove("pred_masks");
    save(&root.join(format!("{mode}.safetensors")), &values)?;
    if mode != "onnx" {
        let reference = candle_core::safetensors::load(root.join("onnx.safetensors"), &Device::Cpu)?;
        for (name, t) in &values {
            if let Some(other) = reference.get(name) {
                let a = t.flatten_all()?.to_vec1::<f32>()?;
                let b = other.flatten_all()?.to_vec1::<f32>()?;
                let abs = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
                let rms = (a.iter().zip(&b).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
                println!("{name}: max_abs={abs} rms={rms}");
                if name == "best_mask" {
                    let intersection = a.iter().zip(&b).filter(|(a, b)| **a > 0.0 && **b > 0.0).count();
                    let union = a.iter().zip(&b).filter(|(a, b)| **a > 0.0 || **b > 0.0).count();
                    println!("low_mask_iou={}", intersection as f64 / union as f64);
                }
            }
        }
    }
    Ok(())
}
