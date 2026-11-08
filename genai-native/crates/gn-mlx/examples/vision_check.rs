//! Checks of Qwen3.5-9B's vision tower on MLX:
//! 1. the group-wise ConvRot dequantization at 256 against qir's (`Converter::dense`) on a
//!    language-model linear;
//! 2. the tower on a picture against qir's Qwen3-VL tower on the CPU in f32 (the same weights,
//!    dequantized here), as the cosine of the merged rows.
//! `cargo run --release -p gn-mlx --example vision_check -- <qwen3.5_9b_int8_convrot.safetensors> <picture.png> [side]`
//! (`side`, default 256, the square the picture is resized to first, to keep the CPU run short).

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use candle_core::{DType, Device, Tensor};
    use gn_mlx::convert::{Converter, Precision};
    use gn_mlx::nn::{compare, to_vec};
    use gn_mlx::vision;
    use mlx_rs::Array;
    use qwen_image_rs::moss::weights::Weights;

    let a: Vec<String> = std::env::args().skip(1).collect();
    let [file, picture, rest @ ..] = a.as_slice() else { anyhow::bail!("usage: vision_check <encoder file> <picture.png> [side]") };
    let file = PathBuf::from(file);
    let side: u32 = rest.first().map_or(Ok(256), |s| s.parse())?;

    // 1. Dequantization at 256 against qir's.
    let t = Array::load_safetensors(&file)?;
    let module = "model.language_model.layers.0.mlp.gate_proj";
    let ours = to_vec(&vision::dequantize(&t, module)?)?;
    let w = Weights::open(&file, &Device::Cpu, DType::F32, false)?;
    let c = Converter::new(&w, mlx_rs::Dtype::Float32, Precision::F16)?;
    let theirs = to_vec(&c.dense(module)?)?;
    let (cos, max) = compare(&ours, &theirs)?;
    println!("dequantize {module} (group 256) against qir: cos {cos:.7}, max |d| {max:.3e}");
    drop(c);
    drop(w);

    // 2. The tower against qir's CPU f32 tower with the same weights.
    let cfg = vision::config();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let to_candle = |a: &Array| -> anyhow::Result<Tensor> {
        let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
        let dims: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        Ok(Tensor::from_vec(to_vec(&a)?, dims, &Device::Cpu)?)
    };
    for (k, v) in &t {
        let Some(rest) = k.strip_prefix("model.visual.") else { continue };
        if rest.ends_with("comfy_quant") || rest.ends_with("weight_scale") {
            continue;
        }
        let module = k.trim_end_matches(".weight");
        let dense = if rest.ends_with(".weight") && t.contains_key(&format!("{module}.comfy_quant")) { vision::dequantize(&t, module)? } else { v.clone() };
        tensors.insert(rest.to_string(), to_candle(&dense)?);
    }
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let reference = qwen_image_rs::model::vision::VisionTower::load(&cfg, vb)?;
    drop(t);

    let img = image::open(picture)?.to_rgba8();
    let img = image::imageops::resize(&img, side, side, image::imageops::FilterType::Triangle);
    let (w, h) = (side as usize, side as usize);
    let out = std::env::temp_dir().join("gn-mlx-vision-check.safetensors");
    vision::convert(&file, &out)?;
    let tower = vision::VisionTower::load(&out)?;
    let ours = tower.forward(img.as_raw(), w, h)?;
    let _ = std::fs::remove_file(&out);

    let planes: Vec<f32> = (0..3).flat_map(|c| img.pixels().map(move |p| f32::from(p.0[c]) / 255.0)).collect();
    let rgb = Tensor::from_vec(planes, (3, h, w), &Device::Cpu)?;
    let (vw, vh) = qwen_image_rs::model::vision::vit_size(w, h, 3136, 12_845_056);
    let rgb = if (vw, vh) == (w, h) { rgb } else { qwen_image_rs::model::vision::resize_bilinear(&rgb, vh, vw)? };
    let (patches, gh, gw) = qwen_image_rs::model::vision::patchify(&rgb, &cfg)?;
    let theirs = reference.forward(&patches, gh, gw)?;
    let (cos, max) = compare(&to_vec(&ours.merged)?, &theirs.merged.flatten_all()?.to_vec1::<f32>()?)?;
    println!("tower at {side}² ({} merged rows) against qir f32 on the CPU: cos {cos:.5}, max |d| {max:.3e}", ours.rows * ours.cols);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
