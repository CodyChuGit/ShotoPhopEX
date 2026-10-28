//! Complete-tensor ComfyUI CPU regression oracles from tools/krea2_style_fixtures.py.
use candle_core::{DType, Device, Tensor};
use qwen_image_rs::lora::{LoraFile, LoraStack};
use qwen_image_rs::model::krea2::{self, Krea2Config, Krea2Dit};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn compare(name: &str, actual: &Tensor, expected: &Tensor, limit: f64) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{name}");
    let a = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0.0f64;
    let mut error = 0.0;
    let mut norm = 0.0;
    for (a, b) in a.iter().zip(&b) {
        let e = f64::from(*a) - f64::from(*b);
        max = max.max(e.abs());
        error += e * e;
        norm += f64::from(*b).powi(2);
    }
    let relative = (error / norm.max(1e-24)).sqrt();
    eprintln!("{name}: max abs {max:.9e}, relative RMS {relative:.9e}");
    assert!(relative <= limit, "{name}: {relative} > {limit}");
    Ok(())
}

#[test]
fn stock_style_with_two_references_and_lora() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let tensors = candle_core::safetensors::load(fixtures().join("krea2_style.safetensors"), &dev)?;
    let vb = candle_nn::VarBuilder::from_tensors(candle_core::safetensors::load(fixtures().join("krea2_style_weights.safetensors"), &dev)?, DType::F32, &dev);
    let cfg = Krea2Config {
        features: 64,
        timestep_dim: 32,
        text_dim: 32,
        heads: 2,
        kv_heads: 1,
        layers: 2,
        text_layers: 3,
        text_heads: 1,
        text_kv_heads: 1,
        ..Krea2Config::default()
    };
    let model = Krea2Dit::load(&cfg, false, vb)?;
    compare("image_without_adapter", &model.forward(&tensors["x"], 0.625, &tensors["context"])?, &tensors["image"], 1e-4)?;
    let edit = krea2::EditRef { latent: tensors["ref0"].clone(), boost: 4.0 };
    compare("legacy_edit", &model.forward_edit(&tensors["x"], 0.625, &tensors["context"], Some(&edit))?, &tensors["edit"], 1e-4)?;
    let lora = LoraFile::load(&fixtures().join("krea2_style_lora.safetensors"), &dev, DType::F32)?;
    let stack = LoraStack { layers: vec![(Arc::new(lora), 1.0)] };
    assert!(model.set_loras(&stack)?.is_empty());
    let refs = vec![tensors["ref0"].clone(), tensors["ref1"].clone()];
    for n in 1..=2 {
        compare(
            &format!("style_{n}_adapter"),
            &model.forward_style(&tensors["x"], 0.625, &tensors["context"], &refs[..n])?,
            &tensors[&format!("style_{n}")],
            1e-3,
        )?;
    }
    let schedule =
        qwen_image_rs::comfy::FlowSchedule::simple(2, qwen_image_rs::comfy::Shift::Flux { shift: 0.5, timesteps: 10_000 }).map_err(anyhow::Error::msg)?;
    compare("sigmas", &Tensor::from_slice(&schedule.sigmas, 3, &dev)?, &tensors["sigmas"], 1e-7)?;
    let noise = qwen_image_rs::comfy::torch_randn(123, 16 * 32 * 32).map_err(anyhow::Error::msg)?;
    let mut x = Tensor::from_vec(noise, (1, 16, 32, 32), &dev)?;
    compare("seed_123_noise", &x, &tensors["noise"], 2e-6)?;
    for i in 0..2 {
        let v = model.forward_style(&x, schedule.sigma(i), &tensors["context"], &refs)?;
        x = (x + (v * f64::from(schedule.dt(i)))?)?;
    }
    compare("two_step_latent_adapter", &x, &tensors["latent_2"], 2e-3)?;
    model.set_loras(&LoraStack::default())?;
    compare("image_after_clearing_adapter", &model.forward(&tensors["x"], 0.625, &tensors["context"])?, &tensors["image"], 1e-4)?;
    Ok(())
}

#[test]
fn editplus_chat_override_is_exact() -> anyhow::Result<()> {
    assert_eq!(krea2::style_template("a ceramic bird", 2), serde_json::from_slice::<String>(&std::fs::read(fixtures().join("krea2_style_template.json"))?)?);
    Ok(())
}

#[test]
#[ignore = "regenerate large complete preprocessing tensors with tools/krea2_style_fixtures.py"]
fn stock_editplus_preprocessing() -> anyhow::Result<()> {
    use qwen_image_rs::model::vision;
    use qwen_image_rs::pipeline::RefImage;
    let dev = Device::Cpu;
    let oracle = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated/krea2_style_preprocess.safetensors"), &dev)?;
    for (i, (width, height)) in [(256, 256), (320, 240)].into_iter().enumerate() {
        let rgba = (0..width * height * 4).map(|i| ((i * 37 + i / 7) % 256) as u8).collect();
        let picture = RefImage { width, height, rgba, offset: None };
        let (rgb, vae) = qwen_image_rs::krea2_pipeline::style_inputs(&picture, &dev)?;
        let (patches, gh, gw) = vision::patchify(&rgb, &vision::VisionConfig::qwen3vl_4b())?;
        compare(&format!("vision_patches_{i}"), &patches, &oracle[&format!("patches_{i}")], 1e-5)?;
        let grid = oracle[&format!("grid_{i}")].flatten_all()?.to_vec1::<i64>()?;
        assert_eq!(grid, [1, gh as i64, gw as i64]);
        compare(&format!("vae_pixels_{i}"), &vae, &oracle[&format!("vae_input_{i}")], 1e-5)?;
    }
    Ok(())
}

/// CPU checkpoint reader dequantizes one requested matrix at a time.
struct CpuCheckpoint {
    path: PathBuf,
    header: serde_json::Value,
    start: u64,
}
impl CpuCheckpoint {
    fn open(path: PathBuf) -> anyhow::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        let mut header = vec![0; usize::try_from(length)?];
        file.read_exact(&mut header)?;
        Ok(Self { path, header: serde_json::from_slice(&header)?, start: 8 + length })
    }
    fn tensor(&self, name: &str, device: &Device) -> candle_core::Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let info = &self.header[name];
        let shape: Vec<usize> = serde_json::from_value(info["shape"].clone()).map_err(candle_core::Error::wrap)?;
        let offsets: [u64; 2] = serde_json::from_value(info["data_offsets"].clone()).map_err(candle_core::Error::wrap)?;
        let dtype = match info["dtype"].as_str() {
            Some("BF16") => DType::BF16,
            Some("F32") => DType::F32,
            Some("F8_E4M3") => DType::F8E4M3,
            _ => candle_core::bail!("unsupported CPU checkpoint dtype for {name}"),
        };
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.start + offsets[0]))?;
        let mut data = vec![0; usize::try_from(offsets[1] - offsets[0]).map_err(candle_core::Error::wrap)?];
        file.read_exact(&mut data)?;
        Tensor::from_raw_buffer(&data, dtype, &shape, device)
    }
}
impl candle_nn::var_builder::SimpleBackend for CpuCheckpoint {
    fn get(&self, shape: candle_core::Shape, name: &str, _: candle_nn::Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("incorrect checkpoint shape for {name}");
        }
        Ok(tensor)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let value = self.tensor(name, device)?;
        if value.dtype() == DType::F8E4M3 {
            let scale = self.tensor(&name.replace(".weight", ".weight_scale"), device)?.to_dtype(DType::BF16)?;
            value.to_dtype(DType::BF16)?.broadcast_mul(&scale)?.to_dtype(dtype)
        } else {
            value.to_dtype(dtype)
        }
    }
    fn contains_tensor(&self, name: &str) -> bool {
        // The CPU oracle explicitly dequantizes; bypass CUDA FP8 dispatch in the encoders.
        !name.ends_with("weight_scale") && self.header.get(name).is_some()
    }
}

#[test]
#[ignore = "installed 4B encoder, ~20 GiB RAM; regenerate tools/krea2_style_encoder_fixtures.py first"]
fn installed_style_encoder() -> anyhow::Result<()> {
    use qwen_image_rs::model::{text_encoder, vision};
    use qwen_image_rs::pipeline::RefImage;
    let dev = Device::Cpu;
    let root = PathBuf::from(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models");
    let path = root.join("text_encoders/qwen3vl_4b_fp8_scaled.safetensors");
    let vb = candle_nn::VarBuilder::from_backend(Box::new(CpuCheckpoint::open(path)?), DType::F32, dev.clone());
    let oracle = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated/krea2_style_encoder.safetensors"), &dev)?;
    let preprocess_oracle =
        candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated/krea2_style_preprocess.safetensors"), &dev)?;
    let mut identical_patch_features = Vec::new();
    let mut features = Vec::new();
    {
        let tower = vision::VisionTower::load(&vision::VisionConfig::qwen3vl_4b(), vb.pp("model.visual"))?;
        for (i, (width, height)) in [(256, 256), (320, 240)].into_iter().enumerate() {
            let rgba = (0..width * height * 4).map(|i| ((i * 37 + i / 7) % 256) as u8).collect();
            let (rgb, _) = qwen_image_rs::krea2_pipeline::style_inputs(&RefImage { width, height, rgba, offset: None }, &dev)?;
            let (patches, gh, gw) = vision::patchify(&rgb, tower.config())?;
            let f = tower.forward_editplus(&patches, gh, gw)?;
            compare(&format!("installed_vision_{i}"), &f.merged, &oracle[&format!("vision_{i}")], 5e-5)?;
            for (j, deep) in f.deepstack.iter().enumerate() {
                compare(&format!("installed_deepstack_{i}_{j}"), deep, &oracle[&format!("deepstack_{i}_{j}")], 5e-5)?;
            }
            identical_patch_features.push(tower.forward_editplus(&preprocess_oracle[&format!("patches_{i}")], gh, gw)?);
            features.push(f);
        }
    }
    let te = text_encoder::QwenTextEncoder::load_at(&text_encoder::TextConfig::qwen3vl_4b(), false, vb.pp("model"))?;
    let tokenizer =
        qwen_image_rs::comfy_files::qwen_tokenizer(&qwen_image_rs::comfy_files::tokenizer_dir(&root).ok_or_else(|| anyhow::anyhow!("missing tokenizer"))?)?;
    let ids = tokenizer.encode(krea2::style_template("a ceramic bird", 2), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let conditioning = krea2::condition_with_images(&te, &ids, &features.iter().collect::<Vec<_>>())?.unsqueeze(0)?;
    compare("installed_conditioning", &conditioning, &oracle["conditioning"], 5e-5)?;
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("conditioning".to_string(), conditioning.clone())]),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated/krea2_style_native_encoder.safetensors"),
    )?;
    let oracle_features = (0..2)
        .map(|i| vision::VisionFeatures {
            merged: oracle[&format!("vision_{i}")].clone(),
            deepstack: (0..3).map(|j| oracle[&format!("deepstack_{i}_{j}")].clone()).collect(),
            rows: features[i].rows,
            cols: features[i].cols,
        })
        .collect::<Vec<_>>();
    let isolated = krea2::condition_with_images(&te, &ids, &oracle_features.iter().collect::<Vec<_>>())?.unsqueeze(0)?;
    compare("installed_text_identical_vision", &isolated, &oracle["conditioning"], 5e-5)?;
    let isolated_max = (&isolated - &oracle["conditioning"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(isolated_max <= 0.02, "isolated text conditioning max abs {isolated_max}");
    let identical = krea2::condition_with_images(&te, &ids, &identical_patch_features.iter().collect::<Vec<_>>())?.unsqueeze(0)?;
    compare("installed_conditioning_identical_patches", &identical, &oracle["conditioning"], 5e-5)?;
    let max = (&conditioning - &oracle["conditioning"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max <= 0.02, "conditioning max abs {max}");
    Ok(())
}

#[test]
#[ignore = "installed RGB Wan VAE at EditPlus megapixel size; regenerate tools/krea2_style_model_fixtures.py first"]
fn installed_style_reference_latents() -> anyhow::Result<()> {
    use qwen_image_rs::{model::wan_vae::WanVae, pipeline::RefImage};
    let dev = Device::Cpu;
    let path = PathBuf::from(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/vae/qwen_image_vae.safetensors");
    // SAFETY: the installed VAE checkpoint is immutable during this test.
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &dev)? };
    let vae = WanVae::load(vb)?;
    let generated = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated");
    let oracle = candle_core::safetensors::load(generated.join("krea2_style_references.safetensors"), &dev)?;
    let mut actual = std::collections::HashMap::new();
    for (i, (width, height)) in [(256, 256), (320, 240)].into_iter().enumerate() {
        let rgba = (0..width * height * 4).map(|i| ((i * 37 + i / 7) % 256) as u8).collect();
        let (_, pixels) = qwen_image_rs::krea2_pipeline::style_inputs(&RefImage { width, height, rgba, offset: None }, &dev)?;
        let latent = vae.encode(&pixels)?;
        let name = format!("ref{i}");
        compare(&format!("installed_{name}_latent"), &latent, &oracle[&name], 1e-5)?;
        actual.insert(name, latent);
    }
    candle_core::safetensors::save(&actual, generated.join("krea2_style_native_references.safetensors"))?;
    Ok(())
}

#[test]
#[ignore = "streams 28 installed Krea blocks three times on CPU; run encoder/reference gates first, one heavy test at a time"]
fn installed_style_dit_and_two_steps() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let dev = Device::Cpu;
    let root = PathBuf::from(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models");
    let vb = candle_nn::VarBuilder::from_backend(
        Box::new(CpuCheckpoint::open(root.join("diffusion_models/krea2_turbo_fp8_scaled.safetensors"))?),
        DType::F32,
        dev.clone(),
    );
    let model = Krea2Dit::load(&Krea2Config { layers: 0, ..Krea2Config::default() }, false, vb.clone())?;
    let lora = LoraFile::load(&root.join("loras/krea2_style_reference.safetensors"), &dev, DType::F32)?;
    assert_eq!(lora.modules().count(), 256);
    let stack = LoraStack { layers: vec![(Arc::new(lora), 1.0)] };
    // The resident shell attaches fusion adapters; streamed blocks attach their own adapters.
    assert_eq!(model.set_loras(&stack)?.len(), 224);
    let generated = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/generated");
    let oracle = candle_core::safetensors::load(generated.join("krea2_style_model.safetensors"), &dev)?;
    let refs = vec![oracle["ref0"].clone(), oracle["ref1"].clone()];
    let forward = model.forward_style_streamed(&oracle["noise"], 0.625, &oracle["context"], &refs, 28, &vb, &stack)?;
    compare("installed_dit_lora_two_refs", &forward, &oracle["forward"], 1e-3)?;
    let actual_refs = candle_core::safetensors::load(generated.join("krea2_style_native_references.safetensors"), &dev)?;
    let refs = vec![actual_refs["ref0"].clone(), actual_refs["ref1"].clone()];
    let actual_cond = candle_core::safetensors::load(generated.join("krea2_style_native_encoder.safetensors"), &dev)?;
    let noise = qwen_image_rs::comfy::torch_randn(123, 16 * 32 * 32).map_err(anyhow::Error::msg)?;
    let mut x = Tensor::from_vec(noise, (1, 16, 32, 32), &dev)?;
    let schedule =
        qwen_image_rs::comfy::FlowSchedule::simple(2, qwen_image_rs::comfy::Shift::Flux { shift: 0.5, timesteps: 10_000 }).map_err(anyhow::Error::msg)?;
    for i in 0..2 {
        let v = model.forward_style_streamed(&x, schedule.sigma(i), &actual_cond["conditioning"], &refs, 28, &vb, &stack)?;
        x = (x + (v * f64::from(schedule.dt(i)))?)?;
        compare(&format!("installed_latent_{}", i + 1), &x, &oracle[&format!("latent_{}", i + 1)], 2e-3)?;
    }
    Ok(())
}
