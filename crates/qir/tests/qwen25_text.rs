//! Regenerate fixtures with tools/qir_qwen25_fixtures.py using portable Python -I.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use qwen_image_rs::layered_files::{LayeredFiles, LayeredLinear};
use qwen_image_rs::model::qwen25_text::{load_tokenizer, prefix_cut, prompt_template, tokenize, Qwen25Config, Qwen25TextEncoder};

fn root() -> PathBuf {
    std::env::var_os("QIR_COMFY_MODELS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models"))
}

fn tokenizer_path() -> PathBuf {
    std::env::var_os("QIR_QWEN25_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\text_encoders\qwen25_tokenizer"))
}

fn metadata() -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_str(include_str!("fixtures/qwen25_text.json"))?)
}

#[test]
fn installed_headers() -> anyhow::Result<()> {
    for name in [
        "diffusion_models/qwen_image_layered_fp8mixed.safetensors",
        "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae/qwen_image_layered_vae.safetensors",
    ] {
        if !root().join(name).is_file() {
            eprintln!("Skipping installed headers: {} absent", root().join(name).display());
            return Ok(());
        }
    }
    LayeredFiles::resolve(root())?;
    Ok(())
}

#[test]
fn biased_fp8_scalar_rows_and_bf16() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::from_slice(&[2f32, 3., -1., 4.], (1, 2, 2), &dev)?;
    for (scale_name, scales, shape, expected) in [
        ("weight_scale", vec![0.5f32], vec![1], vec![6f32, 10., 5.5, 7.5]),
        ("scale_weight", vec![0.5f32], vec![], vec![6f32, 10., 5.5, 7.5]),
        ("weight_scale", vec![0.5f32, 2.], vec![2, 1], vec![6f32, 37., 5.5, 27.]),
    ] {
        let mut weights = HashMap::new();
        weights.insert("weight".into(), Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev)?.to_dtype(DType::F8E4M3)?);
        weights.insert(scale_name.into(), Tensor::from_vec(scales, shape, &dev)?);
        weights.insert("bias".into(), Tensor::from_slice(&[2f32, 1.], 2, &dev)?.to_dtype(DType::BF16)?);
        let layer = LayeredLinear::load(2, 2, true, true, None, VarBuilder::from_tensors(weights, DType::F32, &dev))?;
        assert_eq!(layer.forward(&x)?.flatten_all()?.to_vec1::<f32>()?, expected);
    }
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("weight".into(), Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev)?.to_dtype(DType::BF16)?);
    weights.insert("bias".into(), Tensor::from_slice(&[2f32, 1.], 2, &dev)?.to_dtype(DType::BF16)?);
    let layer = LayeredLinear::load(2, 2, true, false, None, VarBuilder::from_tensors(weights, DType::F32, &dev))?;
    assert_eq!(layer.forward(&x)?.flatten_all()?.to_vec1::<f32>()?, [10., 19., 9., 14.]);
    Ok(())
}

#[test]
fn exact_token_ids_and_cut() -> anyhow::Result<()> {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("Skipping tokenizer parity: {} absent", path.display());
        return Ok(());
    }
    let tok = load_tokenizer(path)?;
    let m = metadata()?;
    for case in ["positive", "negative"] {
        let prompt = &m["prompts"][case];
        let ids = tokenize(&tok, prompt["text"].as_str().ok_or_else(|| anyhow::anyhow!("missing text"))?)?;
        assert_eq!(serde_json::to_value(&ids)?, prompt["ids"]);
        assert_eq!(prefix_cut(&ids) as u64, prompt["prefix_cut"].as_u64().ok_or_else(|| anyhow::anyhow!("missing prefix cut"))?);
    }
    for probe in m["tokenizer_probes"].as_array().ok_or_else(|| anyhow::anyhow!("missing tokenizer probes"))? {
        let text = probe["text"].as_str().ok_or_else(|| anyhow::anyhow!("missing probe text"))?;
        let ids = tok.encode(text, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        assert_eq!(serde_json::to_value(ids.get_ids())?, probe["ids"], "probe {text:?}");
        assert_eq!(probe["ids"], probe["qwen3_ids"], "shared tokenizer probe {text:?}");
    }
    assert!(prompt_template("").contains("user\n<|im_end|>"));
    let custom = "<|im_start|>user\nx";
    assert_eq!(prompt_template(custom), custom);
    assert_eq!(prefix_cut(&[1, 2, 3]), 2);
    assert_eq!(prefix_cut(&[151644, 872, 198]), 0);
    assert_eq!(prefix_cut(&[151644, 872, 198, 7]), 3);
    Ok(())
}

#[test]
fn legacy_marker_and_scale_mapping() -> anyhow::Result<()> {
    use qwen_image_rs::layered_files::TensorFile;
    let path = std::env::temp_dir().join(format!("qir-qwen25-legacy-{}.safetensors", std::process::id()));
    let dev = Device::Cpu;
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("scaled_fp8".into(), Tensor::zeros(2, DType::F32, &dev)?);
    weights.insert("model.linear.weight".into(), Tensor::from_slice(&[1f32, 2.], (1, 2), &dev)?.to_dtype(DType::F8E4M3)?);
    weights.insert("model.linear.bias".into(), Tensor::from_slice(&[3f32], 1, &dev)?.to_dtype(DType::BF16)?);
    weights.insert("model.linear.scale_weight".into(), Tensor::new(0.5f32, &dev)?);
    weights.insert("model.linear.scale_input".into(), Tensor::new(1f32, &dev)?);
    candle_core::safetensors::save(&weights, &path)?;
    let run = || -> anyhow::Result<()> {
        let file = TensorFile::open(&path)?;
        let vb = file.builder("model.", DType::F32, &dev)?.pp("linear");
        assert!(vb.contains_tensor("weight_scale"));
        assert!(!vb.contains_tensor("input_scale"));
        let bytes = vb.get_unchecked_dtype("comfy_quant", DType::U8)?.to_vec1::<u8>()?;
        let record: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(record["full_precision_matrix_mult"], true);
        let layer = LayeredLinear::load(2, 1, true, false, None, vb)?;
        assert_eq!(layer.forward(&Tensor::from_slice(&[2f32, 3.], (1, 2), &dev)?)?.to_vec2::<f32>()?, [[7f32]]);
        drop(file);
        // A corrupt offset must be rejected before any tensor is read.
        let mut bytes = std::fs::read(&path)?;
        bytes.pop();
        std::fs::write(&path, bytes)?;
        assert!(TensorFile::open(&path).is_err());
        Ok(())
    };
    let result = run();
    std::fs::remove_file(path)?;
    result
}

#[test]
fn toy_decoder_final_norm_and_shapes() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let c = Qwen25Config {
        hidden_size: 16,
        intermediate_size: 24,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        vocab_size: 32,
        ..Qwen25Config::default()
    };
    let vars = candle_nn::VarMap::new();
    let model = Qwen25TextEncoder::load(c, VarBuilder::from_varmap(&vars, DType::F32, &dev))?;
    let out = model.forward_taps(&[1, 2, 3], &[0, 1])?;
    assert_eq!(out.conditioning.dims(), [1, 1, 16]);
    assert_eq!(out.taps.len(), 2);
    let last = &out.taps[1].1;
    let normed = candle_nn::ops::rms_norm(last, &Tensor::ones(16, DType::F32, &dev)?, 1e-6)?;
    let err = (out.conditioning - normed.narrow(1, 2, 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(err < 1e-5, "final norm error {err}");
    assert!(model.forward(&[]).is_err());
    assert!(model.forward_taps(&[1], &[2]).is_err());
    Ok(())
}

fn compare(key: &str, actual: &Tensor, reference: &Tensor, meta: &serde_json::Value) -> anyhow::Result<()> {
    let flat = actual.flatten_all()?.to_dtype(DType::F32)?;
    let actual = if meta["full"].as_bool() == Some(true) {
        actual.clone()
    } else {
        let indices = (0..4096).map(|i| (i * (flat.elem_count() - 1) / 4095) as u32).collect::<Vec<_>>();
        flat.index_select(&Tensor::from_vec(indices, 4096, &Device::Cpu)?, 0)?
    };
    assert_eq!(actual.dims(), reference.dims(), "{key}");
    let delta = (&actual - reference)?;
    let abs = delta.abs()?.max_all()?.to_scalar::<f32>()?;
    let relative = delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() / reference.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt().max(1e-12);
    let max_relative = delta.abs()?.broadcast_div(&reference.abs()?.clamp(1e-6, f32::MAX)?)?.max_all()?.to_scalar::<f32>()?;
    eprintln!("{key}: max abs {abs:.9}, relative RMS {relative:.9}, max relative {max_relative:.9}");
    // Measured CPU baseline: <= 0.00425 absolute and 8.2e-6 relative RMS.
    assert!(relative < 5e-5 && abs < 0.02, "{key}: abs={abs}, relative RMS={relative}");
    let values = flat.to_vec1::<f32>()?;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    let std = (values.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    for (name, value) in [
        ("mean", mean),
        ("std", std),
        ("min", values.iter().copied().fold(f32::INFINITY, f32::min) as f64),
        ("max", values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64),
    ] {
        let want = meta[name].as_f64().ok_or_else(|| anyhow::anyhow!("missing {key} {name}"))?;
        eprintln!("  {name}: {value:.9}, reference {want:.9}");
        assert!((value - want).abs() < 0.05f64.max(want.abs() * 5e-5), "{key}/{name}");
    }
    Ok(())
}

#[test]
#[ignore = "streams the real 7B language encoder on CPU"]
fn comfy_cpu_encoder_parity() -> anyhow::Result<()> {
    let path = root().join("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors");
    if !path.is_file() {
        eprintln!("Skipping encoder parity: {} absent", path.display());
        return Ok(());
    }
    let start = Instant::now();
    let model = Qwen25TextEncoder::from_file(path, DType::F32, &Device::Cpu)?;
    let m = metadata()?;
    let fixture = candle_core::safetensors::load_buffer(include_bytes!("fixtures/qwen25_text.safetensors"), &Device::Cpu)?;
    for case in ["positive", "negative"] {
        let ids: Vec<u32> = serde_json::from_value(m["prompts"][case]["ids"].clone())?;
        let timer = Instant::now();
        let out = model.forward_taps(&ids, &[0, 13, 27])?;
        for (li, tensor) in out.taps {
            let key = format!("{case}.layer{li}");
            compare(&key, &tensor, fixture.get(&key).ok_or_else(|| anyhow::anyhow!("missing fixture {key}"))?, &m[&key])?;
        }
        let key = format!("{case}.conditioning");
        compare(&key, &out.conditioning, fixture.get(&key).ok_or_else(|| anyhow::anyhow!("missing fixture {key}"))?, &m[&key])?;
        eprintln!("{case}: {:.3}s", timer.elapsed().as_secs_f64());
    }
    eprintln!("encoder parity total: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn edit_metadata() -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_str(include_str!("fixtures/qwen25_edit.json"))?)
}

#[test]
fn edit_token_spans_and_three_axis_positions() -> anyhow::Result<()> {
    use qwen_image_rs::model::qwen25_text::{edit_layout, edit_prompt_template};
    let meta = edit_metadata()?;
    let path = tokenizer_path();
    let tok = if path.exists() { Some(load_tokenizer(path)?) } else { None };
    for name in ["one", "two"] {
        let case = &meta["prompts"][name];
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone())?;
        let count = case["pictures"].as_u64().unwrap() as usize;
        let grids = (0..count).map(|i| serde_json::from_value(meta[format!("picture{i}")]["grid"].clone())).collect::<Result<Vec<[usize; 3]>, _>>()?;
        let (expanded, spans, positions) = edit_layout(&ids, &grids)?;
        assert_eq!(serde_json::to_value(expanded)?, case["expanded_ids"]);
        assert_eq!(serde_json::to_value(spans)?, case["spans"]);
        assert_eq!(serde_json::to_value(positions)?, case["positions"]);
        assert_eq!(prefix_cut(&ids), case["prefix_cut"].as_u64().unwrap() as usize);
        if let Some(tok) = &tok {
            let text = edit_prompt_template(case["text"].as_str().unwrap(), count)?;
            let actual = tok.encode(text, false).map_err(|e| anyhow::anyhow!("tokenize edit: {e}"))?;
            assert_eq!(actual.get_ids(), ids);
        }
    }
    assert!(edit_prompt_template("x", 4).is_err());
    assert!(edit_layout(&[151655], &[[1, 3, 4]]).is_err());
    assert!(edit_layout(&[1], &[[1, 4, 4]]).is_err());
    // Three pictures and t>1 exercise Qwen25's unusual height repetition exactly.
    let (ids, spans, pos) = edit_layout(&[7, 151655, 8, 151655, 9, 151655, 10], &[[1, 4, 6], [2, 4, 2], [1, 2, 2]])?;
    assert_eq!(spans, [(1, 6), (8, 4), (13, 1)]);
    assert_eq!(&pos[1][8..12], [5, 5, 6, 6]);
    assert_eq!(pos[0][14], 9);
    assert_eq!(ids.len(), 15);
    // A pad token masks following text; image spans remain attended and text positions compact.
    let (_, _, pos) = edit_layout(&[7, 151655, 151643, 8], &[[1, 4, 4]])?;
    assert_eq!(&pos[0][5..], [1, 1]);
    Ok(())
}

#[test]
#[ignore = "streams the installed 7B encoder on CPU; regenerate qwen25_edit_full.safetensors with tools/qir_qwen25_edit_fixtures.py first"]
fn comfy_cpu_edit_encoder_parity() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen25_edit_full.safetensors");
    let full = candle_core::safetensors::load(&fixture_path, &dev)?;
    let samples = candle_core::safetensors::load_buffer(include_bytes!("fixtures/qwen25_edit.safetensors"), &dev)?;
    let meta = edit_metadata()?;
    let tok = load_tokenizer(tokenizer_path())?;
    let model = Qwen25TextEncoder::from_file(root().join("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"), DType::F32, &dev)?;
    let mut passed = true;
    for name in ["one", "two"] {
        let case = &meta["prompts"][name];
        let count = case["pictures"].as_u64().unwrap() as usize;
        let pictures = (0..count)
            .map(|i| -> anyhow::Result<_> {
                Ok((full[&format!("picture{i}")].clone(), serde_json::from_value::<[usize; 3]>(meta[format!("picture{i}")]["grid"].clone())?))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let input = model.prepare_edit(&tok, case["text"].as_str().unwrap(), &pictures)?;
        assert_eq!(serde_json::to_value(&input.ids)?, case["expanded_ids"]);
        assert_eq!(serde_json::to_value(&input.spans)?, case["spans"]);
        assert_eq!(serde_json::to_value(&input.positions)?, case["positions"]);
        assert_eq!(
            serde_json::to_value(input.mask.flatten_all()?.to_vec1::<f32>()?)?,
            serde_json::to_value(serde_json::from_value::<Vec<f32>>(case["mask"].clone())?)?
        );
        // `absolute`: the conditioning's gate (max abs 0.02, its values reach ~120); the LM taps
        // reach 200-1,925, so they are held to a max abs of 1e-4 of their own peak instead. Both
        // keep the relative RMS gate of 5e-5.
        let check = |stage: &str, tensor: &Tensor, absolute: bool| -> anyhow::Result<bool> {
            let key = format!("{name}.{stage}");
            let reference = &full[&key];
            assert_eq!(tensor.dims(), reference.dims());
            let delta = (tensor - reference)?;
            let abs = delta.abs()?.max_all()?.to_scalar::<f32>()?;
            let relative = delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() / reference.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt().max(1e-12);
            let cap = if absolute { 0.02 } else { 1e-4 * reference.abs()?.max_all()?.to_scalar::<f32>()? };
            eprintln!("{key} (full): max abs {abs:.9} (cap {cap:.6}), relative RMS {relative:.9}");
            let peak = delta.abs()?.flatten_all()?.argmax(0)?.to_scalar::<u32>()? as usize;
            eprintln!("  peak row {}, channel {}", peak / 3584, peak % 3584);
            compare(&format!("{key} (checked-in samples)"), tensor, &samples[&key], &meta[&key])?;
            Ok(abs <= cap && relative <= 5e-5)
        };
        passed &= check("embeddings", &input.embeddings, true)?;
        let out = model.forward_edit_taps(&input, &[0, 13, 27])?;
        for (i, tensor) in out.taps {
            passed &= check(&format!("layer{i}"), &tensor, false)?;
        }
        passed &= check("conditioning", &out.conditioning, true)?;
        let mask = input.mask.narrow(1, input.cut, input.ids.len() - input.cut)?;
        assert_eq!((&mask - &full[&format!("{name}.mask")])?.abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
        assert_eq!(mask.dims(), &out.conditioning.dims()[..2]);
    }
    assert!(passed, "full edit tensors exceed max abs 0.02 or relative RMS 5e-5");
    Ok(())
}

#[test]
fn edit_cpu_rms_reference_without_installed_model() -> anyhow::Result<()> {
    use qwen_image_rs::model::qwen25_text::Qwen25EditInput;
    let dev = Device::Cpu;
    let fixtures = candle_core::safetensors::load_buffer(include_bytes!("fixtures/qwen25_edit.safetensors"), &dev)?;
    let mut weights = HashMap::new();
    weights.insert("embed_tokens.weight".into(), Tensor::zeros((8, 3584), DType::F32, &dev)?);
    weights.insert("norm.weight".into(), fixtures["rms.weight"].clone());
    let config = Qwen25Config { num_layers: 0, vocab_size: 8, ..Qwen25Config::default() };
    let model = Qwen25TextEncoder::load(config, VarBuilder::from_tensors(weights, DType::F32, &dev))?;
    let input = Qwen25EditInput {
        ids: vec![1; 8],
        spans: vec![],
        positions: std::array::from_fn(|_| (0..8).collect()),
        embeddings: fixtures["rms.input"].clone(),
        mask: Tensor::ones((1, 8), DType::F32, &dev)?,
        cut: 0,
    };
    let actual = model.forward_edit_taps(&input, &[])?.conditioning;
    let abs = (&actual - &fixtures["rms.output"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(abs < 2e-6, "ComfyUI RMSNorm max abs {abs}");
    Ok(())
}
