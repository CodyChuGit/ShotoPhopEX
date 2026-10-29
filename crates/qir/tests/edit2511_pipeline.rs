use candle_core::{Device, Tensor};
use qwen_image_rs::{
    edit2511_pipeline::{denoised_cfg, edit2511_schedule, masked_replace},
    Result,
};
#[test]
fn mask_and_cfgnorm_boundaries() -> Result<()> {
    let x = Tensor::from_slice(&[2f32, 4.], (1, 2, 1, 1, 1), &Device::Cpu)?;
    let zero = x.zeros_like()?;
    let pos = Tensor::from_slice(&[-1f32, -2.], x.shape(), &Device::Cpu)?;
    let got = denoised_cfg(&x, &pos, Some(&zero), 1., 4.)?;
    let expected = Tensor::from_slice(&[3f32, 6.], x.shape(), &Device::Cpu)?;
    assert!((&got - &expected)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
    let mask = Tensor::full(0.25f32, x.shape(), &Device::Cpu)?;
    let mixed = masked_replace(&x, &zero, Some(&mask))?;
    assert_eq!(mixed.flatten_all()?.to_vec1::<f32>()?, vec![0.5, 1.]);
    assert_eq!(masked_replace(&x, &zero, None)?.flatten_all()?.to_vec1::<f32>()?, vec![2., 4.]);
    let sigmas = edit2511_schedule(8)?.sigmas;
    assert_eq!(sigmas.len(), 9);
    assert_eq!((sigmas[0], sigmas[8]), (1., 0.));
    assert!((sigmas[4] - 3.1 * 0.5 / (1. + 2.1 * 0.5)).abs() < 1e-7);
    Ok(())
}
#[test]
fn sampler_matches_installed_comfy_with_oracle_velocities() -> Result<()> {
    let f = candle_core::safetensors::load_buffer(include_bytes!("fixtures/edit2511_sampler.safetensors"), &Device::Cpu)?;
    let check = |name: &str, a: &Tensor| -> Result<()> {
        let max = (a - &f[name])?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max <= 2e-6, "{name}: {max}");
        Ok(())
    };
    let sigmas = edit2511_schedule(8)?.sigmas;
    check("sigmas", &Tensor::from_slice(&sigmas, 9, &Device::Cpu)?)?;
    let noise = Tensor::from_vec(qwen_image_rs::comfy::torch_randn(7, 240).map_err(anyhow::Error::msg)?, f["noise"].shape(), &Device::Cpu)?;
    check("noise", &noise)?;
    let mut x = noise.clone();
    check("initial", &x)?;
    for i in 0..8 {
        let sigma = sigmas[i];
        let protected = ((&noise * sigma as f64)? + (&f["target"] * (1. - sigma) as f64)?)?;
        let input = masked_replace(&x, &protected, Some(&f["mask"]))?;
        check(&format!("step_{i}.input"), &input)?;
        let pred = denoised_cfg(&input, &f["positive"], Some(&f["negative"]), sigma, 4.)?;
        let pred = masked_replace(&pred, &f["target"], Some(&f["mask"]))?;
        check(&format!("step_{i}.denoised"), &pred)?;
        x = (&x + (((&x - pred)? / sigma as f64)? * (sigmas[i + 1] - sigma) as f64)?)?;
        check(&format!("step_{i}.latent"), &x)?;
    }
    Ok(())
}

#[test]
fn staged_binding_overlay_reloads_exact_weights() -> Result<()> {
    use qwen_image_rs::model::qwen_image_dit::{checkpoint_builder, checkpoint_builder_patched};
    let path = std::env::temp_dir().join(format!("qir-edit2511-overlay-{}.safetensors", std::process::id()));
    let original = Tensor::full(1f32, (2, 2), &Device::Cpu)?;
    let bias = Tensor::from_slice(&[2f32, 3.], 2, &Device::Cpu)?;
    candle_core::safetensors::save(&std::collections::HashMap::from([("weight".to_owned(), original), ("bias".to_owned(), bias.clone())]), &path)?;
    let patches = std::collections::BTreeMap::from([("weight".to_owned(), Tensor::full(7f32, (2, 2), &Device::Cpu)?)]);
    let builder = checkpoint_builder_patched(&path, &Device::Cpu, patches)?;
    assert_eq!(builder.get((2, 2), "weight")?.to_vec2::<f32>()?, vec![vec![7., 7.], vec![7., 7.]]);
    assert_eq!(builder.get(2, "bias")?.to_vec1::<f32>()?, bias.to_vec1::<f32>()?);
    assert!(builder.get((3, 2), "weight").is_err());
    let baseline = checkpoint_builder(&path, &Device::Cpu)?;
    assert_eq!(baseline.get((2, 2), "weight")?.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);
    drop(builder);
    drop(baseline);
    std::fs::remove_file(path)?;
    Ok(())
}

#[cfg(feature = "fp8")]
#[test]
#[ignore = "requires tools/edit2511_lora_probe.py and a free CUDA device"]
fn installed_fp8_patched_weight() -> Result<()> {
    use qwen_image_rs::{
        edit2511_pipeline::key_seed,
        fp8::{requantize_full_precision, Fp8Linear},
    };
    let dev = Device::new_cuda(0)?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/edit2511/evidence");
    for (label, module) in
        [("attention", "transformer_blocks.0.attn.to_q"), ("mlp", "transformer_blocks.0.img_mlp.net.2"), ("modulation", "transformer_blocks.1.img_mod.1")]
    {
        let f = candle_core::safetensors::load(root.join(format!("fp8-patch-{label}.safetensors")), &dev)?;
        let mut weight = Fp8Linear::new_full_precision(f["q"].clone(), &f["scale"])?.precise_scale().dequantized(candle_core::DType::BF16)?;
        let mut i = 0;
        while let Some(a) = f.get(&format!("a{i}")) {
            let b = &f[&format!("b{i}")];
            let strength = f[&format!("strength{i}")].to_scalar::<f32>()?;
            weight = (weight + (b.matmul(a)? * strength as f64)?)?;
            i += 1;
        }
        let (q, scale) = requantize_full_precision(&weight, key_seed(module))?;
        let actual = q.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
        let expected = f["expected_q"].to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<float8::F8E4M3>()?;
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!(
            "{label}: {mismatches} / {} FP8 byte mismatches; scale {:?} / {:?}",
            q.elem_count(),
            scale.to_scalar::<f32>()?,
            f["expected_scale"].to_scalar::<f32>()?
        );
        assert_eq!(scale.to_scalar::<f32>()?, f["expected_scale"].to_scalar::<f32>()?);
        assert_eq!(mismatches, 0, "{label}");
        let output = Fp8Linear::new_full_precision(q, &scale)?.precise_scale().forward(&f["input"])?;
        let error = (output.to_dtype(candle_core::DType::F32)? - f["output"].to_dtype(candle_core::DType::F32)?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let norm = f["output"].to_dtype(candle_core::DType::F32)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let relative = (error / norm.max(1e-20)).sqrt();
        eprintln!("{label}: projection relative RMS {relative}");
        assert!(relative <= 0.01, "{label}: {relative}");
    }
    Ok(())
}
