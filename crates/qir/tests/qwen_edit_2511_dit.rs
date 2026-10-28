//! Complete-tensor CPU parity against ComfyUI's own 2511 class.
use candle_core::{DType, Device, Tensor};
use qwen_image_rs::model::qwen_image_dit::{
    checkpoint_builder, edit_timestep_projection, text_positions, DitKind, LayeredBlock, LayeredDit, LayeredInput, LayeredRope, PackedImages,
};
use std::{collections::HashMap, path::PathBuf};

fn fixture(name: &str) -> anyhow::Result<HashMap<String, Tensor>> {
    let extension = if name == "geometry" { "bin" } else { "safetensors" };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/qwen_edit_2511_{name}.{extension}"));
    Ok(candle_core::safetensors::load(path, &Device::Cpu)?)
}

fn checkpoint() -> PathBuf {
    std::env::var_os("QIR_EDIT2511_DIT").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\qwen_image_edit_2511_fp8mixed.safetensors")
    })
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, max_limit: f64, rms_limit: f64) -> anyhow::Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{name}");
    let a = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let e = expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (mut max, mut error, mut reference) = (0f64, 0f64, 0f64);
    for (a, e) in a.iter().zip(e.iter()) {
        assert!(a.is_finite() && e.is_finite(), "{name}: nonfinite value");
        let diff = f64::from(*a) - f64::from(*e);
        max = max.max(diff.abs());
        error += diff * diff;
        reference += f64::from(*e).powi(2);
    }
    let relative = (error / a.len() as f64).sqrt() / (reference / a.len() as f64).sqrt().max(1e-12);
    eprintln!("{name}: max_abs={max:.9e}, relative_RMS={relative:.9e}");
    anyhow::ensure!(max <= max_limit && relative <= rms_limit, "{name}: max={max}, relative RMS={relative}");
    Ok(())
}

#[test]
fn packed_positions_rope() -> anyhow::Result<()> {
    let f = fixture("geometry")?;
    compare("time_projection", &edit_timestep_projection(&Tensor::new(&[0.625f32, 0., 0.125, 0.999], &Device::Cpu)?)?, &f["time_projection"], 1e-7, 1e-7)?;
    let references = vec![f["ref1"].clone(), f["ref2"].clone(), f["ref3"].clone()];
    let packed = PackedImages::new(DitKind::Edit2511, &f["target"], &references)?;
    assert_eq!(packed.target_tokens, 9);
    compare("packed", &packed.tokens, &f["packed"], 0., 0.)?;
    compare("image_ids", &packed.positions, &f["image_ids"], 0., 0.)?;
    let text = text_positions(1, 5, packed.rows, packed.cols, &Device::Cpu)?;
    compare("text_ids", &text, &f["text_ids"], 0., 0.)?;
    let rope = LayeredRope::new(&Tensor::cat(&[&text, &packed.positions], 1)?, DType::F32)?;
    compare("rope", &rope.matrices()?, &f["rope"], 0., 0.)?;
    // Same independent reference grids, opposite frame sign for Layered.
    let layered = PackedImages::new(DitKind::Layered, &f["target"], &references)?;
    assert_eq!(layered.positions.narrow(1, 9, 1)?.to_vec3::<f32>()?, vec![vec![vec![-1., -1., -2.]]]);
    assert_eq!(packed.positions.narrow(1, 23, 1)?.to_vec3::<f32>()?, vec![vec![vec![3., 0., 0.]]]);
    compare("target without references", &PackedImages::new(DitKind::Edit2511, &f["target"], &[])?.tokens, &f["packed"].narrow(1, 0, 9)?, 0., 0.)?;
    // Batch packing and the no-reference path retain target boundaries.
    let batch = f["target"].repeat((2, 1, 1, 1, 1))?;
    assert_eq!(PackedImages::new(DitKind::Edit2511, &batch, &[])?.tokens.dims(), &[2, 9, 64]);
    Ok(())
}

#[test]
#[ignore = "installed 2511 checkpoint; regenerate real fixture with tools/qwen_edit_2511_fixtures.py --full; CPU, one block at a time"]
fn selected_checkpoint_modules() -> anyhow::Result<()> {
    let f = fixture("geometry")?;
    let real = fixture("real")?;
    let vb = checkpoint_builder(checkpoint(), &Device::Cpu)?;
    assert_eq!(DitKind::detect(&vb)?, DitKind::Edit2511);
    let input = LayeredInput::load(vb.clone())?;
    let temb = input.timestep(&Tensor::new(&[0.625f32, 0.], &Device::Cpu)?, None)?;
    for i in 0..2 {
        compare(&format!("temb_{i}"), &temb.narrow(0, i, 1)?, &real["temb"].narrow(0, i, 1)?, 0.001, 1e-4)?;
    }
    drop(input);
    let packed = PackedImages::new(DitKind::Edit2511, &f["target"], &[f["ref1"].clone()])?;
    let tp = text_positions(1, 5, packed.rows, packed.cols, &Device::Cpu)?;
    let rope = LayeredRope::new(&Tensor::cat(&[tp, packed.positions], 1)?, DType::F32)?;
    let mut failures = Vec::new();
    for i in [0, 29, 59] {
        let block = LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))?;
        // Identical boundary inputs isolate each block from upstream rounding.
        for half in 0..2 {
            let (modulated, gates) = block.image_conditioning(&real["block_input_image"], &real["temb"], half, 9)?;
            for (suffix, value) in [(format!("modulated_{half}"), modulated), (format!("gates_{half}"), gates)] {
                let name = format!("block_{i}_{suffix}");
                compare(&name, &value, &real[&name], 0.005, 1e-4)?;
            }
        }
        let (img, txt) = block.forward_split(&real["block_input_image"], &real["block_input_text"], &real["block_input_time"], &rope, None, Some(9))?;
        if i == 0 {
            let btime =
                Tensor::cat(&[real["block_input_time"].narrow(0, 0, 1)?.repeat((2, 1))?, real["block_input_time"].narrow(0, 1, 1)?.repeat((2, 1))?], 0)?;
            let (bi, bt) = block.forward_split(
                &real["block_input_image"].repeat((2, 1, 1))?,
                &real["block_input_text"].repeat((2, 1, 1))?,
                &btime,
                &rope,
                None,
                Some(9),
            )?;
            compare("block_0_batch_image", &bi, &real["block_0_image"].repeat((2, 1, 1))?, 0.005, 1e-4)?;
            compare("block_0_batch_text", &bt, &real["block_0_text"].repeat((2, 1, 1))?, 0.005, 1e-4)?;
        }
        for (suffix, value) in [("image", img), ("text", txt)] {
            let name = format!("block_{i}_{suffix}");
            if let Err(error) = compare(&name, &value, &real[&name], 0.005, 1e-4) {
                failures.push(error.to_string());
            }
        }
    }
    anyhow::ensure!(failures.is_empty(), "{}", failures.join("; "));
    Ok(())
}

#[test]
#[ignore = "installed 2511 checkpoint and regenerated full oracle; streams 60 blocks per reference count on CPU"]
fn full_streamed_velocity() -> anyhow::Result<()> {
    let f = fixture("geometry")?;
    let real = fixture("real")?;
    let model = LayeredDit::open_streamed(checkpoint(), &Device::Cpu)?;
    let references = [f["ref1"].clone(), f["ref2"].clone()];
    for count in [1, 2] {
        let result = model.forward_latents(&f["target"], &references[..count], &f["text"], &Tensor::new(&[0.625f32], &Device::Cpu)?)?;
        let name = format!("velocity_{count}");
        compare(&name, &result, &real[&name], 0.002, 5e-4)?;
    }
    Ok(())
}
