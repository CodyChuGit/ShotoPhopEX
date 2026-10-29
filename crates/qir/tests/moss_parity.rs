//! MOSS-Image against the official pipeline on the same files (`ref_moss.py`'s outputs: the
//! Hugging Face Qwen3.5 encoder, the official DiT and ComfyUI's FLUX.2 VAE on Comfy-Org's
//! INT8 weights dequantized by ComfyUI). Needs the weights and the reference outputs, so every
//! test is ignored by default:
//!
//! `MOSS_ROOT=<folder with diffusion_models/ text_encoders/ vae/ tokenizer/ ref_out/>
//!  cargo test --release --features cuda,convrot,flash --test moss_parity -- --ignored --test-threads=1 --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use qwen_image_rs::moss::dit::{MossConfig, MossDit};
use qwen_image_rs::moss::pipeline::{MossFiles, MossPipeline};
use qwen_image_rs::moss::text::{self, Qwen35Text, TextConfig};
use qwen_image_rs::moss::vae::Flux2Decoder;
use qwen_image_rs::moss::weights::Weights;
use qwen_image_rs::pipeline::Request;
use qwen_image_rs::Result;

fn root() -> PathBuf {
    PathBuf::from(std::env::var("MOSS_ROOT").unwrap_or_else(|_| r"D:\PhotoCraftModels\moss-image".into()))
}

fn files() -> MossFiles {
    let r = root();
    MossFiles {
        dit: r.join("diffusion_models/moss_image_sft_int8_convrot.safetensors"),
        text: r.join("text_encoders/qwen3.5_9b_int8_convrot.safetensors"),
        vae: r.join("vae/flux2-vae_small.safetensors"),
        tokenizer: r.join("tokenizer/tokenizer.json"),
    }
}

fn reference(name: &str) -> Result<std::collections::HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(root().join("ref_out").join(name), &Device::Cpu)?)
}

fn dev() -> Result<(Device, DType)> {
    let dev = qwen_image_rs::device::best_device()?;
    let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    Ok((dev, dt))
}

/// Cosine similarity and RMS(a − b)/RMS(b), flattened, in f64.
fn compare(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let a: Vec<f64> = a.flatten_all()?.to_dtype(DType::F64)?.to_device(&Device::Cpu)?.to_vec1()?;
    let b: Vec<f64> = b.flatten_all()?.to_dtype(DType::F64)?.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(a.len(), b.len());
    let (mut ab, mut aa, mut bb, mut dd) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
    }
    Ok((ab / (aa.sqrt() * bb.sqrt()), (dd / bb).sqrt()))
}

fn psnr(a: &Tensor, b: &Tensor) -> Result<f64> {
    let mse = (a - b)?.sqr()?.mean_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    Ok(10.0 * (1.0 / mse).log10())
}

#[test]
#[ignore = "needs MOSS-Image's weights and ref_moss.py's outputs"]
fn tokens_and_encoder_match() -> Result<()> {
    let (dev, dt) = dev()?;
    let f = files();
    let tok = tokenizers::Tokenizer::from_file(&f.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let te = Qwen35Text::load(&TextConfig::qwen35_9b(), &Weights::open(&f.text, &dev, dt, true)?)?;
    let r = reference("te.safetensors")?;
    for (name, prompt) in [("cat", "a lovely cat holding a sign says 'moss-image'"), ("empty", "")] {
        let ids = tok.encode(text::template(prompt), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        let want: Vec<i64> = r[&format!("{name}.ids")].to_vec1()?;
        assert_eq!(ids.iter().map(|&i| i64::from(i)).collect::<Vec<_>>(), want, "{name}: token ids");
        let h = te.forward(&ids)?;
        let (cos, rel) = compare(&h, &r[&format!("{name}.hidden")])?;
        println!("encoder {name}: {} tokens, cosine {cos:.6}, relative RMS {rel:.4}", ids.len());
        assert!(cos > 0.99, "{name}: cosine {cos}");
    }
    Ok(())
}

#[test]
#[ignore = "needs MOSS-Image's weights and ref_moss.py's outputs"]
fn dit_velocity_matches() -> Result<()> {
    let (dev, dt) = dev()?;
    let dit = MossDit::load(&MossConfig::default(), &Weights::open(&files().dit, &dev, dt, true)?)?;
    let te = reference("te.safetensors")?;
    let r = reference("dit.safetensors")?;
    let x = r["noise"].to_device(&dev)?;
    let pos = te["cat.hidden"].to_device(&dev)?.to_dtype(dt)?;
    let neg = te["empty.hidden"].to_device(&dev)?.to_dtype(dt)?;
    for t in [1000u32, 500] {
        let v = dit.forward(&x, t as f32, &pos, &[])?;
        let (cos, rel) = compare(&v, &r[&format!("v_pos_t{t}")])?;
        println!("DiT positive t={t}: cosine {cos:.6}, relative RMS {rel:.4}");
        assert!(cos > 0.995, "positive t={t}: cosine {cos}");
        let v = dit.forward(&x, t as f32, &neg, &[22])?;
        let (cos, rel) = compare(&v, &r[&format!("v_neg_skip_t{t}")])?;
        println!("DiT negative (block 22 skipped) t={t}: cosine {cos:.6}, relative RMS {rel:.4}");
        assert!(cos > 0.995, "negative t={t}: cosine {cos}");
        // Both branches in one batched pass.
        let (vp, vn) = dit.forward_pair(&x, t as f32, &pos, &neg, &[22])?;
        let (cp, _) = compare(&vp, &r[&format!("v_pos_t{t}")])?;
        let (cn, _) = compare(&vn, &r[&format!("v_neg_skip_t{t}")])?;
        println!("DiT pair t={t}: positive cosine {cp:.6}, negative {cn:.6}");
        assert!(cp > 0.995 && cn > 0.995, "pair t={t}: {cp} {cn}");
    }
    Ok(())
}

#[test]
#[ignore = "needs MOSS-Image's weights and ref_moss.py's outputs"]
fn vae_decode_matches() -> Result<()> {
    let (dev, _) = dev()?;
    let r = reference("run_cat_512x512_s20_seed42.safetensors")?;
    let theirs = r["img"].squeeze(0)?;
    for dt in [DType::F32, DType::BF16] {
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[files().vae], dt, &dev)? };
        let vae = Flux2Decoder::load(vb)?;
        let latent = r["latent"].to_device(&dev)?;
        let _ = vae.decode(&latent)?;
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        let img = vae.decode(&latent)?; // (1, 3, H, W) in [-1, 1]
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f32();
        let ours = ((img.squeeze(0)?.permute((1, 2, 0))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)? + 1.0)? * 0.5)?;
        let p = psnr(&ours, &theirs)?;
        println!("VAE decode {dt:?}: PSNR {p:.2} dB against ComfyUI's, {secs:.3} s at 512²");
        assert!(p > 40.0, "PSNR {p}");
    }
    Ok(())
}

#[test]
#[ignore = "needs MOSS-Image's weights and ref_moss.py's outputs"]
fn picture_matches() -> Result<()> {
    let p = MossPipeline::load(&files(), &mut |_, _| {})?;
    let mut req = Request::new("a lovely cat holding a sign says 'moss-image'");
    req.width = 512;
    req.height = 512;
    req.steps = 20;
    req.seed = 42;
    req.guidance = 3.0;
    req.negative = String::new();
    let pic = p.generate(&req, &mut |_, _| {}, &|| false)?;
    let out = root().join("ref_out").join("rust_cat_512x512_s20_seed42.png");
    image::save_buffer(&out, &pic.rgba, pic.width, pic.height, image::ColorType::Rgba8)?;
    let ours: Vec<f32> = pic.rgba.chunks(4).flat_map(|px| px[..3].iter().map(|&v| f32::from(v) / 255.0)).collect();
    let ours = Tensor::from_vec(ours, (512, 512, 3), &Device::Cpu)?;
    let r = reference("run_cat_512x512_s20_seed42.safetensors")?;
    let theirs = r["img"].squeeze(0)?;
    let db = psnr(&ours, &theirs)?;
    println!("picture: PSNR {db:.2} dB against the reference, {:?}, saved {}", pic.timings, out.display());
    assert!(db > 20.0, "PSNR {db}");
    Ok(())
}

/// The official defaults at 1024², for the timings (no reference to compare with).
#[test]
#[ignore = "needs MOSS-Image's weights"]
fn picture_at_the_defaults() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).try_init();
    let p = MossPipeline::load(&files(), &mut |_, _| {})?;
    let mut req = Request::new(std::env::var("MOSS_PROMPT").unwrap_or_else(|_| {
        "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten: \"Buy milk and eggs!\"".into()
    }));
    req.width = 1024;
    req.height = 1024;
    req.steps = qwen_image_rs::moss::pipeline::STEPS;
    req.seed = 42;
    req.guidance = qwen_image_rs::moss::pipeline::GUIDANCE;
    req.negative = String::new();
    for run in 0..2 {
        let pic = p.generate(&req, &mut |_, _| {}, &|| false)?;
        let out = root().join("ref_out").join(format!("rust_default_1024_seed42_run{run}.png"));
        image::save_buffer(&out, &pic.rgba, pic.width, pic.height, image::ColorType::Rgba8)?;
        println!("1024² 50 steps, run {run}: {:?}, saved {}", pic.timings, out.display());
    }
    Ok(())
}
