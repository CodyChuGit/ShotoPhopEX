//! Parity against ComfyUI's own `comfy/ldm/minimax/model.py` on a tiny random-weight model.
//! The fixtures are generated outside the repository; set `H3_FIXTURES` to the folder that
//! holds `h3_tiny_av`, `h3_tiny_pdd` and `h3_tiny_t2i.safetensors` to run these.

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};

use super::dit::H3Dit;
use super::layout::{text_tags, Layout};
use crate::moss::weights::Weights;

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("H3_FIXTURES")?).join(name);
    p.exists().then_some(p)
}

fn max_rel(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn run(name: &str, audio_t: Option<usize>, sample: &[f32], with_ref: bool) {
    let Some(path) = fixture(name) else {
        eprintln!("H3_FIXTURES not set or {name} missing: skipped");
        return;
    };
    let dev = Device::Cpu;
    let w = Weights::open(&path, &dev, DType::F32, false).unwrap();
    let net = H3Dit::load(&w).unwrap();
    let ctx = w.get_as("in.ctx", DType::F32).unwrap().i(0).unwrap();
    let video = w.get_as("in.video", DType::F32).unwrap().i(0).unwrap();
    let rf = w.get_as("in.ref", DType::F32).unwrap().i(0).unwrap();
    let sigma = w.get_as("in.sigma", DType::F32).unwrap().to_vec1::<f32>().unwrap()[0];
    let (_, _, lh, lw) = video.dims4().unwrap();
    let ref_dims: &[(usize, usize)] = if with_ref { &[(4, 4)] } else { &[] };
    let lay = Layout::new(text_tags(ctx.dim(0).unwrap()), ref_dims, audio_t, lh, lw);
    let p = net.prepare(&ctx, lay).unwrap();
    let want_text = w.get_as("out.text", DType::F32).unwrap().i(0).unwrap();
    let e = max_rel(&p.text, &want_text);
    assert!(e < 1e-4, "{name} text rel err {e}");
    let audio = audio_t.map(|_| w.get_as("in.audio", DType::F32).unwrap().i(0).unwrap());
    let (v, a) = net.forward(&p, &video, audio.as_ref(), if with_ref { std::slice::from_ref(&rf) } else { &[] }, 0, sigma, sample).unwrap();
    let want_v = w.get_as("out.video", DType::F32).unwrap().i(0).unwrap();
    let e = max_rel(&v, &want_v);
    eprintln!("{name} video rel err {e}");
    assert!(e < 1e-4, "{name} video rel err {e}");
    if let Some(a) = a {
        let want_a = w.get_as("out.audio", DType::F32).unwrap().i(0).unwrap();
        let e = max_rel(&a, &want_a);
        assert!(e < 1e-4, "{name} audio rel err {e}");
    }
}

#[test]
fn av_still_matches_comfy() {
    run("h3_tiny_av.safetensors", Some(2), &[], true);
}

#[test]
fn pdd_still_matches_comfy() {
    run("h3_tiny_pdd.safetensors", None, &[1.0, 0.7, 0.0], true);
}

/// Text to image: no reference pictures, first step (sigma 1.0).
#[test]
fn t2i_first_step_matches_comfy() {
    run("h3_tiny_t2i.safetensors", None, &[1.0, 0.7, 0.0], false);
}

#[test]
fn vae_matches_comfy() {
    let Some(path) = fixture("h3_tiny_vae.safetensors") else {
        eprintln!("H3_FIXTURES not set or h3_tiny_vae missing: skipped");
        return;
    };
    let w = Weights::open(&path, &Device::Cpu, DType::F32, false).unwrap();
    let vae = super::vae::H3Vae::load(&w).unwrap();
    let z = vae.decode(&w.get_as("in.latent", DType::F32).unwrap().i(0).unwrap()).unwrap();
    let want = w.get_as("out.pixels", DType::F32).unwrap().i(0).unwrap().squeeze(1).unwrap();
    let e = max_rel(&z, &want);
    eprintln!("vae decode rel err {e}");
    assert!(e < 1e-4, "decode rel err {e}");
    let px = w.get_as("in.pixels", DType::F32).unwrap().i(0).unwrap().squeeze(1).unwrap();
    let lat = vae.encode(&px).unwrap();
    let want = w.get_as("out.latent", DType::F32).unwrap().i(0).unwrap();
    let e = max_rel(&lat, &want);
    eprintln!("vae encode rel err {e}");
    assert!(e < 1e-4, "encode rel err {e}");
}

/// A LoRA beside the weights gives what the same LoRA merged into them gives (ComfyUI merges
/// it, `W + s·B A`), on the refiner and the blocks; taking it off restores the plain model.
#[test]
fn lora_side_path_matches_merged_weights() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::lora::{LoraFile, LoraStack};

    let Some(path) = fixture("h3_tiny_t2i.safetensors") else {
        eprintln!("H3_FIXTURES not set or h3_tiny_t2i missing: skipped");
        return;
    };
    let dev = Device::Cpu;
    let raw = candle_core::safetensors::load(&path, &dev).unwrap();
    let dir = std::env::temp_dir().join(format!("qir-h3-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut merged: HashMap<String, Tensor> = raw.clone();
    let mut lora: HashMap<String, Tensor> = HashMap::new();
    let modules = ["token_refiner.blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.1.mlp.fc1", "blocks.1.mlp.fc2"];
    let (rank, alpha, strength) = (4usize, 2f32, 0.8f32);
    for (i, m) in modules.iter().enumerate() {
        let w = &raw[&format!("{m}.weight")];
        let (out, inp) = w.dims2().unwrap();
        let a = (Tensor::randn(0f32, 1.0, (rank, inp), &dev).unwrap() * (0.3 / (inp as f64).sqrt())).unwrap();
        let b = (Tensor::randn(0f32, 1.0, (out, rank), &dev).unwrap() * (0.3 + 0.1 * i as f64)).unwrap();
        let s = f64::from(strength * alpha / rank as f32);
        merged.insert(format!("{m}.weight"), (w + (b.matmul(&a).unwrap() * s).unwrap()).unwrap());
        lora.insert(format!("diffusion_model.{m}.lora_A.weight"), a);
        lora.insert(format!("diffusion_model.{m}.lora_B.weight"), b);
        lora.insert(format!("diffusion_model.{m}.alpha"), Tensor::new(&[alpha], &dev).unwrap());
    }
    let merged_path = dir.join("merged.safetensors");
    let lora_path = dir.join("lora.safetensors");
    // keep the fixture's metadata (its PDD config)
    let bytes = std::fs::read(&path).unwrap();
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let info = meta.metadata().clone();
    safetensors::tensor::serialize_to_file(merged.iter().map(|(k, v)| (k.as_str(), v)), info, &merged_path).unwrap();
    candle_core::safetensors::save(&lora, &lora_path).unwrap();

    let run = |w: &Weights, stack: Option<&LoraStack>| -> Tensor {
        let net = H3Dit::load(w).unwrap();
        if let Some(s) = stack {
            net.set_loras(s).unwrap();
        }
        let ctx = w.get_as("in.ctx", DType::F32).unwrap().i(0).unwrap();
        let video = w.get_as("in.video", DType::F32).unwrap().i(0).unwrap();
        let (_, _, lh, lw) = video.dims4().unwrap();
        let p = net.prepare(&ctx, Layout::new(text_tags(ctx.dim(0).unwrap()), &[], None, lh, lw)).unwrap();
        let v = net.forward(&p, &video, None, &[], 0, 0.6, &[1.0, 0.7, 0.0]).unwrap().0;
        if stack.is_some() {
            net.set_loras(&LoraStack::default()).unwrap();
            // the refined text depends on the refiner's adapters: prepare again
            let p = net.prepare(&ctx, Layout::new(text_tags(ctx.dim(0).unwrap()), &[], None, lh, lw)).unwrap();
            let plain = net.forward(&p, &video, None, &[], 0, 0.6, &[1.0, 0.7, 0.0]).unwrap().0;
            let base = H3Dit::load(&Weights::open(&path, &dev, DType::F32, false).unwrap()).unwrap();
            let pb = base.prepare(&ctx, Layout::new(text_tags(ctx.dim(0).unwrap()), &[], None, lh, lw)).unwrap();
            let want = base.forward(&pb, &video, None, &[], 0, 0.6, &[1.0, 0.7, 0.0]).unwrap().0;
            assert!(max_rel(&plain, &want) < 1e-6, "taking the LoRA off restores the model");
        }
        v
    };
    let file = Arc::new(LoraFile::load(&lora_path, &dev, DType::F32).unwrap());
    let stack = LoraStack { layers: vec![(file, strength)] };
    let side = run(&Weights::open(&path, &dev, DType::F32, false).unwrap(), Some(&stack));
    let want = run(&Weights::open(&merged_path, &dev, DType::F32, false).unwrap(), None);
    let plain = run(&Weights::open(&path, &dev, DType::F32, false).unwrap(), None);
    let e = max_rel(&side, &want);
    eprintln!("lora side path vs merged rel err {e}, effect {}", max_rel(&plain, &want));
    assert!(e < 1e-4, "side path vs merged rel err {e}");
    assert!(max_rel(&plain, &want) > 1e-3, "the LoRA changes the output");

    // a LoRA for another model is refused
    let mut other = HashMap::new();
    other.insert("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight".to_string(), Tensor::zeros((2, 4), DType::F32, &dev).unwrap());
    other.insert("diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight".to_string(), Tensor::zeros((4, 2), DType::F32, &dev).unwrap());
    let other_path = dir.join("other.safetensors");
    candle_core::safetensors::save(&other, &other_path).unwrap();
    let net = H3Dit::load(&Weights::open(&path, &dev, DType::F32, false).unwrap()).unwrap();
    let bad = LoraStack { layers: vec![(Arc::new(LoraFile::load(&other_path, &dev, DType::F32).unwrap()), 1.0)] };
    assert!(net.set_loras(&bad).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// P-H3's W4A8 files: a layer comfy_kitchen quantized (with and without a codebook) reads back to
/// the weight comfy_kitchen dequantizes it to.
#[test]
fn w4a8_reads_like_comfy_kitchen() {
    let Some(path) = fixture("h3_w4a8.safetensors") else {
        eprintln!("H3_FIXTURES not set or h3_w4a8 missing: skipped");
        return;
    };
    let w = Weights::open(&path, &Device::Cpu, DType::F32, false).unwrap();
    for m in ["a", "b"] {
        assert!(w.is_quantized(m));
        let got = crate::moss::weights::dequantize(&w.prequantized(m).unwrap()).unwrap();
        let want = w.get_as(&format!("ref.{m}"), DType::F32).unwrap();
        let e = max_rel(&got, &want);
        eprintln!("w4a8 {m} rel err {e}");
        assert!(e < 1e-5, "w4a8 {m} rel err {e}");
    }
}
