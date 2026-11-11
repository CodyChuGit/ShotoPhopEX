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
