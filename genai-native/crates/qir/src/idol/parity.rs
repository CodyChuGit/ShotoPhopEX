//! Parity against ComfyUI's own TripoSplat code (`comfy/ldm/triposplat`, `comfy/image_encoders/
//! dino3.py`) on tiny random-weight models, in f32 on the CPU. The fixtures are made outside the
//! repository; set `IDOL_FIXTURES` to the folder holding `idol_misc`, `idol_tiny_dit`,
//! `idol_tiny_dino` and `idol_tiny_decoder.safetensors` to run these.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use super::decoder::{IdolDecoder, TorchRand};
use super::dino::Dino;
use super::dit::IdolDit;
use super::nn::{log2_freqs, point_embedding, pow2_freqs};
use crate::comfy::FlowSchedule;
use crate::moss::weights::Weights;

fn fixture(name: &str) -> Option<Weights> {
    let p = PathBuf::from(std::env::var_os("IDOL_FIXTURES")?).join(name);
    if !p.exists() {
        eprintln!("{name} missing: skipped");
        return None;
    }
    Some(Weights::open(&p, &Device::Cpu, DType::F32, false).unwrap())
}

fn get(w: &Weights, name: &str) -> Tensor {
    w.get_as(name, DType::F32).unwrap()
}

fn vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn sobol_noise_and_schedule_match_torch() {
    let Some(w) = fixture("idol_misc.safetensors") else { return };
    let sobol = super::sobol::scrambled_3d(8192, 123);
    assert_eq!(sobol, vec(&get(&w, "sobol")), "sobol anchors");
    let (z, cam) = super::pipeline::noise(42).unwrap();
    // torch's Windows build fills randn with its AVX2 Box-Muller, which rounds a last bit
    // differently from the scalar f64 transform here: same draws, ~1 ulp apart.
    let e = rel(&z, &vec(&get(&w, "noise_z")));
    assert!(e < 1e-5, "latent noise rel err {e}");
    let e = rel(&cam, &vec(&get(&w, "noise_cam")));
    assert!(e < 1e-6, "camera noise rel err {e}: {cam:?} vs {:?}", vec(&get(&w, "noise_cam")));
    let s = FlowSchedule::simple(50, super::pipeline::SHIFT).unwrap();
    let e = rel(&s.sigmas, &vec(&get(&w, "sigmas")));
    assert!(e < 1e-6, "sigmas rel err {e}");
    let x = Tensor::from_vec(sobol.clone(), (8192, 3), &Device::Cpu).unwrap();
    let pe = point_embedding(&x, &pow2_freqs(64, 3, 16), 64).unwrap();
    let e = rel(&vec(&pe), &vec(&get(&w, "pos_emb64")));
    assert!(e < 1e-5, "pow2 embedding rel err {e}");
    let pe = point_embedding(&x.narrow(0, 0, 100).unwrap(), &log2_freqs(64, 3, 10), 64).unwrap();
    let e = rel(&vec(&pe), &vec(&get(&w, "pos_log2_64")));
    assert!(e < 1e-5, "log2 embedding rel err {e}");
}

#[test]
fn dit_matches_comfy() {
    let Some(w) = fixture("idol_tiny_dit.safetensors") else { return };
    let dit = IdolDit::load(&w).unwrap();
    let ctx = dit.context(&get(&w, "in.feat1"), &get(&w, "in.feat2")).unwrap();
    let ctx = dit.refine_context(&ctx).unwrap();
    let t = vec(&get(&w, "in.t"))[0];
    let (vz, vc) = dit.forward(&get(&w, "in.z"), &get(&w, "in.cam"), t, &ctx).unwrap();
    let e = rel(&vec(&vz), &vec(&get(&w, "out.vz")));
    eprintln!("dit latent rel err {e}");
    assert!(e < 1e-4, "dit latent rel err {e}");
    let e = rel(&vec(&vc), &vec(&get(&w, "out.vcam")));
    assert!(e < 1e-4, "dit camera rel err {e}");
}

#[test]
fn dino_matches_comfy() {
    let Some(w) = fixture("idol_tiny_dino.safetensors") else { return };
    let f = Dino::load(&w).unwrap().features(&get(&w, "in.rgb")).unwrap();
    let e = rel(&vec(&f), &vec(&get(&w, "out.feature1")));
    eprintln!("dino rel err {e}");
    assert!(e < 1e-4, "dino rel err {e}");
}

#[test]
fn decoder_matches_comfy() {
    let Some(w) = fixture("idol_tiny_decoder.safetensors") else { return };
    let dec = IdolDecoder::load(&w).unwrap();
    let latent = get(&w, "in.latent");
    let e = rel(&dec.octree_logits(&get(&w, "in.oct_x").unsqueeze(0).unwrap(), 4, &latent).unwrap(), &vec(&get(&w, "out.oct_logits")));
    assert!(e < 1e-4, "octree logits rel err {e}");

    let want = vec(&get(&w, "out.points"));
    let pts = dec.sample_points(&latent, want.len() / 3, 8, &mut TorchRand::new(7)).unwrap();
    let e = rel(&pts, &want);
    eprintln!("octree points rel err {e}");
    assert!(e < 1e-5, "octree points rel err {e}");

    let fixed = vec(&get(&w, "in.fixed"));
    let s = dec.gaussians(&latent, &fixed).unwrap();
    for (name, got) in [
        ("positions", &s.positions),
        ("scales", &s.scales),
        ("rotations", &s.rotations),
        ("opacities", &s.opacities),
        ("sh", &s.sh_dc),
    ] {
        let e = rel(got, &vec(&get(&w, &format!("out.{name}"))));
        eprintln!("gaussian {name} rel err {e}");
        assert!(e < 1e-4, "gaussian {name} rel err {e}");
    }
}
