//! Reads Comfy-Org's Krea 2 Turbo NVFP4 file against its fp8 file: which byte layout the NVFP4
//! weights are in (swizzled or plain block scales, high or low nibble first), and how far the
//! FP4 GEMM lands from the fp8 weights' bf16 matmul. Needs both files in a ComfyUI models folder
//! named by `QIR_NVFP4_CHECK_MODELS`, and a CUDA build (`--features fp8`); skipped otherwise.
//!
//! `QIR_NVFP4_CHECK_MODELS=<models> cargo test --release --features fp8 --test nvfp4_file_check -- --nocapture`

#![cfg(feature = "fp8")]

use candle_core::{DType, Device, Tensor};
use qwen_image_rs::nvfp4::{self, Layout};

fn rel(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap();
    let b = b.to_dtype(DType::F32).unwrap();
    let num = (&a - &b).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    let den = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    (num / den).sqrt()
}

#[test]
fn krea2_nvfp4_matches_its_fp8_file() -> anyhow::Result<()> {
    let Some(root) = std::env::var_os("QIR_NVFP4_CHECK_MODELS").map(std::path::PathBuf::from) else {
        eprintln!("QIR_NVFP4_CHECK_MODELS is not set: skipped");
        return Ok(());
    };
    let dm = root.join("diffusion_models");
    let (f4, f8) = (dm.join("krea2_turbo_nvfp4.safetensors"), dm.join("krea2_turbo_fp8_scaled.safetensors"));
    let dev = Device::new_cuda(0)?;
    let vb4 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&f4], DType::BF16, &dev)? };
    let vb8 = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&f8], DType::BF16, &dev)? };
    let mut checked = 0;
    for (name, out_c, in_c) in
        [("blocks.0.attn.wq", 6144, 6144), ("blocks.0.attn.wk", 1536, 6144), ("blocks.10.mlp.up", 0, 6144), ("blocks.20.mlp.down", 6144, 0)]
    {
        let (p4, p8) = (vb4.pp(name), vb8.pp(name));
        if !nvfp4::is_nvfp4(&p4) {
            eprintln!("{name}: not NVFP4 in this file");
            continue;
        }
        let w = p4.get_unchecked_dtype("weight", DType::U8)?;
        let (n, half) = w.dims2()?;
        let (out_c, in_c) = (if out_c == 0 { n } else { out_c }, if in_c == 0 { half * 2 } else { in_c });
        let l4 = nvfp4::load_linear(in_c, out_c, false, &p4)?.expect("an NVFP4 layer");
        let reference = match qwen_image_rs::fp8::load_linear(in_c, out_c, None, &p8)? {
            Some(l) => l.dequantized(DType::F32)?,
            None => p8.get((out_c, in_c), "weight")?.to_dtype(DType::F32)?,
        };
        let mut best = (f32::MAX, Layout::default());
        for linear_scales in [false, true] {
            for low_first in [false, true] {
                let layout = Layout { linear_scales, low_first };
                let e = rel(&l4.dequantized_as(layout)?, &reference);
                eprintln!("{name} ({out_c}, {in_c}) {layout:?}: {:.2} % from the fp8 weight", e * 100.0);
                if e < best.0 {
                    best = (e, layout);
                }
            }
        }
        assert_eq!(best.1, Layout::default(), "{name}: the file is in another layout");
        assert!(best.0 < 0.2, "{name}: {:.1} % from the fp8 weight", best.0 * 100.0);
        if nvfp4::gemm_supported() {
            let x = Tensor::randn(0f32, 1.0, (4096, in_c), &dev)?.to_dtype(DType::BF16)?;
            let want = x.to_dtype(DType::F32)?.matmul(&reference.t()?)?;
            let y = l4.forward(&x)?;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = l4.forward(&x)?;
            }
            dev.synchronize()?;
            let ms4 = t0.elapsed().as_secs_f64() * 100.0;
            let wb = reference.to_dtype(DType::BF16)?;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = x.matmul(&wb.t()?)?;
            }
            dev.synchronize()?;
            let ms16 = t0.elapsed().as_secs_f64() * 100.0;
            eprintln!("{name}: FP4 GEMM {:.2} % from the fp8 weight's matmul; {ms4:.2} ms against bf16 {ms16:.2} ms (M = 4096)", rel(&y, &want) * 100.0);
        }
        checked += 1;
    }
    assert!(checked > 0, "no NVFP4 layer found");
    Ok(())
}
