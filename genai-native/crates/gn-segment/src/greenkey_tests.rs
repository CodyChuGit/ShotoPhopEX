use std::collections::HashMap;

use super::*;
use crate::hiera::tiny_config;

fn tiny() -> GreenKeyConfig {
    GreenKeyConfig { hiera: tiny_config(64), decoder_dim: 16, refiner_channels: 16, ..GreenKeyConfig::default() }
}

fn vals(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn a_random_small_keyer_runs_end_to_end() -> Result<()> {
    let dev = Device::Cpu;
    let st = Store::random(&dev);
    let m = GreenKeyer::from_store(&st, tiny(), DType::F32, &dev)?;
    let (w, h) = (40, 24);
    let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 % 256) as u8).collect();
    let hint: Vec<f32> = (0..w * h).map(|i| if i % w < 20 { 1.0 } else { 0.0 }).collect();
    let out = m.key(&rgba, &hint, w, h, 1.0)?;
    assert_eq!((out.alpha.len(), out.fg.len()), (w * h, w * h * 3));
    assert!(out.alpha.iter().chain(&out.fg).all(|v| (0.0..=1.0).contains(v)));
    assert_eq!(to_rgba8(&out).len(), w * h * 4);
    // The raw outputs at the working size.
    let x = Tensor::from_vec(prepare(&rgba, &hint, w, h, 64)?, (1, 4, 64, 64), &dev)?;
    let k = m.forward(&x, 1.0)?;
    assert_eq!((k.alpha.dims(), k.fg.dims()), (&[1, 1, 64, 64][..], &[1, 3, 64, 64][..]));
    // Strength 0 is the coarse answer; the refiner moves it.
    let coarse = m.forward(&x, 0.0)?;
    assert!(max_abs(&vals(&coarse.alpha)?, &vals(&k.alpha)?) > 0.0);
    Ok(())
}

#[test]
fn names_are_found_under_a_wrapper_and_heads_told_apart_by_their_outputs() -> Result<()> {
    // The same weights renamed: a `module.` wrapper, the heads under other names (the colour
    // head first alphabetically), loaded through discovery, give the same answer.
    let dev = Device::Cpu;
    let st = Store::random(&dev);
    let m = GreenKeyer::from_store(&st, tiny(), DType::F32, &dev)?;
    let map: HashMap<String, Tensor> =
        st.recorded().into_iter().map(|(k, v)| (format!("module.{}", k.replace("alpha_decoder.", "zz_head.").replace("fg_decoder.", "aa_head.")), v)).collect();
    let st2 = Store::from_map(map, &dev);
    let m2 = GreenKeyer::from_store(&st2, tiny(), DType::F32, &dev)?;
    let x = Tensor::rand(-1f32, 1.0, (1, 4, 64, 64), &dev)?;
    let (a, b) = (m.forward(&x, 1.0)?, m2.forward(&x, 1.0)?);
    assert!(max_abs(&vals(&a.alpha)?, &vals(&b.alpha)?) < 1e-6);
    assert!(max_abs(&vals(&a.fg)?, &vals(&b.fg)?) < 1e-6);
    Ok(())
}

#[test]
fn group_norm_matches_hand_computed_values() -> Result<()> {
    // 4 channels, 2 groups, 1x2 pixels. Group 0 holds 1, 3, 5, 7 (mean 4, variance 5).
    let x = Tensor::new(&[1f32, 3.0, 5.0, 7.0, 0.0, 0.0, 2.0, 2.0], &Device::Cpu)?.reshape((1, 4, 1, 2))?;
    let gamma = Tensor::new(&[1f32, 2.0, 1.0, 1.0], &Device::Cpu)?;
    let beta = Tensor::new(&[0f32, 1.0, 0.0, 0.5], &Device::Cpu)?;
    let y = vals(&group_norm(&x, &gamma, &beta, 2, 0.0, None, false, DType::F32)?)?;
    let s = 5f32.sqrt();
    let want = [-3.0 / s, -1.0 / s, 2.0 * 1.0 / s + 1.0, 2.0 * 3.0 / s + 1.0, -1.0, -1.0, 1.5, 1.5];
    assert!(max_abs(&y, &want) < 1e-5, "{y:?}");
    // The residual is added after the affine step, then ReLU.
    let r = Tensor::ones((1, 4, 1, 2), DType::F32, &Device::Cpu)?;
    let y = vals(&group_norm(&x, &gamma, &beta, 2, 0.0, Some(&r), true, DType::F32)?)?;
    let want: Vec<f32> = want.iter().map(|v| (v + 1.0).max(0.0)).collect();
    assert!(max_abs(&y, &want) < 1e-5, "{y:?}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// A direct 3x3 convolution at `d` (zero padding `d`), for (Cin, H, W) planes.
fn naive_conv(x: &[f32], cin: usize, h: usize, w: usize, k: &[f32], b: &[f32], cout: usize, d: usize) -> Vec<f32> {
    let mut y = vec![0f32; cout * h * w];
    for o in 0..cout {
        for yy in 0..h {
            for xx in 0..w {
                let mut s = b[o];
                for i in 0..cin {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let sy = yy as isize + (ky as isize - 1) * d as isize;
                            let sx = xx as isize + (kx as isize - 1) * d as isize;
                            if sy >= 0 && sx >= 0 && (sy as usize) < h && (sx as usize) < w {
                                s += k[((o * cin + i) * 3 + ky) * 3 + kx] * x[(i * h + sy as usize) * w + sx as usize];
                            }
                        }
                    }
                }
                y[(o * h + yy) * w + xx] = s;
            }
        }
    }
    y
}

fn naive_gn(x: &[f32], c: usize, hw: usize, groups: usize, g: &[f32], b: &[f32], eps: f64) -> Vec<f32> {
    let cg = c / groups;
    let mut y = vec![0f32; x.len()];
    for gi in 0..groups {
        let s = &x[gi * cg * hw..(gi + 1) * cg * hw];
        let mean = s.iter().map(|v| *v as f64).sum::<f64>() / s.len() as f64;
        let var = s.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / s.len() as f64;
        for ci in 0..cg {
            let ch = gi * cg + ci;
            for p in 0..hw {
                y[ch * hw + p] = (((x[ch * hw + p] as f64 - mean) / (var + eps).sqrt()) as f32) * g[ch] + b[ch];
            }
        }
    }
    y
}

#[test]
fn the_dilated_refiner_matches_a_direct_computation() -> Result<()> {
    let dev = Device::Cpu;
    let st = Store::random(&dev);
    let c = GreenKeyConfig { refiner_channels: 8, refiner_groups: 4, ..tiny() };
    let r = Refiner::load(&st, "refiner.", &c, DType::F32)?;
    let rec = st.recorded();
    let get = |n: &str| -> Result<Vec<f32>> { vals(rec.get(n).with_context(|| n.to_string())?) };
    let (h, w, ch) = (21, 19, 8);
    let x = Tensor::rand(-1f32, 1.0, (1, 7, h, w), &dev)?;
    let got = vals(&r.forward(&x)?)?;

    let relu = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|a| a.max(0.0)).collect() };
    let conv = |x: &[f32], cin: usize, p: &str, d: usize| -> Result<Vec<f32>> {
        Ok(naive_conv(x, cin, h, w, &get(&format!("{p}.weight"))?, &get(&format!("{p}.bias"))?, ch, d))
    };
    let gn = |x: &[f32], p: &str| -> Result<Vec<f32>> { Ok(naive_gn(x, ch, h * w, 4, &get(&format!("{p}.weight"))?, &get(&format!("{p}.bias"))?, 1e-5)) };
    let mut hcur = relu(gn(&conv(&vals(&x)?, 7, "refiner.stem.0", 1)?, "refiner.stem.1")?);
    for (i, d) in [1, 2, 4, 8].into_iter().enumerate() {
        let p = format!("refiner.res{}", i + 1);
        let y = relu(gn(&conv(&hcur, ch, &format!("{p}.conv1"), d)?, &format!("{p}.gn1"))?);
        let y = gn(&conv(&y, ch, &format!("{p}.conv2"), d)?, &format!("{p}.gn2"))?;
        hcur = relu(y.iter().zip(&hcur).map(|(a, b)| a + b).collect());
    }
    let (fw, fb) = (get("refiner.final.weight")?, get("refiner.final.bias")?);
    let mut want = vec![0f32; 4 * h * w];
    for o in 0..4 {
        for p in 0..h * w {
            let s: f32 = (0..ch).map(|i| fw[o * ch + i] * hcur[i * h * w + p]).sum();
            want[o * h * w + p] = (s + fb[o]) * 10.0;
        }
    }
    let m = max_abs(&got, &want);
    assert!(m < 1e-3, "max abs {m}");
    Ok(())
}

#[test]
fn strip_convolution_equals_one_convolution() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::rand(-1f32, 1.0, (1, 3, 37, 23), &dev)?;
    let k = Tensor::rand(-1f32, 1.0, (5, 3, 3, 3), &dev)?;
    let b = Tensor::rand(-1f32, 1.0, 5, &dev)?;
    for d in [1, 2, 4, 8] {
        let whole = x.conv2d(&k, d, 1, d, 1)?.broadcast_add(&b.reshape((1, 5, 1, 1))?)?;
        // A budget of 3 rows' im2col forces 13 strips.
        let strips = conv3_strips(&x, &k, Some(&b), d, 23 * 3 * 9 * 3)?;
        assert_eq!(strips.dims(), whole.dims());
        assert!(max_abs(&vals(&whole)?, &vals(&strips)?) < 1e-5, "dilation {d}");
    }
    Ok(())
}

#[test]
fn lanczos_keeps_constants_and_identity() {
    let src = vec![0.25f32; 30 * 20];
    for (dw, dh) in [(7, 5), (61, 47), (30, 20)] {
        assert!(lanczos3(&src, 30, 20, dw, dh).iter().all(|v| (v - 0.25).abs() < 1e-5));
    }
    let ramp: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(lanczos3(&ramp, 4, 3, 4, 3), ramp);
    // Upsampling a ramp 2x stays close to it (inside, away from the clamped border; Lanczos
    // is not exactly linear) and increasing.
    let r: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let up = lanczos3(&r, 16, 1, 32, 1);
    assert!((up[15] - 7.25).abs() < 0.03 && (up[16] - 7.75).abs() < 0.03, "{up:?}");
    assert!(up.windows(2).all(|p| p[1] > p[0]), "{up:?}");
}

#[test]
fn input_is_normalised_rgb_and_a_raw_hint() -> Result<()> {
    let rgba = [255u8, 0, 128, 7];
    let x = prepare(&rgba, &[0.75], 1, 1, 2)?;
    assert_eq!(x.len(), 16);
    let want = [(1.0 - MEAN[0]) / STD[0], (0.0 - MEAN[1]) / STD[1], (128.0 / 255.0 - MEAN[2]) / STD[2], 0.75];
    for c in 0..4 {
        assert!(x[c * 4..c * 4 + 4].iter().all(|v| (v - want[c]).abs() < 1e-6), "{x:?}");
    }
    assert!(prepare(&rgba, &[0.5, 0.5], 1, 1, 2).is_err());
    Ok(())
}

/// Parity with reference outputs, skipped unless `GREENKEY_WEIGHTS` names the checkpoint and
/// `GREENKEY_REF_DIR` holds `input_tensor.npy` ((1, 4, S, S) model input, f32) and the
/// reference's raw outputs at S x S: `alpha_full_fp32.npy` / `fg_full_fp32.npy` (refiner
/// strength 1) and `alpha_norefiner_fp32.npy` / `fg_norefiner_fp32.npy` (strength 0), each
/// pair optional. With `input.png` and `hint.png` there too, the preprocessing is checked
/// against `input_tensor.npy`. `GREENKEY_DTYPE` (f32, f16, bf16) picks the compute dtype.
#[test]
fn parity_with_reference_outputs() -> Result<()> {
    let (Ok(weights), Ok(dir)) = (std::env::var("GREENKEY_WEIGHTS"), std::env::var("GREENKEY_REF_DIR")) else {
        eprintln!("greenkey parity: skipped (GREENKEY_WEIGHTS / GREENKEY_REF_DIR unset)");
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    let dev = Device::Cpu;
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0).unwrap_or(dev);
    let dtype = match std::env::var("GREENKEY_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let x = Tensor::read_npy(dir.join("input_tensor.npy"))?.to_dtype(DType::F32)?;
    let report = |name: &str, got: &[f32], want: &[f32]| -> Result<(f32, f32)> {
        ensure!(got.len() == want.len(), "{name}: {} values against the reference's {}", got.len(), want.len());
        let mean = (got.iter().zip(want).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / got.len() as f64) as f32;
        let max = max_abs(got, want);
        eprintln!("greenkey parity {name}: mean abs {mean:.6} ({:.4}/255), max abs {max:.6} ({:.2}/255)", mean * 255.0, max * 255.0);
        Ok((mean, max))
    };
    let (png, hint) = (dir.join("input.png"), dir.join("hint.png"));
    if png.is_file() && hint.is_file() {
        let img = image::open(&png)?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let hint: Vec<f32> = image::open(&hint)?.to_luma8().pixels().map(|p| f32::from(p.0[0]) / 255.0).collect();
        let s = x.dim(2)?;
        report("preprocess", &prepare(img.as_raw(), &hint, w, h, s)?, &vals(&x)?)?;
    }
    let m = GreenKeyer::load(&weights, &dev, dtype)?;
    let mut failures = Vec::new();
    for (tag, strength) in [("norefiner", 0.0f32), ("full", 1.0)] {
        let (pa, pf) = (dir.join(format!("alpha_{tag}_fp32.npy")), dir.join(format!("fg_{tag}_fp32.npy")));
        if !pa.is_file() || !pf.is_file() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let out = m.forward(&x, strength)?;
        eprintln!("greenkey parity {tag}: forward in {:.1} s ({dtype:?})", t0.elapsed().as_secs_f32());
        for (name, got, path) in [("alpha", &out.alpha, pa), ("fg", &out.fg, pf)] {
            let (mean, _) = report(&format!("{name} {tag}"), &vals(got)?, &vals(&Tensor::read_npy(path)?)?)?;
            if mean >= 2.0 / 255.0 {
                failures.push(format!("{name} {tag}: mean abs {mean}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
    Ok(())
}

#[test]
fn half_precision_runs_and_stays_close_to_f32() -> Result<()> {
    // The GPU path's dtype plumbing (f16 compute, f32 norms and sigmoids), on the CPU.
    let dev = Device::Cpu;
    let st = Store::random(&dev);
    let m = GreenKeyer::from_store(&st, tiny(), DType::F32, &dev)?;
    let st16 = Store::from_map(st.recorded(), &dev);
    let x = Tensor::rand(-1f32, 1.0, (1, 4, 64, 64), &dev)?;
    let a = m.forward(&x, 1.0)?;
    // (candle's CPU backend has no bf16 matmul; bf16 runs on CUDA only.)
    let dtype = DType::F16;
    let h = GreenKeyer::from_store(&st16, tiny(), dtype, &dev)?.forward(&x, 1.0)?;
    assert_eq!(h.alpha.dtype(), DType::F32);
    let (da, df) = (max_abs(&vals(&a.alpha)?, &vals(&h.alpha)?), max_abs(&vals(&a.fg)?, &vals(&h.fg)?));
    assert!(da < 0.05 && df < 0.05, "{dtype:?}: alpha {da}, fg {df}");
    Ok(())
}
