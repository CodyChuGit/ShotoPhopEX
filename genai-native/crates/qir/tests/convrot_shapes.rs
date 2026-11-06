//! ConvRot INT8 linears at Qwen-Image-Layered's shapes against the bf16 matmul (an `#[ignore]`d
//! GPU benchmark): the fused rotate+quantize of the activations, the GEMM with its dequantizing
//! epilogue, and the two together, at M = 8272 (one 1 MP image's tokens) and M = 40000 (the
//! 3-layer split's ~20k tokens in both CFG lanes).
//! `cargo test --release --features cuda,convrot --test convrot_shapes -- --ignored --nocapture`

/// Every compiled tile config (and each big one with the 64x64 tail split) at the shapes the
/// worker runs on the RTX 5090: Layered at M = 40000 and 8272 (K/N 3072/3072, 3072/12288,
/// 12288/3072), MOSS-Image at M = 8272 (N/K 9216/3072, 3072/3072, 16384/3072, 3072/8192) and
/// Qwen-Image-2.1 (INNER 4096) at text to image (~3900 rows) and with a reference (~7800).
#[cfg(all(feature = "cuda", feature = "convrot"))]
#[test]
#[ignore = "exclusive GPU benchmark"]
fn gemm_plans_on_this_card() -> qwen_image_rs::Result<()> {
    use qwen_image_rs::convrot::{bench_gemm_plans, gemm_plan, EpilogueOut, GemmPlan, GEMM_CONFIGS};
    let shapes = [
        (40000usize, 3072usize, 3072usize),
        (40000, 12288, 3072),
        (40000, 3072, 12288),
        (8272, 3072, 3072),
        (8272, 12288, 3072),
        (8272, 3072, 12288),
        (8272, 9216, 3072),
        (8272, 16384, 3072),
        (8272, 3072, 8192),
        (3988, 4096, 4096),
        (3988, 8192, 4096),
        (3988, 24576, 4096),
        (3988, 4096, 12288),
        (7876, 4096, 4096),
        (7876, 8192, 4096),
        (7876, 24576, 4096),
        (7876, 4096, 12288),
    ];
    let mut plans: Vec<GemmPlan> = (0..GEMM_CONFIGS.len() as u8).map(GemmPlan::single).collect();
    for main in [0u8, 1, 3, 4] {
        plans.push(GemmPlan { main, tail: Some(2) });
    }
    let label = |p: &GemmPlan| match p.tail {
        None => format!("{}", p.main),
        Some(t) => format!("{}+{t}", p.main),
    };
    let mut all_same = true;
    for (m, n, k, times, same) in bench_gemm_plans(&shapes, &plans, EpilogueOut::Bf16, 20)? {
        let ops = 2.0 * (m * n * k) as f64;
        let best = times.iter().zip(&plans).filter_map(|(t, p)| t.map(|t| (t, p))).min_by(|a, b| a.0.total_cmp(&b.0));
        let sel = gemm_plan(m, n, k);
        let sel_t = times.iter().zip(&plans).find(|(_, p)| **p == sel).and_then(|(t, _)| *t);
        let cells: Vec<String> = times.iter().zip(&plans).map(|(t, p)| match t {
            Some(ms) => format!("{}:{:.0}", label(p), ops / ms / 1e9),
            None => format!("{}:n/a", label(p)),
        }).collect();
        println!(
            "M={m} N={n} K={k} selected {} {:.0} TOPS | best {} {:.0} TOPS | {} | bit-identical={same}",
            label(&sel),
            sel_t.map_or(0.0, |t| ops / t / 1e9),
            best.map_or("-".into(), |b| label(b.1)),
            best.map_or(0.0, |b| ops / b.0 / 1e9),
            cells.join(" ")
        );
        all_same &= same && times.iter().all(Option::is_some);
    }
    assert!(all_same, "plans disagree or failed to launch");
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "convrot"))]
#[test]
#[ignore = "exclusive GPU benchmark"]
fn convrot_at_layered_shapes() -> qwen_image_rs::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use qwen_image_rs::convrot::{ConvRotLinear, EpilogueOut};
    use std::time::Instant;

    let dev = Device::new_cuda(0)?;
    let r = qwen_image_rs::model::rotation::regular_hadamard_256(&dev)?;
    let iters = 20;
    let time = |f: &mut dyn FnMut() -> qwen_image_rs::Result<()>| -> qwen_image_rs::Result<f64> {
        f()?;
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64 * 1e3)
    };
    println!("    M      K      N | quant ms  gemm ms (TOPS) | int8 total ms (TOPS) | bf16 ms (TFLOPS) | bf16/int8");
    for m in [8272usize, 40000] {
        for (k, n) in [(3072usize, 3072usize), (3072, 12288), (12288, 3072)] {
            let w = (Tensor::randn(0f32, 1.0, (n, k), &dev)? * 0.02)?.to_dtype(DType::BF16)?;
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::BF16)?;
            let lin = ConvRotLinear::from_weight(&w, &r)?;
            let qa = lin.quantize(&x)?;
            let quant = time(&mut || {
                lin.quantize(&x)?;
                Ok(())
            })?;
            let gemm = time(&mut || {
                lin.forward_quantized(&qa, EpilogueOut::Bf16)?;
                Ok(())
            })?;
            let total = time(&mut || {
                lin.forward(&x)?;
                Ok(())
            })?;
            let wt = w.t()?.contiguous()?;
            let bf16 = time(&mut || {
                x.matmul(&wt)?;
                Ok(())
            })?;
            let ops = 2.0 * (m * k * n) as f64;
            let rate = |ms: f64| ops / (ms / 1e3) / 1e12;
            println!(
                "{m:>5} {k:>6} {n:>6} | {quant:>8.3} {gemm:>8.3} ({:>4.0}) | {total:>13.3} ({:>4.0}) | {bf16:>7.3} ({:>4.0}) | {:>8.2}x",
                rate(gemm),
                rate(total),
                rate(bf16),
                bf16 / total
            );
        }
    }
    Ok(())
}
