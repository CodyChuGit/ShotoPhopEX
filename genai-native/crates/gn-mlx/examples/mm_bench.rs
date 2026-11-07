//! The M1's matmul ceiling and the MOSS DiT's linears: f16 dense against 8-bit and 4-bit g64
//! quantized_matmul, and dequantize-then-dense, at the token counts of 512² and 1024².
//! `cargo run --release -p gn-mlx --example mm_bench`
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::{Array, Dtype, ops, random};
    use std::time::Instant;
    let time = |f: &dyn Fn() -> anyhow::Result<Array>| -> anyhow::Result<f64> {
        f()?.eval()?; // warm
        let n = 5;
        let t = Instant::now();
        for _ in 0..n {
            f()?.eval()?;
        }
        Ok(t.elapsed().as_secs_f64() / n as f64)
    };
    let rnd =
        |s: &[i32], dt: Dtype| -> anyhow::Result<Array> { Ok(random::normal::<f32>(s, None, None, None)?.multiply(Array::from_f32(0.02))?.as_dtype(dt)?) };
    let report = |name: &str, m: i32, k: i32, n: i32, secs: f64, weight_bytes: f64| {
        let tf = 2.0 * f64::from(m) * f64::from(k) * f64::from(n) / secs / 1e12;
        println!("{name:<44} {m:>5}x{k:>5}x{n:>5}  {:>8.2} ms  {tf:>5.2} TFLOPS  {:>6.1} GB/s weights", secs * 1e3, weight_bytes / secs / 1e9);
    };
    for dt in [Dtype::Float16, Dtype::Float32] {
        let (a, b) = (rnd(&[4096, 4096], dt)?, rnd(&[4096, 4096], dt)?);
        let s = time(&|| Ok(a.matmul(&b)?))?;
        report(&format!("dense {dt:?} square"), 4096, 4096, 4096, s, 0.0);
    }
    // MOSS DiT linears (in → out): qkv, o, up_gate, down; tokens at 512² (1024 + text) and 1024² (4096 + text).
    for &m in &[1050i32, 4122] {
        for &(name, k, n) in &[("qkv", 3072i32, 9216i32), ("o", 3072, 3072), ("up_gate", 3072, 16384), ("down", 8192, 3072)] {
            let x = rnd(&[m, k], Dtype::Float16)?;
            let w = rnd(&[n, k], Dtype::Float16)?;
            let s = time(&|| Ok(x.matmul(w.t())?))?;
            report(&format!("{name} f16 dense"), m, k, n, s, f64::from(n * k) * 2.0);
            for bits in [8, 4] {
                let (wq, sc, bi) = ops::quantize(&w, 64, bits)?;
                let s = time(&|| Ok(ops::quantized_matmul(&x, &wq, &sc, &bi, true, 64, bits)?))?;
                report(&format!("{name} q{bits} quantized_matmul"), m, k, n, s, f64::from(n * k) * f64::from(bits) / 8.0);
                let s = time(&|| Ok(x.matmul(ops::dequantize(&wq, &sc, &bi, 64, bits)?.as_dtype(Dtype::Float16)?.t())?))?;
                report(&format!("{name} q{bits} dequantize + f16 dense"), m, k, n, s, f64::from(n * k) * f64::from(bits) / 8.0);
                let x32 = x.as_dtype(Dtype::Float32)?;
                let s = time(&|| Ok(ops::quantized_matmul(&x32, &wq, &sc, &bi, true, 64, bits)?))?;
                report(&format!("{name} q{bits} quantized_matmul, f32 input"), m, k, n, s, f64::from(n * k) * f64::from(bits) / 8.0);
            }
        }
    }
    // Attention at MOSS's shapes: 24 heads × 128, bidirectional.
    for &m in &[1050i32, 4122] {
        let q = rnd(&[1, 24, m, 128], Dtype::Float16)?;
        let s = time(&|| Ok(mlx_rs::fast::scaled_dot_product_attention(&q, &q, &q, 1.0 / 128f32.sqrt(), None, None)?))?;
        let flops = 4.0 * 24.0 * f64::from(m) * f64::from(m) * 128.0;
        println!("{:<44} {m:>5} tokens       {:>8.2} ms  {:>5.2} TFLOPS", "sdpa f16 24x128", s * 1e3, flops / s / 1e12);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
