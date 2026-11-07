//! One decode token's matmuls (a row against a quantized weight) at 4, 6 and 8 bits, at
//! Qwen3.5-9B's shapes: GB/s of weight read against the M1's 68.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::{Array, Dtype, ops, random};
    use std::time::Instant;
    for &(name, k, n) in &[("mlp gate/up", 4096i32, 12288i32), ("mlp down", 12288, 4096), ("lm_head", 4096, 248320)] {
        let x = random::normal::<f32>(&[1, k], None, None, None)?.as_dtype(Dtype::Float16)?;
        let w = random::normal::<f32>(&[n, k], None, None, None)?.multiply(Array::from_f32(0.02))?.as_dtype(Dtype::Float16)?;
        for bits in [4, 6, 8] {
            let (wq, sc, bi) = ops::quantize(&w, 64, bits)?;
            mlx_rs::transforms::eval([&wq, &sc, &bi])?;
            let f = || -> anyhow::Result<Array> { Ok(ops::quantized_matmul(&x, &wq, &sc, &bi, true, 64, bits)?) };
            f()?.eval()?;
            let reps = 50;
            let t = Instant::now();
            for _ in 0..reps {
                f()?.eval()?;
            }
            let s = t.elapsed().as_secs_f64() / f64::from(reps);
            let bytes = f64::from(n) * f64::from(k) * f64::from(bits) / 8.0;
            println!("{name:<12} {k:>6}x{n:<7} q{bits}: {:>7.3} ms  {:>5.1} GB/s", s * 1e3, bytes / s / 1e9);
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
