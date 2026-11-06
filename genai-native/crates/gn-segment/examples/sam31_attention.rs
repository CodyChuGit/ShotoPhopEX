//! Bound the remaining unfused attention cost at SAM's actual CUDA shapes.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;
fn bench(device: &Device, name: &str, b: usize, h: usize, q: usize, k: usize, d: usize) -> Result<f64> {
    let make =
        |n: usize| -> Result<Tensor> { Ok(Tensor::arange(0_u32, n as u32, device)?.to_dtype(DType::F32)?.affine(0.017, 0.0)?.sin()?.to_dtype(DType::F16)?) };
    let query = make(b * h * q * d)?.reshape((b, h, q, d))?;
    let key = make(b * h * k * d)?.reshape((b, h, k, d))?;
    let value = (&key * 0.3)?;
    for _ in 0..3 {
        let _ = gn_segment::attention::attention(&query, &key, &value, None, false, None, 256)?;
    }
    device.synchronize()?;
    let t = Instant::now();
    for _ in 0..20 {
        let _ = gn_segment::attention::attention(&query, &key, &value, None, false, None, 256)?;
    }
    device.synchronize()?;
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
    println!("{name} B={b} H={h} Q={q} K={k} D={d}: {ms:.4} ms");
    Ok(ms)
}
fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let window = bench(&device, "window", 9, 16, 576, 576, 64)?;
    let global = bench(&device, "global", 1, 16, 5184, 5184, 64)?;
    let detector = bench(&device, "detector self", 1, 8, 5184, 5184, 32)?;
    let _ = bench(&device, "tracker token to image", 1, 8, 8, 5184, 16)?;
    let _ = bench(&device, "tracker image to token", 1, 8, 5184, 8, 16)?;
    println!("two vision passes attention total estimate={:.4} ms; six detector self layers={:.4} ms", 2.0 * (28.0 * window + 4.0 * global), 6.0 * detector);
    Ok(())
}
