//! The pieces every model shares: the named weights, linears (dense or 4-bit), slicing and the
//! comparison metrics of the checks.

use std::collections::HashMap;

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::{Ellipsis, TryIndexOp};
use mlx_rs::{Array, Dtype, ops};

/// A model's weights by qir's tensor names: `{m}.weight` (dense, or the packed 4-bit words with
/// `{m}.scales` and `{m}.biases`), `{m}.bias`, norms.
pub type Params = HashMap<String, Array>;

/// MLX's quantization group size.
pub const GROUP: i32 = 64;

pub fn get(p: &Params, name: &str) -> Result<Array> {
    p.get(name).cloned().with_context(|| format!("missing weight {name}"))
}

/// Before the first GPU work: inside an app bundle, MLX's kernels from
/// `Contents/Resources/mlx.metallib` (MLX looks beside the executable, then in the path the
/// build compiled in, under the builder's home); and MLX's cache of freed buffers capped (by
/// default it may grow to the whole memory limit, and on a 16 GB Mac that pushes the system into
/// swap).
pub fn limit_cache() -> Result<()> {
    let bundled = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("../Resources/mlx.metallib"))).filter(|p| p.is_file());
    if let Some(path) = bundled {
        mlx_rs::metal::set_metallib_path(path.to_string_lossy())?;
    }
    mlx_rs::memory::set_cache_limit(1 << 30)?;
    Ok(())
}

/// `usize` to MLX's `i32` dimensions.
pub fn dim(v: usize) -> Result<i32> {
    i32::try_from(v).with_context(|| format!("dimension {v} too large"))
}

pub fn shape(dims: &[usize]) -> Result<Vec<i32>> {
    dims.iter().map(|&d| dim(d)).collect()
}

/// A scalar in `dt`, so it does not promote the array it multiplies.
pub fn scalar(v: f32, dt: Dtype) -> Result<Array> {
    Ok(Array::from_f32(v).as_dtype(dt)?)
}

/// `x[..., a..b]`.
pub fn cols(x: &Array, a: usize, b: usize) -> Result<Array> {
    Ok(x.try_index((Ellipsis, dim(a)?..dim(b)?))?)
}

/// `x[a..b]` along the first axis.
pub fn rows(x: &Array, a: usize, b: usize) -> Result<Array> {
    Ok(x.try_index(dim(a)?..dim(b)?)?)
}

/// The array's values as f32 on the host.
pub fn to_vec(x: &Array) -> Result<Vec<f32>> {
    let x = x.as_dtype(Dtype::Float32)?.contiguous()?;
    x.eval()?;
    Ok(x.try_as_slice::<f32>().map_err(|e| anyhow::anyhow!("read back an array: {e:?}"))?.to_vec())
}

/// A candle tensor (any dtype, any device) as an f32 MLX array of the same shape.
pub fn from_candle(t: &candle_core::Tensor) -> Result<Array> {
    let dims = shape(t.dims())?;
    let v = t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(Array::from_slice(&v, &dims))
}

/// A linear: `x Wᵀ + b`, the weight dense (`(out, in)`) or 4-bit.
pub enum Linear {
    Dense { w: Array, b: Option<Array> },
    Quant { w: Array, scales: Array, biases: Array, b: Option<Array>, bits: i32 },
}

impl Linear {
    pub fn load(p: &Params, m: &str) -> Result<Self> {
        let w = get(p, &format!("{m}.weight"))?;
        let b = p.get(&format!("{m}.bias")).cloned();
        Ok(match (p.get(&format!("{m}.scales")), p.get(&format!("{m}.biases"))) {
            (Some(s), Some(z)) => {
                // The bits from the shapes: `in = groups·GROUP` inputs packed into 32-bit words.
                let k = s.shape().last().copied().unwrap_or(0) * GROUP;
                let words = w.shape().last().copied().unwrap_or(0);
                anyhow::ensure!(k > 0 && (words * 32) % k == 0, "{m}: quantized shapes {:?} and {:?}", w.shape(), s.shape());
                Self::Quant { w, scales: s.clone(), biases: z.clone(), b, bits: words * 32 / k }
            }
            _ => Self::Dense { w, b },
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let (y, b) = match self {
            Self::Dense { w, b } => (x.matmul(w.t())?, b),
            Self::Quant { w, scales, biases, b, bits } => (ops::quantized_matmul(x, w, scales, biases, true, GROUP, *bits)?, b),
        };
        Ok(match b {
            Some(b) => y.add(b)?,
            None => y,
        })
    }
}

/// Cosine similarity and the largest absolute difference of two equally long vectors.
pub fn compare(a: &[f32], b: &[f32]) -> Result<(f64, f64)> {
    anyhow::ensure!(a.len() == b.len() && !a.is_empty(), "comparing {} values with {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut max) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        max = max.max((x - y).abs());
    }
    Ok((dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), max))
}

/// PSNR in dB of two pictures in [-1, 1].
pub fn psnr(a: &[f32], b: &[f32]) -> Result<f64> {
    anyhow::ensure!(a.len() == b.len() && !a.is_empty(), "comparing {} values with {}", a.len(), b.len());
    let mse = a.iter().zip(b).map(|(&x, &y)| f64::from(x - y).powi(2)).sum::<f64>() / a.len() as f64;
    // The range is 2.
    Ok(10.0 * (4.0 / mse.max(1e-20)).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics() -> Result<()> {
        let (c, m) = compare(&[1.0, 0.0], &[1.0, 0.5])?;
        assert!((c - 1.0 / 1.25f64.sqrt()).abs() < 1e-9 && (m - 0.5).abs() < 1e-9);
        assert!(compare(&[], &[]).is_err());
        assert!(psnr(&[0.0; 4], &[0.0; 4])? > 150.0);
        Ok(())
    }
}
