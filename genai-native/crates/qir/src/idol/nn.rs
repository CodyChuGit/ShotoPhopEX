//! The small pieces Idol's networks share (ComfyUI `comfy/ldm/triposplat/model.py`): linears,
//! layer norms with and without weights, the per-head RMSNorm, the tanh-GELU MLP and
//! multi-head attention over `(B, L, H, D)`.

use candle_core::{DType, Tensor, D};

use crate::moss::weights::{QLinear, Weights};
use crate::Result;

/// A linear of the file (dense, or ConvRot INT8 where the file has it).
pub struct Lin(QLinear);

impl Lin {
    pub fn load(w: &Weights, module: &str) -> Result<Self> {
        Ok(Self(w.linear(module)?))
    }

    /// `x (..., in)` to `(..., out)`, any number of leading dims.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap_or(&0);
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let y = self.0.forward(&x.reshape((rows, k))?)?;
        let n = y.dim(1)?;
        let mut out = dims[..dims.len() - 1].to_vec();
        out.push(n);
        Ok(y.reshape(out)?)
    }
}

/// Layer norm over the last dim in f32 (rounded back to the input dtype), with optional
/// weight and bias.
pub struct Norm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl Norm {
    /// No learned weights (`elementwise_affine=False`).
    pub fn plain(eps: f64) -> Self {
        Self { weight: None, bias: None, eps }
    }

    /// `<name>.weight` and `<name>.bias` from the file.
    pub fn load(w: &Weights, name: &str, eps: f64) -> Result<Self> {
        let bias = if w.contains(&format!("{name}.bias")) { Some(w.get_as(&format!("{name}.bias"), DType::F32)?) } else { None };
        Ok(Self { weight: Some(w.get_as(&format!("{name}.weight"), DType::F32)?), bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let mut y = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        if let Some(wt) = &self.weight {
            y = y.broadcast_mul(wt)?;
        }
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        Ok(y.to_dtype(dt)?)
    }
}

/// RMSNorm per head (`MultiHeadRMSNorm`): eps 1e-6, no weight, then times `gamma (H, D)`.
pub struct HeadRms {
    gamma: Tensor,
}

impl HeadRms {
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self { gamma: w.get_as(&format!("{name}.gamma"), DType::F32)? })
    }

    /// `x (B, L, H, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let y = x.broadcast_div(&(x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?;
        Ok(y.broadcast_mul(&self.gamma)?.to_dtype(dt)?)
    }
}

/// `Linear, GELU(tanh), Linear` (`MLP`, its weights at `<name>.mlp.0` and `<name>.mlp.2`).
pub struct Mlp {
    fc1: Lin,
    fc2: Lin,
}

impl Mlp {
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self { fc1: Lin::load(w, &format!("{name}.mlp.0"))?, fc2: Lin::load(w, &format!("{name}.mlp.2"))? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

/// Bidirectional attention over `q (B, Lq, H, D)`, `k`, `v (B, Lk, H, D)`; `(B, Lq, H·D)`.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, lq, h, d) = q.dims4()?;
    let lk = k.dim(1)?;
    let scale = 1.0 / (d as f64).sqrt();
    #[cfg(feature = "flash")]
    if q.device().is_cuda() && d <= crate::flash::HEAD_DIM {
        // The fused kernel takes 128-wide bf16 heads: zero-pad q, k and v (the scores and the
        // first `d` output channels are unchanged) and keep the true scale.
        let pad = |t: &Tensor| -> Result<Tensor> {
            let t = t.to_dtype(DType::BF16)?;
            let t = if d < crate::flash::HEAD_DIM { t.pad_with_zeros(3, 0, crate::flash::HEAD_DIM - d)? } else { t };
            Ok(t.transpose(1, 2)?) // (B, H, L, 128) view
        };
        let out = crate::flash::attention(&pad(q)?, &pad(k)?, &pad(v)?, scale as f32)?; // (B, Lq, H, 128)
        return Ok(out.narrow(3, 0, d)?.reshape((b, lq, h * d))?.to_dtype(q.dtype())?);
    }
    const ROWS: usize = 1024;
    let mut items = Vec::with_capacity(b);
    for i in 0..b {
        let qh = q.get(i)?.transpose(0, 1)?.to_dtype(DType::F32)?.contiguous()?; // (H, Lq, D)
        let kt = k.get(i)?.transpose(0, 1)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // (H, D, Lk)
        let vh = v.get(i)?.transpose(0, 1)?.to_dtype(DType::F32)?.contiguous()?; // (H, Lk, D)
        debug_assert_eq!(kt.dim(2)?, lk);
        let mut outs = Vec::with_capacity(lq.div_ceil(ROWS));
        let mut r0 = 0;
        while r0 < lq {
            let n = ROWS.min(lq - r0);
            let scores = (qh.narrow(1, r0, n)?.matmul(&kt)? * scale)?;
            outs.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vh)?);
            r0 += n;
        }
        items.push(Tensor::cat(&outs, 1)?.transpose(0, 1)?.reshape((lq, h * d))?);
    }
    Ok(Tensor::stack(&items, 0)?.to_dtype(q.dtype())?)
}

/// Sinusoidal point embedding (`PcdAbsolutePositionEmbedder`) of `x (N, 3)` in f32: per point
/// and per axis `[sin(f·x·π), cos(f·x·π)]` over `freqs`, zero-padded to `channels`.
pub fn point_embedding(x: &Tensor, freqs: &[f32], channels: usize) -> Result<Tensor> {
    let (n, axes) = x.dims2()?;
    let f = Tensor::from_slice(freqs, (1, freqs.len()), x.device())?;
    let flat = x.to_dtype(DType::F32)?.reshape((n * axes, 1))?;
    let ang = (flat.broadcast_mul(&f)? * std::f64::consts::PI)?;
    let out = Tensor::cat(&[ang.sin()?, ang.cos()?], 1)?.reshape((n, axes * 2 * freqs.len()))?;
    let used = axes * 2 * freqs.len();
    Ok(if used < channels { out.pad_with_zeros(1, 0, channels - used)? } else { out })
}

/// The `pow2` schedule (the flow model's latent anchors): `2^k · 2` for `k < max_res`, then
/// `max_res` more steps spread over the rest.
pub fn pow2_freqs(channels: usize, axes: usize, max_res: usize) -> Vec<f32> {
    let freq_dim = channels / axes / 2;
    let res_dim = freq_dim.saturating_sub(max_res);
    let mut e: Vec<f32> = (0..max_res).map(|k| k as f32).collect();
    e.extend((0..res_dim).map(|k| k as f32 / res_dim.max(1) as f32 * max_res as f32));
    e.truncate(freq_dim);
    e.iter().map(|&v| 2f32.powf(v) * 2.0).collect()
}

/// The `log2` schedule (the octree and gaussian decoders): `2^linspace(0, max_res, n)`.
pub fn log2_freqs(channels: usize, axes: usize, max_res: usize) -> Vec<f32> {
    let n = channels / axes / 2;
    // torch.linspace in f32: from the start for the first half, from the end for the rest.
    let end = max_res as f32;
    let step = if n > 1 { end / (n - 1) as f32 } else { 0.0 };
    (0..n).map(|i| 2f32.powf(if i < n / 2 { step * i as f32 } else { end - step * (n - 1 - i) as f32 })).collect()
}

/// `[cos(t·f), sin(t·f)]` with `f = exp(-ln(max_period)·i/half)` (DiT's timestep embedding).
pub fn sinusoid(t: &[f32], dim: usize, max_period: f64, scale: f64, dev: &candle_core::Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut v = Vec::with_capacity(t.len() * dim);
    // f32 throughout, as torch computes it.
    let neg_log = -(max_period.ln()) as f32;
    let freqs: Vec<f32> = (0..half).map(|i| (neg_log * i as f32 / half as f32).exp()).collect();
    for &ti in t {
        let args: Vec<f32> = freqs.iter().map(|&f| if scale == 1.0 { ti * f } else { ti * f * scale as f32 }).collect();
        v.extend(args.iter().map(|a| a.cos()));
        v.extend(args.iter().map(|a| a.sin()));
        if dim % 2 == 1 {
            v.push(0.0);
        }
    }
    Ok(Tensor::from_vec(v, (t.len(), dim), dev)?)
}
