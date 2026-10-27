//! Bounded-memory f32 scaled dot-product attention for SAM's head sizes.

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

/// Attend with `[B,H,Q,D]` queries and `[B,H,K,D]` keys/values in f32.
/// Padding mask `[B,K]` uses one for valid keys and zero for padding. Bias must
/// broadcast to `[B,H,Q,K]`; causal masking uses `key_index <= query_index`.
/// Heads are processed separately and queries in chunks to bound global score storage.
/// Fully masked rows return zero, matching PyTorch SDPA rather than producing NaNs.
/// Such rows are replaced before softmax and zeroed afterward on the same device.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<&Tensor>, causal: bool, bias: Option<&Tensor>, chunk: usize) -> Result<Tensor> {
    let (b, h, nq, d) = q.dims4()?;
    let (bk, hk, nk, dk) = k.dims4()?;
    ensure!(b > 0 && h > 0 && nq > 0 && nk > 0 && d > 0 && chunk > 0, "attention dimensions and chunk must be positive");
    ensure!((b, h, d) == (bk, hk, dk) && v.dims() == [b, h, nk, d], "attention shapes disagree");
    let (q, k, v) = (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);
    let padding = padding
        .map(|p| -> Result<Tensor> {
            ensure!(p.dims() == [b, nk], "padding mask must be [B,K]");
            Ok(p.to_dtype(DType::U8)?.reshape((b, 1, 1, nk))?)
        })
        .transpose()?;
    let bias = bias.map(|t| t.to_dtype(DType::F32)?.broadcast_as((b, h, nq, nk))).transpose()?;
    let mut heads = Vec::with_capacity(h);
    for head in 0..h {
        let kh = k.narrow(1, head, 1)?.squeeze(1)?.transpose(1, 2)?.contiguous()?;
        let vh = v.narrow(1, head, 1)?.squeeze(1)?.contiguous()?;
        let mut chunks = Vec::new();
        for start in (0..nq).step_by(chunk) {
            let len = chunk.min(nq - start);
            let qh = q.narrow(1, head, 1)?.squeeze(1)?.narrow(1, start, len)?.contiguous()?;
            let mut scores = (qh.matmul(&kh)? / (d as f64).sqrt())?;
            if let Some(bias) = &bias {
                scores = scores.broadcast_add(&bias.narrow(1, head, 1)?.squeeze(1)?.narrow(1, start, len)?)?;
            }
            if let Some(p) = &padding {
                let valid = p.squeeze(1)?.broadcast_as((b, len, nk))?;
                scores = valid.where_cond(&scores, &Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?)?;
            }
            if causal {
                let valid: Vec<u8> = (start..start + len).flat_map(|i| (0..nk).map(move |j| u8::from(j <= i))).collect();
                let valid = Tensor::from_vec(valid, (1, len, nk), q.device())?.broadcast_as((b, len, nk))?;
                scores = valid.where_cond(&scores, &Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?)?;
            }
            let valid = scores.max_keepdim(2)?.ne(f32::NEG_INFINITY)?;
            let safe = valid.broadcast_as(scores.shape())?.where_cond(&scores, &scores.zeros_like()?)?;
            let prob = candle_nn::ops::softmax_last_dim(&safe)?;
            let prob = valid.broadcast_as(prob.shape())?.where_cond(&prob, &prob.zeros_like()?)?;
            chunks.push(prob.matmul(&vh)?);
        }
        heads.push(Tensor::cat(&chunks, 1)?.unsqueeze(1)?);
    }
    Ok(Tensor::cat(&heads, 1)?)
}
