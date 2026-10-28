//! Bounded-memory CPU reference and batched CUDA attention for SAM's head sizes.

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

/// Attend with `[B,H,Q,D]` queries and `[B,H,K,D]` keys/values; CUDA preserves the input dtype.
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
    if q.device().is_cuda() {
        return cuda_attention(q, k, v, padding, causal, bias);
    }
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

/// Batch heads together and bound score storage; softmax and masking stay in f32.
fn cuda_attention(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<&Tensor>, causal: bool, bias: Option<&Tensor>) -> Result<Tensor> {
    let (b, h, nq, d) = q.dims4()?;
    let nk = k.dim(2)?;
    let split_scale = d != 64;
    let scale = (d as f64).powf(-0.25);
    let q = if split_scale { (q * scale)? } else { q.clone() };
    let k = if split_scale { (k * scale)? } else { k.clone() };
    let kt = k.transpose(2, 3)?.contiguous()?;
    if padding.is_none() && bias.is_none() && !causal {
        let mut outputs = Vec::new();
        for start in (0..nq).step_by(1024) {
            let len = 1024.min(nq - start);
            let scores = q.narrow(2, start, len)?.contiguous()?.matmul(&kt)?;
            let scores = if split_scale { scores } else { (scores / (d as f64).sqrt())? };
            // Candle's half softmax kernel accumulates its exponential sum in f32.
            outputs.push(candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?.matmul(v)?);
        }
        return Ok(Tensor::cat(&outputs, 2)?);
    }
    let mut outputs = Vec::new();
    for start in (0..nq).step_by(512) {
        let len = 512.min(nq - start);
        let mut scores = q.narrow(2, start, len)?.contiguous()?.matmul(&kt)?;
        if !split_scale {
            scores = (scores / (d as f64).sqrt())?;
        }
        if let Some(bias) = bias {
            scores = scores.broadcast_add(&bias.to_dtype(scores.dtype())?.broadcast_as((b, h, nq, nk))?.narrow(2, start, len)?)?;
        }
        scores = scores.to_dtype(DType::F32)?;
        if let Some(padding) = padding {
            ensure!(padding.dims() == [b, nk], "padding mask must be [B,K]");
            let valid = padding.to_dtype(DType::U8)?.reshape((b, 1, 1, nk))?.broadcast_as(scores.shape())?;
            scores = valid.where_cond(&scores, &Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?)?;
        }
        if causal {
            let valid: Vec<u8> = (start..start + len).flat_map(|i| (0..nk).map(move |j| u8::from(j <= i))).collect();
            let valid = Tensor::from_vec(valid, (1, 1, len, nk), q.device())?.broadcast_as(scores.shape())?;
            scores = valid.where_cond(&scores, &Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?)?;
        }
        let valid = scores.max_keepdim(3)?.ne(f32::NEG_INFINITY)?;
        let safe = valid.broadcast_as(scores.shape())?.where_cond(&scores, &scores.zeros_like()?)?;
        let prob = candle_nn::ops::softmax_last_dim(&safe)?;
        let prob = valid.broadcast_as(prob.shape())?.where_cond(&prob, &prob.zeros_like()?)?;
        outputs.push(prob.to_dtype(v.dtype())?.matmul(v)?);
    }
    Ok(Tensor::cat(&outputs, 2)?)
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn cuda_head_sizes_masks_and_chunk_tails() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for d in [16, 32, 64] {
            let q: Vec<f32> = (0..2 * 1037 * d).map(|i| (i as f32 * 0.017).sin() * 0.25).collect();
            let k: Vec<f32> = (0..2 * 43 * d).map(|i| (i as f32 * 0.031).cos() * 0.25).collect();
            let q = Tensor::from_vec(q, (1, 2, 1037, d), &device)?.to_dtype(DType::F16)?;
            let k = Tensor::from_vec(k, (1, 2, 43, d), &device)?.to_dtype(DType::F16)?;
            let v = (&k * 0.7)?;
            for masked in [false, true] {
                let padding = Tensor::from_vec((0..43).map(|i| u8::from(i % 3 != 0)).collect::<Vec<_>>(), (1, 43), &device)?;
                let p = if masked { Some(&padding) } else { None };
                let actual = attention(&q, &k, &v, p, false, None, 128)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let cpu = |t: &Tensor| -> Result<Tensor> { Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?) };
                let padding_cpu = padding.to_device(&Device::Cpu)?;
                let expected = attention(&cpu(&q)?, &cpu(&k)?, &cpu(&v)?, if masked { Some(&padding_cpu) } else { None }, false, None, 128)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let error = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
                assert!(error < 0.005, "head dim {d}, mask {masked}, error {error}");
            }
            let padding = Tensor::zeros((1, 43), DType::U8, &device)?;
            let actual = attention(&q, &k, &v, Some(&padding), false, None, 128)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(actual.iter().all(|v| *v == 0.0));
        }
        Ok(())
    }
}
