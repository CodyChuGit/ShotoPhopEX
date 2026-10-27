//! Exact bf16 attention in one pass (kernels/flash/flash_fwd.cu, the FlashAttention-2
//! algorithm): fp32 scores and softmax, no INT8 or FP8 operands. Krea 2's 8-step Turbo is
//! sensitive to the quantized attention of SageAttention (17 of 28 pictures at ComfyUI's
//! parity gate against 21 with exact attention, `benchmarks.md`), so its DiT blocks use this.
//! Only compiled under the `flash` feature (which implies `cuda`).

use std::ffi::c_void;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

use crate::Result;

/// The head dim the kernel takes.
pub const HEAD_DIM: usize = 128;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn qir_flash_fwd(o: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, b: i32, h: i32, kvh: i32, s: i32, skv: i32, head_dim: i32, strides: *const std::ffi::c_long, scale: f32, stream: *mut c_void) -> i32;
}

/// A `(B, heads, rows, 128)` view's batch, head and row strides and its start, in elements:
/// any view whose rows are dense (a BHSD tensor, or BSHD seen through `transpose(1, 2)`).
fn bhs_strides(l: &Layout, what: &str) -> candle_core::Result<([std::ffi::c_long; 3], usize)> {
    let st = l.stride();
    if st.len() != 4 || st[3] != 1 {
        candle_core::bail!("{what}: rows must be dense, got strides {st:?}");
    }
    Ok(([st[0] as std::ffi::c_long, st[1] as std::ffi::c_long, st[2] as std::ffi::c_long], l.start_offset()))
}

struct Flash {
    scale: f32,
}

impl candle_core::CustomOp3 for Flash {
    fn name(&self) -> &'static str {
        "flash-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("flash-attention is CUDA-only")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout, k: &CudaStorage, k_l: &Layout, v: &CudaStorage, v_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (b, h, s, d) = q_l.shape().dims4()?;
        let (kb, kvh, skv, kd) = k_l.shape().dims4()?;
        if v_l.shape().dims4()? != (kb, kvh, skv, kd) || kb != b || kd != d || d != HEAD_DIM || kvh == 0 || h % kvh != 0 {
            candle_core::bail!("flash-attention: q {:?}, k {:?}, v {:?} (head dim {HEAD_DIM}, H a multiple of KVH)", q_l.shape(), k_l.shape(), v_l.shape());
        }
        let (qst, qo) = bhs_strides(q_l, "flash q")?;
        let (kst, ko) = bhs_strides(k_l, "flash k")?;
        let (vst, vo) = bhs_strides(v_l, "flash v")?;
        let strides = [qst[0], qst[1], qst[2], kst[0], kst[1], kst[2], vst[0], vst[1], vst[2]];
        let (qo, ko, vo) = ((qo * 2) as u64, (ko * 2) as u64, (vo * 2) as u64);
        let (qs, ks, vs) = (q.as_cuda_slice::<half::bf16>()?, k.as_cuda_slice::<half::bf16>()?, v.as_cuda_slice::<half::bf16>()?);
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(b * s * h * d)? };
        {
            let (qp, _gq) = qs.device_ptr(&stream);
            let (kp, _gk) = ks.device_ptr(&stream);
            let (vp, _gv) = vs.device_ptr(&stream);
            let (op, _go) = out.device_ptr(&stream);
            let rc = unsafe {
                qir_flash_fwd(
                    op as *mut c_void,
                    (qp + qo) as *const c_void,
                    (kp + ko) as *const c_void,
                    (vp + vo) as *const c_void,
                    b as i32,
                    h as i32,
                    kvh as i32,
                    s as i32,
                    skv as i32,
                    d as i32,
                    strides.as_ptr(),
                    self.scale,
                    stream.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                candle_core::bail!("flash-attention failed, rc={rc}");
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (b, s, h, d).into()))
    }
}

/// `softmax(q kᵀ · scale) v` for `q (B, H, S, 128)`, `k`, `v (B, KVH, Skv, 128)` in bf16 (KV
/// heads shared by `H / KVH` query heads each); the result is `(B, S, H, 128)`. The inputs may
/// be views with dense rows (the BSHD projections through `transpose(1, 2)`): no copy is made.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    anyhow::ensure!(q.dtype() == DType::BF16 && k.dtype() == DType::BF16 && v.dtype() == DType::BF16, "flash-attention takes bf16");
    let dense_rows = |t: &Tensor| -> Result<Tensor> { Ok(if t.stride().last() == Some(&1) { t.clone() } else { t.contiguous()? }) };
    Ok(dense_rows(q)?.apply_op3_no_bwd(&dense_rows(k)?, &dense_rows(v)?, &Flash { scale })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The kernel against the plain f32 attention, on a shape with a ragged tail and grouped
    /// heads. Skipped without a CUDA device.
    #[test]
    fn flash_attention_is_the_plain_attention() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (b, h, kvh, s, d) = (1, 6, 2, 333, HEAD_DIM);
        // q as the projections leave it, BSHD, seen as BHSD through a transpose (strided rows).
        let q = (Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)? * 2.0)?.to_dtype(DType::BF16)?.transpose(1, 2)?;
        let k = (Tensor::randn(0f32, 1.0, (b, kvh, s, d), &dev)? * 2.0)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (b, kvh, s, d), &dev)?.to_dtype(DType::BF16)?;
        let scale = 1.0 / (d as f32).sqrt();
        let got = attention(&q, &k, &v, scale)?.to_dtype(DType::F32)?; // (B,S,H,D)
        let rep = |t: &Tensor| -> Result<Tensor> { Ok(t.unsqueeze(2)?.broadcast_as((b, kvh, h / kvh, s, d))?.reshape((b, h, s, d))?.to_dtype(DType::F32)?) };
        let (qf, kf, vf) = (q.contiguous()?.to_dtype(DType::F32)?, rep(&k)?, rep(&v)?);
        let p = candle_nn::ops::softmax_last_dim(&(qf.matmul(&kf.t()?)? * f64::from(scale))?)?;
        let want = p.matmul(&vf)?.transpose(1, 2)?; // (B,S,H,D)
        let err = (got - want)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("flash attention: max abs error {err} against the f32 reference");
        assert!(err < 0.02, "max abs error {err}");
        Ok(())
    }
}
