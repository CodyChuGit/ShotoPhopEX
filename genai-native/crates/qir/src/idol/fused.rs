//! Idol's fused CUDA kernels (`kernels/idol/idol_fused.cu`, built with the `flash` feature).
//! The plain candle ops in `dit.rs` and `nn.rs` remain the CPU path and the reference.

use std::ffi::c_void;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

use crate::Result;

/// The head width the fused kernels take.
pub const HEAD: usize = 64;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn qir_idol_qkv_prep(
        out: *mut c_void,
        qkv: *const c_void,
        delta: *const c_void,
        params: *const c_void,
        rows: i64,
        heads: i32,
        n0: i32,
        n1: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// A dense f16/f32 tensor's device pointer, offset to its first element.
fn ptr<T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    s: &CudaStorage,
    l: &Layout,
    what: &str,
    stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> candle_core::Result<u64> {
    if !l.is_contiguous() {
        candle_core::bail!("{what}: must be contiguous, got strides {:?}", l.stride());
    }
    let slice = s.as_cuda_slice::<T>()?;
    let (p, _g) = slice.device_ptr(stream);
    Ok(p + (l.start_offset() * std::mem::size_of::<T>()) as u64)
}

/// RoPE + per-head RMS + gamma + bf16 for q and k, and bf16 for v, in one pass.
struct QkvPrep {
    heads: usize,
    /// Pairs on axis 0 and on axis 1 (the rest are on axis 2).
    n0: usize,
    n1: usize,
}

impl candle_core::CustomOp3 for QkvPrep {
    fn name(&self) -> &'static str {
        "idol-qkv-prep"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("idol-qkv-prep is CUDA-only")
    }

    fn cuda_fwd(&self, qkv: &CudaStorage, qkv_l: &Layout, delta: &CudaStorage, delta_l: &Layout, params: &CudaStorage, params_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dims = qkv_l.shape().dims();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let width = *dims.last().unwrap_or(&0);
        if width != 3 * self.heads * HEAD || delta_l.shape().elem_count() != rows * self.heads * 3 || params_l.shape().elem_count() != 32 + 2 * self.heads * HEAD {
            candle_core::bail!("idol-qkv-prep: qkv {:?}, delta {:?}, params {:?} for {} heads of {HEAD}", qkv_l.shape(), delta_l.shape(), params_l.shape(), self.heads);
        }
        if qkv.dtype() != DType::F16 || delta.dtype() != DType::F16 || params.dtype() != DType::F32 {
            candle_core::bail!("idol-qkv-prep: qkv and delta f16, params f32");
        }
        let dev = qkv.device().clone();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(3 * rows * self.heads * HEAD)? };
        {
            let qp = ptr::<half::f16>(qkv, qkv_l, "qkv", &stream)?;
            let dp = ptr::<half::f16>(delta, delta_l, "delta", &stream)?;
            let pp = ptr::<f32>(params, params_l, "params", &stream)?;
            let (op, _go) = out.device_ptr(&stream);
            let rc = unsafe {
                qir_idol_qkv_prep(op as *mut c_void, qp as *const c_void, dp as *const c_void, pp as *const c_void, rows as i64, self.heads as i32, self.n0 as i32, self.n1 as i32, 1e-6, stream.cu_stream() as *mut c_void)
            };
            if rc != 0 {
                candle_core::bail!("idol-qkv-prep failed, rc={rc}");
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (3, rows, self.heads, HEAD).into()))
    }
}

/// The per-block constants of [`qkv_prep`]: RePo's frequencies `freqs` (axes 0, 1, 2; 32 pairs in
/// all) then the q and k norms' `gamma (H, 64)`, as one f32 tensor; and the pair counts of axes 0
/// and 1.
pub fn qkv_params(freqs: &[Tensor; 3], gamma_q: &Tensor, gamma_k: &Tensor) -> Result<(Tensor, usize, usize)> {
    let flat = |t: &Tensor| -> Result<Tensor> { Ok(t.to_dtype(DType::F32)?.flatten_all()?) };
    let (n0, n1, n2) = (freqs[0].elem_count(), freqs[1].elem_count(), freqs[2].elem_count());
    anyhow::ensure!(n0 + n1 + n2 == HEAD / 2, "RePo has {} frequency pairs, the head {}", n0 + n1 + n2, HEAD / 2);
    let p = Tensor::cat(&[flat(&freqs[0])?, flat(&freqs[1])?, flat(&freqs[2])?, flat(gamma_q)?, flat(gamma_k)?], 0)?;
    Ok((p, n0, n1))
}

/// `qkv (B, L, 3·H·64)` f16 and RePo's offsets `delta (B, L, H·3)` f16 to `(q, k, v)`, each
/// `(B, L, H, 64)` bf16: q and k rotated, RMS-normed per head and times their gamma.
pub fn qkv_prep(qkv: &Tensor, delta: &Tensor, params: &(Tensor, usize, usize), heads: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, l, _) = qkv.dims3()?;
    let out = qkv.contiguous()?.apply_op3_no_bwd(&delta.contiguous()?, &params.0, &QkvPrep { heads, n0: params.1, n1: params.2 })?;
    let part = |i: usize| -> Result<Tensor> { Ok(out.get(i)?.reshape((b, l, heads, HEAD))?) };
    Ok((part(0)?, part(1)?, part(2)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    /// The fused kernel against the plain ops in the model's order (RoPE, then the per-head
    /// RMSNorm and gamma), in f32. Skipped without CUDA.
    #[test]
    fn qkv_prep_matches_the_plain_ops() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (b, l, h) = (2, 37, 4);
        let qkv = (Tensor::randn(0f32, 1.0, (b, l, 3 * h * HEAD), &dev)? * 1.5)?.to_dtype(DType::F16)?;
        let delta = Tensor::randn(0f32, 0.5, (b, l, h * 3), &dev)?.to_dtype(DType::F16)?;
        let freqs = [
            Tensor::arange(0f32, 10.0, &dev)?.affine(0.7, 1.0)?,
            Tensor::arange(0f32, 10.0, &dev)?.affine(0.3, 0.5)?,
            Tensor::arange(0f32, 12.0, &dev)?.affine(1.1, 0.2)?,
        ];
        let gq = (Tensor::randn(0f32, 0.2, (h, HEAD), &dev)? + 1.0)?;
        let gk = (Tensor::randn(0f32, 0.2, (h, HEAD), &dev)? + 1.0)?;
        let params = qkv_params(&freqs, &gq, &gk)?;
        let (q, k, v) = qkv_prep(&qkv, &delta, &params, h)?;

        // Reference, f32.
        let x = qkv.to_dtype(DType::F32)?.reshape((b, l, 3, h, HEAD))?;
        let d = delta.to_dtype(DType::F32)?.reshape((b, l, h, 3))?;
        let ang = Tensor::cat(&(0..3).map(|a| Ok(d.narrow(3, a, 1)?.broadcast_mul(&freqs[a].reshape((1, 1, 1, ()))?)?)).collect::<Result<Vec<_>>>()?, 3)?;
        let ang = (ang * std::f64::consts::PI)?;
        let (cos, sin) = (ang.cos()?, ang.sin()?);
        let rot_norm = |t: &Tensor, g: &Tensor| -> Result<Tensor> {
            let t = t.reshape((b, l, h, HEAD / 2, 2))?;
            let (x0, x1) = (t.narrow(4, 0, 1)?.squeeze(4)?, t.narrow(4, 1, 1)?.squeeze(4)?);
            let r = Tensor::stack(&[(cos.mul(&x0)? - sin.mul(&x1)?)?, (sin.mul(&x0)? + cos.mul(&x1)?)?], 4)?.reshape((b, l, h, HEAD))?;
            let rms = (r.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
            Ok(r.broadcast_div(&rms)?.broadcast_mul(&g.unsqueeze(0)?.unsqueeze(0)?)?)
        };
        let want_q = rot_norm(&x.narrow(2, 0, 1)?.squeeze(2)?, &gq)?;
        let want_k = rot_norm(&x.narrow(2, 1, 1)?.squeeze(2)?, &gk)?;
        let want_v = x.narrow(2, 2, 1)?.squeeze(2)?;
        for (name, got, want) in [("q", &q, &want_q), ("k", &k, &want_k), ("v", &v, &want_v)] {
            let got = got.to_dtype(DType::F32)?;
            let err = (&got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
            eprintln!("qkv_prep {name}: max abs err {err} (max |x| {scale})");
            assert!(err <= scale * 8e-3, "{name}: max abs err {err} against max |x| {scale}");
        }
        Ok(())
    }
}
