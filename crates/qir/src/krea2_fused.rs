//! Krea 2's small ops fused (kernels/fusednorm/krea2_fused.cu, feature `fusednorm`): the
//! modulated RMSNorm, the per-head q/k RMSNorm with RoPE, SwiGLU's `silu(g)·u`, the attention's
//! sigmoid gate and the gated residual. Each replaces a chain of candle ops (a pass over the
//! activations apiece, f32 copies for the norms) with one pass, f32 math, one bf16 rounding.

use std::ffi::{c_long, c_void};

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

use crate::Result;
use crate::layout::dense_byte_offset;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn qir_k2_rms_mod(y: *mut c_void, x: *const c_void, w: *const f32, scale: *const c_void, shift: *const c_void, rows: i32, c: i32, rows_per_batch: i32, mod_ld: c_long, eps: f32, stream: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn qir_k2_head_rope(y: *mut c_void, x: *const c_void, w: *const f32, cs: *const f32, sn: *const f32, rows: c_long, heads: i32, s: i32, eps: f32, stream: *mut c_void) -> i32;
    fn qir_k2_swiglu(y: *mut c_void, g: *const c_void, u: *const c_void, n: usize, stream: *mut c_void) -> i32;
    fn qir_k2_sig_mul(y: *mut c_void, a: *const c_void, g: *const c_void, n: usize, stream: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn qir_k2_gate_add(y: *mut c_void, x: *const c_void, v: *const c_void, gate: *const c_void, n: usize, c: i32, rows_per_batch: c_long, mod_ld: c_long, stream: *mut c_void) -> i32;
}

fn check(rc: i32, what: &str) -> candle_core::Result<()> {
    if rc != 0 {
        candle_core::bail!("{what} failed, rc={rc}");
    }
    Ok(())
}

fn no_cpu() -> candle_core::Result<(CpuStorage, Shape)> {
    candle_core::bail!("the Krea 2 fused ops are CUDA-only")
}

/// Pointer of a dense bf16 / f32 view, its storage's start plus the view's offset.
macro_rules! ptr {
    ($s:expr, $l:expr, $t:ty, $stream:expr, $guard:ident, $what:expr) => {{
        let off = dense_byte_offset::<$t>($l, $what)?;
        let slice = $s.as_cuda_slice::<$t>()?;
        let (p, g) = slice.device_ptr($stream);
        $guard.push(g);
        p + off
    }};
}

/// `rms(x) · w`, then `· (1 + mods[scale]) + mods[shift]` per batch item when `chunks` is set.
struct RmsMod {
    eps: f32,
    rows_per_batch: usize,
    /// `(scale, shift, width)`: the chunks of the `(B, 1, n·width)` modulation tensor.
    chunks: Option<(usize, usize, usize)>,
}

impl RmsMod {
    fn run(&self, x: &CudaStorage, x_l: &Layout, w: &CudaStorage, w_l: &Layout, m: Option<(&CudaStorage, &Layout)>) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let c = *x_l.shape().dims().last().ok_or_else(|| candle_core::Error::Msg("rms of a scalar".into()))?;
        let rows = x_l.shape().elem_count() / c.max(1);
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(rows * c)? };
        {
            let mut guards = Vec::new();
            let xp = ptr!(x, x_l, half::bf16, &stream, guards, "rms x");
            let wp = ptr!(w, w_l, f32, &stream, guards, "rms w");
            let (sp, hp, ld) = match (m, self.chunks) {
                (Some((ms, ml)), Some((si, hi, width))) => {
                    let base = ptr!(ms, ml, half::bf16, &stream, guards, "rms mods");
                    let ld = ml.stride().first().copied().unwrap_or(0);
                    (base + (si * width * 2) as u64, base + (hi * width * 2) as u64, ld)
                }
                _ => (0, 0, 0),
            };
            let (op, _go) = out.device_ptr(&stream);
            check(
                unsafe { qir_k2_rms_mod(op as *mut c_void, xp as *const c_void, wp as *const f32, sp as *const c_void, hp as *const c_void, rows as i32, c as i32, self.rows_per_batch.max(1) as i32, ld as c_long, self.eps, stream.cu_stream() as *mut c_void) },
                "k2 rms",
            )?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

impl candle_core::CustomOp2 for RmsMod {
    fn name(&self) -> &'static str {
        "k2-rms"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, x_l: &Layout, w: &CudaStorage, w_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, x_l, w, w_l, None)
    }
}

impl candle_core::CustomOp3 for RmsMod {
    fn name(&self) -> &'static str {
        "k2-rms-mod"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, x_l: &Layout, w: &CudaStorage, w_l: &Layout, m: &CudaStorage, m_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, x_l, w, w_l, Some((m, m_l)))
    }
}

/// `x (…, C)` bf16 normalised by its RMS and `w (C,)` f32; with `mods = (tensor (B, 1, n·C),
/// scale chunk, shift chunk)` also modulated per batch item (`x` is `(B, S, C)`).
pub fn rms_mod(x: &Tensor, w: &Tensor, mods: Option<(&Tensor, usize, usize)>, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    let dims = x.dims();
    let c = *dims.last().ok_or_else(|| anyhow::anyhow!("rms of a scalar"))?;
    let rows_per_batch = if dims.len() == 3 { dims[1] } else { x.elem_count() / c.max(1) };
    match mods {
        None => Ok(x.apply_op2_no_bwd(w, &RmsMod { eps: eps as f32, rows_per_batch, chunks: None })?),
        Some((m, si, hi)) => Ok(x.apply_op3_no_bwd(w, &m.contiguous()?, &RmsMod { eps: eps as f32, rows_per_batch, chunks: Some((si, hi, c)) })?),
    }
}

/// Per-head RMSNorm (`w (128,)` f32) of `x (B, S, H, 128)` bf16, then the interleaved RoPE of
/// `cs (2, S, 64)` f32 (cos, sin) when given.
struct HeadRope {
    eps: f32,
}

impl candle_core::CustomOp3 for HeadRope {
    fn name(&self) -> &'static str {
        "k2-head-rope"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, x_l: &Layout, w: &CudaStorage, w_l: &Layout, cs: &CudaStorage, cs_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (_, s, h, d) = x_l.shape().dims4()?;
        if d != 128 {
            candle_core::bail!("k2 head rope takes head dim 128, got {d}");
        }
        let rows = x_l.shape().elem_count() / 128;
        let rotate = cs_l.shape().dims3().is_ok_and(|(two, cs_s, half)| two == 2 && cs_s == s && half == 64);
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(rows * 128)? };
        {
            let mut guards = Vec::new();
            let xp = ptr!(x, x_l, half::bf16, &stream, guards, "head rope x");
            let wp = ptr!(w, w_l, f32, &stream, guards, "head rope w");
            let cp = ptr!(cs, cs_l, f32, &stream, guards, "head rope tables");
            let (cos, sin) = if rotate { (cp, cp + (s * 64 * 4) as u64) } else { (0, 0) };
            let (op, _go) = out.device_ptr(&stream);
            check(
                unsafe { qir_k2_head_rope(op as *mut c_void, xp as *const c_void, wp as *const f32, cos as *const f32, sin as *const f32, rows as c_long, h as i32, s as i32, self.eps, stream.cu_stream() as *mut c_void) },
                "k2 head rope",
            )?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

/// `x (B, S, H, 128)` with its heads normalised (`w`) and, given `cs (2, S, 64)`, rotated.
pub fn head_rope(x: &Tensor, w: &Tensor, cs: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let none;
    let cs = match cs {
        Some(t) => t.clone(),
        None => {
            none = Tensor::zeros(1, DType::F32, x.device())?;
            none
        }
    };
    Ok(x.contiguous()?.apply_op3_no_bwd(w, &cs.contiguous()?, &HeadRope { eps: eps as f32 })?)
}

/// The elementwise pairs: `silu(a)·b` and `a·sigmoid(b)`.
#[derive(Clone, Copy)]
enum Pair {
    SwiGlu,
    SigMul,
}

impl candle_core::CustomOp2 for Pair {
    fn name(&self) -> &'static str {
        match self {
            Pair::SwiGlu => "k2-swiglu",
            Pair::SigMul => "k2-sig-mul",
        }
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, a: &CudaStorage, a_l: &Layout, b: &CudaStorage, b_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        if a_l.shape() != b_l.shape() {
            candle_core::bail!("{}: shapes {:?} and {:?}", candle_core::CustomOp2::name(self), a_l.shape(), b_l.shape());
        }
        let dev = a.device().clone();
        let n = a_l.shape().elem_count();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        {
            let mut guards = Vec::new();
            let ap = ptr!(a, a_l, half::bf16, &stream, guards, "pair a");
            let bp = ptr!(b, b_l, half::bf16, &stream, guards, "pair b");
            let (op, _go) = out.device_ptr(&stream);
            let s = stream.cu_stream() as *mut c_void;
            let rc = unsafe {
                match self {
                    Pair::SwiGlu => qir_k2_swiglu(op as *mut c_void, ap as *const c_void, bp as *const c_void, n, s),
                    Pair::SigMul => qir_k2_sig_mul(op as *mut c_void, ap as *const c_void, bp as *const c_void, n, s),
                }
            };
            check(rc, candle_core::CustomOp2::name(self))?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), a_l.shape().clone()))
    }
}

/// `silu(g) · u`.
pub fn swiglu(g: &Tensor, u: &Tensor) -> Result<Tensor> {
    Ok(g.contiguous()?.apply_op2_no_bwd(&u.contiguous()?, &Pair::SwiGlu)?)
}

/// `a · sigmoid(g)`.
pub fn sig_mul(a: &Tensor, g: &Tensor) -> Result<Tensor> {
    Ok(a.contiguous()?.apply_op2_no_bwd(&g.contiguous()?, &Pair::SigMul)?)
}

/// `x + v · mods[chunk]` for `x`, `v (B, S, C)` and `mods (B, 1, n·C)`.
struct GateAdd {
    chunk: usize,
}

impl candle_core::CustomOp3 for GateAdd {
    fn name(&self) -> &'static str {
        "k2-gate-add"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, x_l: &Layout, v: &CudaStorage, v_l: &Layout, m: &CudaStorage, m_l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let (_, s, c) = x_l.shape().dims3()?;
        if v_l.shape() != x_l.shape() {
            candle_core::bail!("k2 gate add: {:?} and {:?}", x_l.shape(), v_l.shape());
        }
        let dev = x.device().clone();
        let n = x_l.shape().elem_count();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(n)? };
        {
            let mut guards = Vec::new();
            let xp = ptr!(x, x_l, half::bf16, &stream, guards, "gate add x");
            let vp = ptr!(v, v_l, half::bf16, &stream, guards, "gate add v");
            let mp = ptr!(m, m_l, half::bf16, &stream, guards, "gate add mods") + (self.chunk * c * 2) as u64;
            let ld = m_l.stride().first().copied().unwrap_or(0);
            let (op, _go) = out.device_ptr(&stream);
            check(
                unsafe { qir_k2_gate_add(op as *mut c_void, xp as *const c_void, vp as *const c_void, mp as *const c_void, n, c as i32, s as c_long, ld as c_long, stream.cu_stream() as *mut c_void) },
                "k2 gate add",
            )?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), x_l.shape().clone()))
    }
}

/// `x + v · mods[chunk]` (the gated residual of a DiT block).
pub fn gate_add(x: &Tensor, v: &Tensor, mods: &Tensor, chunk: usize) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op3_no_bwd(&v.contiguous()?, &mods.contiguous()?, &GateAdd { chunk })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, Device};

    fn max_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    }

    fn rms_ref(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
        let x32 = x.to_dtype(DType::F32)?;
        let rms = (x32.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
        Ok(x32.broadcast_div(&rms)?.broadcast_mul(w)?)
    }

    /// Each fused op against the candle chain it replaces (in f32). Skipped without CUDA.
    #[test]
    fn the_fused_ops_are_the_candle_chains() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (b, s, c, eps) = (2, 37, 256, 1e-6);
        let x = Tensor::randn(0f32, 1.0, (b, s, c), &dev)?.to_dtype(DType::BF16)?;
        let w = (Tensor::randn(0f32, 0.1, c, &dev)? + 1.0)?;
        let mods = Tensor::randn(0f32, 0.5, (b, 1, 6 * c), &dev)?.to_dtype(DType::BF16)?;
        let m = |i: usize| -> Result<Tensor> { Ok(mods.narrow(2, i * c, c)?.to_dtype(DType::F32)?) };

        let want = rms_ref(&x, &w, eps)?.broadcast_mul(&(m(0)? + 1.0)?)?.broadcast_add(&m(1)?)?;
        assert!(max_err(&rms_mod(&x, &w, Some((&mods, 0, 1)), eps)?, &want)? < 0.05);
        assert!(max_err(&rms_mod(&x, &w, None, eps)?, &rms_ref(&x, &w, eps)?)? < 0.03);

        let v = Tensor::randn(0f32, 1.0, (b, s, c), &dev)?.to_dtype(DType::BF16)?;
        let want = (x.to_dtype(DType::F32)? + v.to_dtype(DType::F32)?.broadcast_mul(&m(2)?)?)?;
        assert!(max_err(&gate_add(&x, &v, &mods, 2)?, &want)? < 0.05);

        let (xf, vf) = (x.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);
        assert!(max_err(&swiglu(&x, &v)?, &(candle_nn::ops::silu(&xf)? * &vf)?)? < 0.05);
        assert!(max_err(&sig_mul(&x, &v)?, &(&xf * candle_nn::ops::sigmoid(&vf)?)?)? < 0.03);

        let (h, d) = (3, 128);
        let q = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)?.to_dtype(DType::BF16)?;
        let wq = (Tensor::randn(0f32, 0.1, d, &dev)? + 1.0)?;
        let ang = Tensor::randn(0f32, 1.0, (s, d / 2), &dev)?;
        let (cos, sin) = (ang.cos()?, ang.sin()?);
        let cs = Tensor::stack(&[&cos, &sin], 0)?;
        let normed = rms_ref(&q, &wq, eps)?; // (B,S,H,D) f32
        let want = candle_nn::rotary_emb::rope_i(&normed.transpose(1, 2)?.contiguous()?, &cos, &sin)?.transpose(1, 2)?;
        assert!(max_err(&head_rope(&q, &wq, Some(&cs), eps)?, &want)? < 0.05);
        assert!(max_err(&head_rope(&q, &wq, None, eps)?, &normed)? < 0.05);
        Ok(())
    }
}
