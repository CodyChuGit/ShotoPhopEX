//! The Wan VAE's convolutions as implicit GEMM (`kernels/convrot/vae_conv.cu`, CUTLASS's
//! tensor-op Conv2dFprop) instead of candle's im2col + GEMM: NHWC bf16 activations, KRSC bf16
//! weights, f32 accumulation, the bias in the epilogue. `QIR_VAE_IGEMM=0` keeps candle's.

use std::ffi::c_void;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

use crate::layout::dense_byte_offset;
use crate::Result;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn qir_vae_conv2d_nhwc(x: *const c_void, w: *const c_void, bias: *const c_void, y: *mut c_void, n: i32, h: i32, w_: i32, c: i32, k: i32, r: i32, stream: *mut c_void) -> i32;
}

/// Whether the implicit-GEMM convolutions are on (`QIR_VAE_IGEMM=0` turns them off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_VAE_IGEMM").map_or(true, |v| v.trim() != "0"))
}

/// Whether a `(O, I, k, k)` weight takes this path: bf16 on CUDA, odd `k`, `O` and `I`
/// multiples of 8.
pub fn supports(weight: &Tensor) -> bool {
    weight.device().is_cuda()
        && weight.dtype() == DType::BF16
        && weight.dims4().is_ok_and(|(o, i, kh, kw)| o % 8 == 0 && i % 8 == 0 && kh == kw && kh % 2 == 1)
}

/// The weight `(O, I, k, k)` in the kernel's KRSC layout `(O, k, k, I)`.
pub fn krsc(weight: &Tensor) -> Result<Tensor> {
    Ok(weight.permute((0, 2, 3, 1))?.contiguous()?)
}

/// `conv(x) + bias` for `x (N, C, H, W)` bf16, `krsc (K, R, R, C)` ([`krsc`]), `bias (K,)`,
/// stride 1, padding R/2: `(N, K, H, W)` bf16.
pub fn conv2d_nchw(x: &Tensor, krsc: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let nhwc = x.permute((0, 2, 3, 1))?.contiguous()?;
    let y = nhwc.apply_op3(krsc, &bias.contiguous()?, ConvNhwc)?;
    Ok(y.permute((0, 3, 1, 2))?.contiguous()?)
}

struct ConvNhwc;

impl candle_core::CustomOp3 for ConvNhwc {
    fn name(&self) -> &'static str {
        "vae-conv2d-nhwc"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vae-conv2d-nhwc is CUDA-only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout, b: &CudaStorage, bl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let (n, h, wd, c) = xl.shape().dims4()?;
        let (k, r, s, wc) = wl.shape().dims4()?;
        if r != s || wc != c || bl.shape().dims1()? != k {
            candle_core::bail!("vae-conv2d-nhwc: x {:?}, weight {:?}, bias {:?} disagree", xl.shape(), wl.shape(), bl.shape());
        }
        let dev = x.device().clone();
        let xo = dense_byte_offset::<half::bf16>(xl, "vae-conv x")?;
        let wo = dense_byte_offset::<half::bf16>(wl, "vae-conv weight")?;
        let bo = dense_byte_offset::<half::bf16>(bl, "vae-conv bias")?;
        let (xs, ws, bs) = (x.as_cuda_slice::<half::bf16>()?, w.as_cuda_slice::<half::bf16>()?, b.as_cuda_slice::<half::bf16>()?);
        let stream = dev.cuda_stream();
        let y = unsafe { dev.alloc::<half::bf16>(n * h * wd * k)? };
        let rc;
        {
            let (xp, _a) = xs.device_ptr(&stream);
            let (wp, _b) = ws.device_ptr(&stream);
            let (bp, _c) = bs.device_ptr(&stream);
            let (yp, _d) = y.device_ptr(&stream);
            rc = unsafe {
                qir_vae_conv2d_nhwc(
                    (xp + xo) as *const c_void,
                    (wp + wo) as *const c_void,
                    (bp + bo) as *const c_void,
                    yp as *mut c_void,
                    n as i32,
                    h as i32,
                    wd as i32,
                    c as i32,
                    k as i32,
                    r as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        if rc != 0 {
            candle_core::bail!("qir_vae_conv2d_nhwc failed rc={rc} (N={n} H={h} W={wd} C={c} K={k} R={r})");
        }
        Ok((CudaStorage::wrap_cuda_slice(y, dev), (n, h, wd, k).into()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;

    /// The implicit GEMM against candle's im2col convolution in f32, for a wide 3x3, a narrow
    /// 3x3 and a 1x1, on an odd-sized frame.
    #[test]
    fn implicit_gemm_matches_candles_convolution() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for (c, k, r) in [(96, 192, 3), (192, 16, 3), (384, 384, 1)] {
            let x = Tensor::randn(0f32, 1.0, (2, c, 37, 53), &dev)?.to_dtype(DType::BF16)?;
            let w = (Tensor::randn(0f32, 1.0, (k, c, r, r), &dev)? / ((c * r * r) as f64).sqrt())?.to_dtype(DType::BF16)?;
            let b = Tensor::randn(0f32, 0.5, k, &dev)?.to_dtype(DType::BF16)?;
            assert!(supports(&w));
            let fast = conv2d_nchw(&x, &krsc(&w)?, &b)?.to_dtype(DType::F32)?;
            let exact = x.to_dtype(DType::F32)?.conv2d(&w.to_dtype(DType::F32)?, r / 2, 1, 1, 1)?.broadcast_add(&b.to_dtype(DType::F32)?.reshape((1, k, 1, 1))?)?;
            assert_eq!(fast.dims(), exact.dims());
            let err = (&fast - &exact)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = exact.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(err <= scale * 0.01, "C={c} K={k} R={r}: max error {err} of {scale}");
        }
        Ok(())
    }
}
