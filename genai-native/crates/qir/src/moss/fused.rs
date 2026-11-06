//! MOSS-Image's SiTU activation in one pass (kernels/fusednorm/moss_fused.cu, feature
//! `fusednorm`), read straight from the merged `ffn_up_gate` output. The block's norms, q/k
//! norms with RoPE and gated residuals use Krea 2's fused ops (`crate::krea2_fused`).

use std::ffi::{c_long, c_void};

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};

use crate::layout::dense_byte_offset;
use crate::Result;

extern "C" {
    fn qir_moss_situ(y: *mut c_void, hg: *const c_void, rows: c_long, f: i32, stream: *mut c_void) -> i32;
}

struct Situ;

impl candle_core::CustomOp1 for Situ {
    fn name(&self) -> &'static str {
        "moss-situ"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("the MOSS fused SiTU is CUDA-only")
    }

    fn cuda_fwd(&self, hg: &CudaStorage, l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let (rows, two_f) = l.shape().dims2()?;
        let f = two_f / 2;
        let dev = hg.device().clone();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::bf16>(rows * f)? };
        {
            let off = dense_byte_offset::<half::bf16>(l, "situ hg")?;
            let slice = hg.as_cuda_slice::<half::bf16>()?;
            let (p, _g) = slice.device_ptr(&stream);
            let (op, _go) = out.device_ptr(&stream);
            let rc = unsafe { qir_moss_situ(op as *mut c_void, (p + off) as *const c_void, rows as c_long, f as i32, stream.cu_stream() as *mut c_void) };
            if rc != 0 {
                candle_core::bail!("moss situ failed, rc={rc}");
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((rows, f))))
    }
}

/// SiTU of the merged `hg (M, 2F)` bf16 (hidden first, gate second): `(M, F)` bf16.
pub fn situ(hg: &Tensor) -> Result<Tensor> {
    Ok(hg.contiguous()?.apply_op1_no_bwd(&Situ)?)
}
