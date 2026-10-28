//! Load-time ConvRot weight patching kernels.
use crate::{layout::dense_byte_offset, Result};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};
use std::ffi::c_void;

extern "C" {
    fn qir_lora_weight_dequant(out: *mut c_void, q: *const u8, scales: *const f32, n: i32, k: i32, stream: *mut c_void) -> i32;
    fn qir_lora_weight_rotate(out: *mut f32, w: *const c_void, n: i32, k: i32, round_rot: i32, stream: *mut c_void) -> i32;
    fn qir_lora_weight_quant(out: *mut u8, rotated: *const f32, scales: *mut f32, n: i32, k: i32, seed: u64, stream: *mut c_void) -> i32;
}

fn status(code: i32) -> candle_core::Result<()> {
    if code != 0 {
        candle_core::bail!("LoRA materialization kernel failed: {code}");
    }
    Ok(())
}

fn matrix(layout: &Layout) -> candle_core::Result<(usize, usize)> {
    let (n, k) = layout.shape().dims2()?;
    if n == 0 || k == 0 || !k.is_multiple_of(256) || n.checked_mul(k).is_none_or(|len| len > i32::MAX as usize) {
        candle_core::bail!("invalid ConvRot materialization matrix shape");
    }
    Ok((n, k))
}

/// ComfyUI dynamic materialization seeds by module name, without `.weight`.
pub fn key_seed(module: &str) -> u64 {
    let mut crc = !0u32;
    for byte in format!("diffusion_model.{module}").bytes() {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320u32 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    u64::from(!crc)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parameter_seed_matches_installed_comfyui() {
        assert_eq!(super::key_seed("transformer_blocks.0.attn.to_q"), 706425777);
    }
}

pub(crate) fn merge(q: &Tensor, scales: &Tensor, pairs: &[crate::lora::LoraPair], module: &str) -> Result<(Tensor, Tensor)> {
    if pairs.is_empty() {
        return Ok((q.clone(), scales.clone()));
    }
    let mut weight = dequantize(q, scales)?;
    for pair in pairs {
        weight = pair.merge_weight(weight)?;
    }
    requantize(&weight, key_seed(module), false)
}

pub fn dequantize(q: &Tensor, scales: &Tensor) -> Result<Tensor> {
    Ok(q.contiguous()?.apply_op2_no_bwd(&scales.contiguous()?, &Dequant)?)
}

pub fn requantize(weight: &Tensor, seed: u64, round_rot: bool) -> Result<(Tensor, Tensor)> {
    let n = weight.dim(0)?;
    let rotated = weight.contiguous()?.apply_op1_no_bwd(&Rotate(round_rot))?;
    // This fresh scale buffer is private until the custom op finishes writing it.
    let scales = Tensor::zeros(n, DType::F32, weight.device())?;
    let q = rotated.apply_op2_no_bwd(&scales, &Quant(seed))?;
    Ok((q, scales))
}

struct Dequant;
impl candle_core::CustomOp2 for Dequant {
    fn name(&self) -> &'static str {
        "lora-weight-dequant"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA required")
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout, s: &CudaStorage, sl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();
        let (n, k) = matrix(ql)?;
        if !k.is_multiple_of(256) || sl.shape().elem_count() != n {
            candle_core::bail!("invalid ConvRot weight shape");
        }
        let qs = q.as_cuda_slice::<u8>()?;
        let ss = s.as_cuda_slice::<f32>()?;
        let out = unsafe { dev.alloc::<half::bf16>(n * k)? };
        {
            let (qp, _a) = qs.device_ptr(&stream);
            let (sp, _b) = ss.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            unsafe {
                status(qir_lora_weight_dequant(
                    op as *mut c_void,
                    (qp + dense_byte_offset::<u8>(ql, "weight")?) as *const u8,
                    (sp + dense_byte_offset::<f32>(sl, "scale")?) as *const f32,
                    n as i32,
                    k as i32,
                    stream.cu_stream() as *mut c_void,
                ))?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (n, k).into()))
    }
}

struct Rotate(bool);
impl candle_core::CustomOp1 for Rotate {
    fn name(&self) -> &'static str {
        "lora-weight-rotate"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA required")
    }
    fn cuda_fwd(&self, w: &CudaStorage, wl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = w.device().clone();
        let stream = dev.cuda_stream();
        let (n, k) = matrix(wl)?;
        if !k.is_multiple_of(256) {
            candle_core::bail!("ConvRot group must divide weight width");
        }
        let ws = w.as_cuda_slice::<half::bf16>()?;
        let out = unsafe { dev.alloc::<f32>(n * k)? };
        {
            let (wp, _a) = ws.device_ptr(&stream);
            let (op, _b) = out.device_ptr(&stream);
            unsafe {
                status(qir_lora_weight_rotate(
                    op as *mut f32,
                    (wp + dense_byte_offset::<half::bf16>(wl, "weight")?) as *const c_void,
                    n as i32,
                    k as i32,
                    i32::from(self.0),
                    stream.cu_stream() as *mut c_void,
                ))?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (n, k).into()))
    }
}

struct Quant(u64);
impl candle_core::CustomOp2 for Quant {
    fn name(&self) -> &'static str {
        "lora-weight-quant"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA required")
    }
    fn cuda_fwd(&self, w: &CudaStorage, wl: &Layout, s: &CudaStorage, sl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = w.device().clone();
        let stream = dev.cuda_stream();
        let (n, k) = matrix(wl)?;
        if sl.shape().elem_count() != n {
            candle_core::bail!("ConvRot scale count mismatch");
        }
        let ws = w.as_cuda_slice::<f32>()?;
        let ss = s.as_cuda_slice::<f32>()?;
        let out = unsafe { dev.alloc::<u8>(n * k)? };
        {
            let (wp, _a) = ws.device_ptr(&stream);
            let (sp, _b) = ss.device_ptr(&stream);
            let (op, _c) = out.device_ptr(&stream);
            unsafe {
                status(qir_lora_weight_quant(
                    op as *mut u8,
                    (wp + dense_byte_offset::<f32>(wl, "rotated")?) as *const f32,
                    (sp + dense_byte_offset::<f32>(sl, "scale")?) as *mut f32,
                    n as i32,
                    k as i32,
                    self.0,
                    stream.cu_stream() as *mut c_void,
                ))?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (n, k).into()))
    }
}
