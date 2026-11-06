//! cuBLASLt BF16 projections for narrow adapter matrices.
use std::{collections::HashMap, ffi::c_void, sync::Mutex};

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::{
    cublaslt::{result as lt, sys},
    driver::{CudaSlice, DevicePtr},
};
use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};

use crate::{layout::dense_byte_offset, Result};

const WORKSPACE: usize = 4 << 20;

struct State {
    handle: sys::cublasLtHandle_t,
    plans: HashMap<(usize, usize, usize, [u32; 4]), Plan>,
    workspaces: HashMap<usize, CudaSlice<u8>>,
    device: candle_core::cuda_backend::DeviceId,
    owner: candle_core::CudaDevice,
}

// SAFETY: every use of the handle is serialized by STATE.
unsafe impl Send for State {}
static STATE: Mutex<Option<State>> = Mutex::new(None);

impl Drop for State {
    fn drop(&mut self) {
        let _ = self.owner.cuda_stream().context().bind_to_thread();
        unsafe {
            let _ = lt::destroy_handle(self.handle);
        }
    }
}

struct Descriptors {
    mm: sys::cublasLtMatmulDesc_t,
    layouts: Vec<sys::cublasLtMatrixLayout_t>,
    pref: Option<sys::cublasLtMatmulPreference_t>,
}

struct Plan {
    descriptors: Descriptors,
    algo: sys::cublasLtMatmulAlgo_t,
}

fn plan(m: usize, n: usize, k: usize, alignments: [u32; 4], handle: sys::cublasLtHandle_t) -> candle_core::Result<Plan> {
    let mm = lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F).map_err(err)?;
    let mut desc = Descriptors { mm, layouts: Vec::new(), pref: None };
    let transpose: i32 = 1;
    unsafe {
        lt::set_matmul_desc_attribute(
            mm,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transpose as *const _ as *const c_void,
            std::mem::size_of_val(&transpose),
        )
        .map_err(err)?;
    }
    for (rows, cols, ld) in [(k, n, k), (k, m, k), (n, m, n)] {
        desc.layouts.push(lt::create_matrix_layout(sys::cudaDataType::CUDA_R_16BF, rows as u64, cols as u64, ld as i64).map_err(err)?);
    }
    let (a, b, d) = (desc.layouts[0], desc.layouts[1], desc.layouts[2]);
    let pref = lt::create_matmul_pref().map_err(err)?;
    desc.pref = Some(pref);
    unsafe {
        for (attribute, alignment) in [
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ]
        .into_iter()
        .zip(alignments)
        {
            lt::set_matmul_pref_attribute(pref, attribute, &alignment as *const _ as *const c_void, std::mem::size_of_val(&alignment)).map_err(err)?;
        }
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WORKSPACE as *const _ as *const c_void,
            std::mem::size_of_val(&WORKSPACE),
        )
        .map_err(err)?;
    }
    let heuristic = unsafe { lt::get_matmul_algo_heuristic(handle, mm, a, b, d, d, pref) }.map_err(err)?;
    Ok(Plan { descriptors: desc, algo: heuristic.algo })
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        unsafe {
            for l in self.layouts.drain(..) {
                let _ = lt::destroy_matrix_layout(l);
            }
            if let Some(p) = self.pref {
                let _ = lt::destroy_matmul_pref(p);
            }
            let _ = lt::destroy_matmul_desc(self.mm);
        }
    }
}

fn err(e: impl std::fmt::Debug) -> candle_core::Error {
    candle_core::Error::Msg(format!("LoRA cuBLASLt: {e:?}"))
}

/// `x (M,K) @ weight (N,K).T`, with FP32 accumulation.
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if !weight.dim(0)?.is_multiple_of(8)
        || !weight.dim(1)?.is_multiple_of(8)
        || !x.layout().start_offset().is_multiple_of(8)
        || !weight.layout().start_offset().is_multiple_of(8)
    {
        return Ok(x.matmul(&weight.t()?)?);
    }
    Ok(x.contiguous()?.apply_op2_no_bwd(&weight.contiguous()?, &Gemm)?)
}

/// Project and accumulate into a fresh output; `base` may have live aliases.
pub fn linear_add(x: &Tensor, weight: &Tensor, base: &Tensor, scale: f32) -> Result<Tensor> {
    if !weight.dim(0)?.is_multiple_of(8)
        || !weight.dim(1)?.is_multiple_of(8)
        || !x.layout().start_offset().is_multiple_of(8)
        || !weight.layout().start_offset().is_multiple_of(8)
        || !base.layout().start_offset().is_multiple_of(8)
    {
        return Ok((base + (x.matmul(&weight.t()?)? * f64::from(scale))?)?);
    }
    Ok(x.contiguous()?.apply_op3_no_bwd(&weight.contiguous()?, &base.contiguous()?, &GemmAdd(scale))?)
}

struct Gemm;
impl candle_core::CustomOp2 for Gemm {
    fn name(&self) -> &'static str {
        "lora-gemm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("LoRA cuBLASLt requires CUDA")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        gemm(x, xl, w, wl, None, 1.0)
    }
}

struct GemmAdd(f32);
impl candle_core::CustomOp3 for GemmAdd {
    fn name(&self) -> &'static str {
        "lora-gemm-add"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("LoRA cuBLASLt requires CUDA")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        base: &CudaStorage,
        bl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        gemm(x, xl, w, wl, Some((base, bl)), self.0)
    }
}

fn gemm(
    x: &CudaStorage,
    xl: &Layout,
    w: &CudaStorage,
    wl: &Layout,
    base: Option<(&CudaStorage, &Layout)>,
    scale: f32,
) -> candle_core::Result<(CudaStorage, Shape)> {
    let dev = x.device().clone();
    let (m, k) = xl.shape().dims2()?;
    let (n, wk) = wl.shape().dims2()?;
    if k != wk {
        candle_core::bail!("LoRA GEMM inner dimension mismatch");
    }
    let xs = x.as_cuda_slice::<half::bf16>()?;
    let ws = w.as_cuda_slice::<half::bf16>()?;
    let stream = dev.cuda_stream();
    let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
    let cs = base
        .map(|(s, l)| -> candle_core::Result<_> {
            if l.shape().dims2()? != (m, n) {
                candle_core::bail!("LoRA GEMM accumulator shape mismatch");
            }
            Ok((s.as_cuda_slice::<half::bf16>()?, dense_byte_offset::<half::bf16>(l, "LoRA base")?))
        })
        .transpose()?;
    {
        let (xp, _gx) = xs.device_ptr(&stream);
        let (wp, _gw) = ws.device_ptr(&stream);
        let (op, _go) = out.device_ptr(&stream);
        let cp = cs.as_ref().map(|(s, offset)| {
            let (p, guard) = s.device_ptr(&stream);
            (p + offset, guard)
        });
        let xp = xp + dense_byte_offset::<half::bf16>(xl, "LoRA input")?;
        let wp = wp + dense_byte_offset::<half::bf16>(wl, "LoRA weight")?;
        let cptr = cp.as_ref().map_or(op, |(p, _)| *p);
        let alignments = [wp, xp, cptr, op].map(|p| 1u32 << p.trailing_zeros().min(8));
        let key = (m, n, k, alignments);
        let mut guard = STATE.lock().map_err(err)?;
        if guard.as_ref().is_none_or(|s| s.device != dev.id()) {
            *guard = Some(State {
                handle: lt::create_handle().map_err(err)?,
                plans: HashMap::new(),
                workspaces: HashMap::new(),
                device: dev.id(),
                owner: dev.clone(),
            });
        }
        stream.context().bind_to_thread().map_err(err)?;
        let state = guard.as_mut().ok_or_else(|| err("missing handle"))?;
        if !state.plans.contains_key(&key) {
            tracing::debug!(m, n, k, ?alignments, "LoRA GEMM plan");
            state.plans.insert(key, plan(m, n, k, alignments, state.handle)?);
        }
        let stream_key = stream.cu_stream() as usize;
        if let std::collections::hash_map::Entry::Vacant(slot) = state.workspaces.entry(stream_key) {
            slot.insert(unsafe { dev.alloc::<u8>(WORKSPACE)? });
        }
        let plan = state.plans.get(&key).ok_or_else(|| err("missing plan"))?;
        let work = state.workspaces.get(&stream_key).ok_or_else(|| err("missing workspace"))?;
        let (scratch, _gs) = work.device_ptr(&stream);
        let desc = &plan.descriptors;
        let (a, b, d) = (desc.layouts[0], desc.layouts[1], desc.layouts[2]);
        let (alpha, beta) = (scale, if cp.is_some() { 1f32 } else { 0f32 });
        let launch = |algo: &sys::cublasLtMatmulAlgo_t| unsafe {
            lt::matmul(
                state.handle,
                desc.mm,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                wp as *const c_void,
                a,
                xp as *const c_void,
                b,
                cptr as *const c_void,
                d,
                op as *mut c_void,
                d,
                algo,
                scratch as *mut c_void,
                WORKSPACE,
                stream.cu_stream() as sys::cudaStream_t,
            )
            .map_err(err)
        };
        launch(&plan.algo)?;
    }
    Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, n).into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn accumulator_views_keep_their_aliases_and_offsets() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let x = Tensor::randn(0f32, 1., (9, 32), &dev)?.to_dtype(DType::BF16)?.narrow(0, 2, 5)?;
        let w = Tensor::randn(0f32, 1., (24, 32), &dev)?.to_dtype(DType::BF16)?.narrow(0, 8, 16)?;
        let slab = Tensor::ones((9, 16), DType::BF16, &dev)?;
        let base = slab.narrow(0, 2, 5)?;
        for scale in [0., 0.5, -1.] {
            let got = linear_add(&x, &w, &base, scale)?.to_dtype(DType::F32)?;
            let want = (x.to_dtype(DType::F32)?.matmul(&w.to_dtype(DType::F32)?.t()?)? * f64::from(scale))?.broadcast_add(&base.to_dtype(DType::F32)?)?;
            let error = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(error < 0.125, "{error}");
        }
        assert_eq!(slab.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?, 144.);
        Ok(())
    }
}
