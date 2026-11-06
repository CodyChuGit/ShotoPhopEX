//! Layered-only bf16 GEMMs cache cuBLASLt plans for the three image-stream shapes.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, Weak};

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::{
    cublaslt::{result as lt, sys},
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
};
use candle_core::{CpuStorage, CudaStorage, Device, Layout, Shape, Tensor};

use crate::layout::dense_offset;
use crate::Result;

const WORKSPACE: usize = 32 << 20;

thread_local! {
    static HANDLE: RefCell<(usize, Weak<Handle>)> = const { RefCell::new((0, Weak::new())) };
}

pub(super) struct Handle {
    raw: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    stream: Arc<CudaStream>,
    plans: Mutex<HashMap<(usize, usize, usize), Plan>>,
}

// SAFETY: plan access is serialized, and all workspace uses are ordered on one fixed CUDA stream.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        if let Ok(plans) = self.plans.get_mut() {
            plans.clear();
        }
        // SAFETY: the last owning Arc is gone and no matmul can still be enqueued through it.
        let _ = unsafe { lt::destroy_handle(self.raw) };
    }
}

/// Share plans and workspace during a split; the weak cache retains no GPU allocation.
pub(super) fn handle(device: &Device) -> Result<Arc<Handle>> {
    let Device::Cuda(device) = device else { anyhow::bail!("Layered GEMM requires CUDA") };
    let stream = device.cuda_stream();
    let key = Arc::as_ptr(&stream) as usize;
    HANDLE.with(|cache| {
        let mut cache = cache.try_borrow_mut().map_err(|e| anyhow::anyhow!("Layered GEMM cache: {e}"))?;
        if cache.0 == key {
            if let Some(handle) = cache.1.upgrade() {
                return Ok(handle);
            }
        }
        // Allocate fallibly before creating the handle so allocation errors cannot leak it.
        let workspace = unsafe { device.alloc::<u8>(WORKSPACE)? };
        let raw = lt::create_handle()?;
        let handle = Arc::new(Handle { raw, workspace, stream, plans: Mutex::new(HashMap::new()) });
        *cache = (key, Arc::downgrade(&handle));
        Ok(handle)
    })
}

struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    layouts: Vec<sys::cublasLtMatrixLayout_t>,
    pref: Option<sys::cublasLtMatmulPreference_t>,
    algo: Option<sys::cublasLtMatmulAlgo_t>,
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: these descriptors belong to this plan and are never shared outside the cache.
        unsafe {
            for layout in self.layouts.drain(..) {
                let _ = lt::destroy_matrix_layout(layout);
            }
            if let Some(pref) = self.pref.take() {
                let _ = lt::destroy_matmul_pref(pref);
            }
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

fn error(e: impl std::fmt::Debug) -> candle_core::Error {
    candle_core::Error::Msg(format!("Layered cuBLASLt: {e:?}"))
}

impl Plan {
    fn new(handle: sys::cublasLtHandle_t, m: usize, n: usize, k: usize) -> candle_core::Result<Self> {
        let desc = lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F).map_err(error)?;
        let mut plan = Self { desc, layouts: Vec::new(), pref: None, algo: None };
        let (transposed, normal) = (1i32, 0i32);
        // SAFETY: cuBLASLt copies the scalar attributes; these are valid descriptor handles.
        unsafe {
            for (attribute, value) in [
                (sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &transposed),
                (sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &normal),
                (sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC, &normal),
            ] {
                lt::set_matmul_desc_attribute(desc, attribute, value as *const _ as *const c_void, std::mem::size_of::<i32>()).map_err(error)?;
            }
        }
        for (rows, cols, ld) in [(k, n, k), (k, m, k), (n, m, n)] {
            plan.layouts.push(lt::create_matrix_layout(sys::cudaDataType::CUDA_R_16BF, rows as u64, cols as u64, ld as i64).map_err(error)?);
        }
        let pref = lt::create_matmul_pref().map_err(error)?;
        plan.pref = Some(pref);
        // SAFETY: layouts describe dense bf16 matrices and workspace is owned by the handle.
        unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE as *const _ as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .map_err(error)?;
            plan.algo = Some(
                lt::get_matmul_algo_heuristic(handle, desc, plan.layouts[0], plan.layouts[1], plan.layouts[2], plan.layouts[2], pref).map_err(error)?.algo,
            );
        }
        Ok(plan)
    }
}

pub(super) fn forward(x: &Tensor, weight: &Tensor, handle: &Arc<Handle>) -> Result<Tensor> {
    let (n, k) = weight.dims2()?;
    let mut shape = x.dims().to_vec();
    if let Some(last) = shape.last_mut() {
        *last = n;
    }
    Ok(x.reshape(((), k))?.apply_op2_no_bwd(weight, &Gemm(handle.clone()))?.reshape(shape)?)
}

struct Gemm(Arc<Handle>);

impl candle_core::CustomOp2 for Gemm {
    fn name(&self) -> &'static str {
        "layered-bf16-lt"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Layered GEMM requires CUDA")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if k != wk {
            candle_core::bail!("Layered GEMM width mismatch");
        }
        let xo = dense_offset(xl, "Layered GEMM x")?;
        let wo = dense_offset(wl, "Layered GEMM weight")?;
        let dev = x.device().clone();
        let stream = &self.0.stream;
        let xv = x.as_cuda_slice::<half::bf16>()?;
        let wv = w.as_cuda_slice::<half::bf16>()?;
        // SAFETY: cuBLASLt writes every element of this fresh output on its owning stream.
        let mut out = unsafe { dev.alloc::<half::bf16>(m * n)? };
        let mut plans = self.0.plans.lock().map_err(error)?;
        if let std::collections::hash_map::Entry::Vacant(entry) = plans.entry((m, n, k)) {
            entry.insert(Plan::new(self.0.raw, m, n, k)?);
        }
        let plan = plans.get(&(m, n, k)).ok_or_else(|| error("missing GEMM plan"))?;
        let algo = plan.algo.as_ref().ok_or_else(|| error("missing GEMM algorithm"))?;
        let ((xp, _gx), (wp, _gw), (op, _go), (scratch, _gs)) =
            (xv.device_ptr(stream), wv.device_ptr(stream), out.device_ptr_mut(stream), self.0.workspace.device_ptr(stream));
        let (alpha, beta) = (1f32, 0f32);
        // SAFETY: validated dense bf16 views, matching cached layouts, and stream-ordered owners.
        unsafe {
            lt::matmul(
                self.0.raw,
                plan.desc,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                (wp + (wo * 2) as u64) as *const c_void,
                plan.layouts[0],
                (xp + (xo * 2) as u64) as *const c_void,
                plan.layouts[1],
                op as *const c_void,
                plan.layouts[2],
                op as *mut c_void,
                plan.layouts[2],
                algo,
                scratch as *mut c_void,
                WORKSPACE,
                stream.cu_stream() as sys::cudaStream_t,
            )
            .map_err(error)?;
        }
        drop((_gx, _gw, _go, _gs));
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, n).into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn layered_gemm_offsets_plans_and_workspace_lifetime() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let first = handle(&dev)?;
        let second = handle(&dev)?;
        assert!(Arc::ptr_eq(&first, &second));
        let x = Tensor::randn(0f32, 0.5, (3, 37, 128), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2)?;
        let w = Tensor::randn(0f32, 0.5, (192, 128), &dev)?.to_dtype(DType::BF16)?.narrow(0, 96, 96)?;
        let want = x.reshape((74, 128))?.matmul(&w.t()?)?.reshape((2, 37, 96))?;
        for _ in 0..2 {
            let got = forward(&x, &w, &first)?;
            let err = (want.to_dtype(DType::F32)? - got.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(err <= 0.03125, "GEMM max error {err}");
        }
        assert_eq!(first.plans.lock().map_err(error)?.len(), 1);
        let weak = Arc::downgrade(&first);
        drop(first);
        drop(second);
        assert!(weak.upgrade().is_none());
        Ok(())
    }
}
