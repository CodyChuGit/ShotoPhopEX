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

// ---- AdaLN: LayerNorm + modulation, gated residuals ------------------------------------------

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn qir_idol_ln_mod(h: *mut c_void, x: *const c_void, m: *const c_void, rows: i64, l: i32, c: i32, sh: i32, sc: i32, eps: f32, stream: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn qir_idol_res_ln_mod(out: *mut c_void, x: *const c_void, a: *const c_void, m: *const c_void, rows: i64, l: i32, c: i32, g: i32, sh: i32, sc: i32, eps: f32, stream: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn qir_idol_gate_res(out: *mut c_void, x: *const c_void, f: *const c_void, m: *const c_void, rows: i64, l: i32, c: i32, g: i32, stream: *mut c_void) -> i32;
}

/// Which modulation chunks (of the six in `m (B, 6C)`) a fused AdaLN op reads.
#[derive(Clone, Copy)]
enum AdaOp {
    /// `LN(x)·(1 + m[sc]) + m[sh]`
    LnMod { sh: usize, sc: usize },
    /// `x + a·m[g]`, then `LN` of it modulated: both planes out.
    ResLnMod { g: usize, sh: usize, sc: usize },
    /// `x + f·m[g]`
    GateRes { g: usize },
}

struct Ada {
    op: AdaOp,
    eps: f32,
}

impl Ada {
    /// `x`'s rows, their length `L` per batch item and their width `C`; checks `m` is `(B, 6C)`.
    fn dims(xl: &Layout, ml: &Layout) -> candle_core::Result<(usize, usize, usize)> {
        let (b, l, c) = xl.shape().dims3()?;
        if ml.shape().dims2()? != (b, 6 * c) || !c.is_multiple_of(8) || c / 8 > 1024 || c / 8 % 32 != 0 {
            candle_core::bail!("idol AdaLN: x {:?}, m {:?} (m must be (B, 6C), C/8 a multiple of 32 up to 1024)", xl.shape(), ml.shape());
        }
        Ok((b * l, l, c))
    }

    fn run(&self, x: &CudaStorage, xl: &Layout, a: Option<(&CudaStorage, &Layout)>, m: &CudaStorage, ml: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let (rows, l, c) = Self::dims(xl, ml)?;
        if [Some(x.dtype()), a.map(|(s, _)| s.dtype()), Some(m.dtype())].into_iter().flatten().any(|d| d != DType::F16) {
            candle_core::bail!("idol AdaLN takes f16");
        }
        if let Some((_, al)) = a {
            if al.shape() != xl.shape() {
                candle_core::bail!("idol AdaLN: x {:?} and its update {:?} differ", xl.shape(), al.shape());
            }
        }
        let dev = x.device().clone();
        let stream = dev.cuda_stream();
        let planes = if matches!(self.op, AdaOp::ResLnMod { .. }) { 2 } else { 1 };
        let out = unsafe { dev.alloc::<half::f16>(planes * rows * c)? };
        {
            let xp = ptr::<half::f16>(x, xl, "x", &stream)? as *const c_void;
            let ap = match a {
                Some((s, al)) => ptr::<half::f16>(s, al, "update", &stream)? as *const c_void,
                None => std::ptr::null(),
            };
            let mp = ptr::<half::f16>(m, ml, "modulation", &stream)? as *const c_void;
            let (op, _go) = out.device_ptr(&stream);
            let op = op as *mut c_void;
            let s = stream.cu_stream() as *mut c_void;
            let (rows_i, l_i, c_i) = (rows as i64, l as i32, c as i32);
            let rc = unsafe {
                match self.op {
                    AdaOp::LnMod { sh, sc } => qir_idol_ln_mod(op, xp, mp, rows_i, l_i, c_i, sh as i32, sc as i32, self.eps, s),
                    AdaOp::ResLnMod { g, sh, sc } => qir_idol_res_ln_mod(op, xp, ap, mp, rows_i, l_i, c_i, g as i32, sh as i32, sc as i32, self.eps, s),
                    AdaOp::GateRes { g } => qir_idol_gate_res(op, xp, ap, mp, rows_i, l_i, c_i, g as i32, s),
                }
            };
            if rc != 0 {
                candle_core::bail!("idol AdaLN kernel failed, rc={rc}");
            }
        }
        let shape: Shape = if planes == 2 { (2, rows / l, l, c).into() } else { xl.shape().clone() };
        Ok((CudaStorage::wrap_cuda_slice(out, dev), shape))
    }
}

impl candle_core::CustomOp2 for Ada {
    fn name(&self) -> &'static str {
        "idol-adaln"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("idol AdaLN is CUDA-only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, m: &CudaStorage, ml: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, xl, None, m, ml)
    }
}

impl candle_core::CustomOp3 for Ada {
    fn name(&self) -> &'static str {
        "idol-adaln"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("idol AdaLN is CUDA-only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, a: &CudaStorage, al: &Layout, m: &CudaStorage, ml: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        self.run(x, xl, Some((a, al)), m, ml)
    }
}

/// Whether the fused AdaLN ops take `x (B, L, C)`: CUDA, f16, `C/8` a multiple of 32 up to 1024.
pub fn adaln_fits(x: &Tensor) -> bool {
    let c = x.dims().last().copied().unwrap_or(0);
    x.device().is_cuda() && x.dtype() == DType::F16 && x.rank() == 3 && c.is_multiple_of(8) && c / 8 <= 1024 && (c / 8).is_multiple_of(32)
}

/// `LN(x)·(1 + m[sc]) + m[sh]` for `x (B, L, C)` and the modulation `m (B, 6C)` (chunk `j` is
/// `m[:, jC..(j+1)C]`), the LayerNorm without weights.
pub fn ln_mod(x: &Tensor, m: &Tensor, sh: usize, sc: usize, eps: f64) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op2_no_bwd(&m.contiguous()?, &Ada { op: AdaOp::LnMod { sh, sc }, eps: eps as f32 })?)
}

/// `(x2, h)` with `x2 = x + a·m[g]` (rounded to f16, the residual stream) and
/// `h = LN(x2)·(1 + m[sc]) + m[sh]`.
pub fn res_ln_mod(x: &Tensor, a: &Tensor, m: &Tensor, g: usize, sh: usize, sc: usize, eps: f64) -> Result<(Tensor, Tensor)> {
    let out = x.contiguous()?.apply_op3_no_bwd(&a.contiguous()?, &m.contiguous()?, &Ada { op: AdaOp::ResLnMod { g, sh, sc }, eps: eps as f32 })?;
    Ok((out.get(0)?, out.get(1)?))
}

/// `x + f·m[g]`.
pub fn gate_res(x: &Tensor, f: &Tensor, m: &Tensor, g: usize) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op3_no_bwd(&f.contiguous()?, &m.contiguous()?, &Ada { op: AdaOp::GateRes { g }, eps: 0.0 })?)
}

// ---- cuBLASLt linears with the bias (and GELU) in the epilogue -------------------------------
//
// `y = x Wᵀ + b` (and `gelu_tanh(x Wᵀ + b)`) in f16 with f32 accumulation, in one GEMM: candle's
// Linear adds the bias in a second pass over the output, which for Idol's 24k-row activations
// costs about as much as the GEMM itself. Idol-local on purpose: the shared QLinear is untouched.

mod lt_gemm {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::Mutex;

    use candle_core::cuda_backend::cudarc::cublaslt::{result as lt, sys};
    use candle_core::cuda_backend::cudarc::driver::CudaSlice;

    pub const WORKSPACE: usize = 4 << 20;

    pub struct State {
        pub handle: sys::cublasLtHandle_t,
        pub plans: HashMap<(usize, usize, usize, bool, [u32; 4]), Plan>,
        pub workspaces: HashMap<usize, CudaSlice<u8>>,
        pub device: candle_core::cuda_backend::DeviceId,
        pub owner: candle_core::CudaDevice,
    }

    // SAFETY: every use of the handle and of the plans' descriptors is serialized by STATE.
    unsafe impl Send for State {}
    pub static STATE: Mutex<Option<State>> = Mutex::new(None);

    impl Drop for State {
        fn drop(&mut self) {
            let _ = self.owner.cuda_stream().context().bind_to_thread();
            unsafe {
                let _ = lt::destroy_handle(self.handle);
            }
        }
    }

    pub struct Plan {
        pub mm: sys::cublasLtMatmulDesc_t,
        pub layouts: [sys::cublasLtMatrixLayout_t; 3],
        pub pref: sys::cublasLtMatmulPreference_t,
        pub algo: sys::cublasLtMatmulAlgo_t,
    }

    impl Drop for Plan {
        fn drop(&mut self) {
            unsafe {
                for l in self.layouts {
                    let _ = lt::destroy_matrix_layout(l);
                }
                let _ = lt::destroy_matmul_pref(self.pref);
                let _ = lt::destroy_matmul_desc(self.mm);
            }
        }
    }

    pub fn err(e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Msg(format!("Idol cuBLASLt: {e:?}"))
    }

    /// `D (n×m, col-major) = op(W) · X` with `W (k×n, col-major, transposed)` and `X (k×m)`, f16,
    /// f32 accumulation, the bias (per output feature) and optionally GELU in the epilogue.
    pub fn plan(m: usize, n: usize, k: usize, gelu: bool, alignments: [u32; 4], handle: sys::cublasLtHandle_t) -> candle_core::Result<Plan> {
        let mm = lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F).map_err(err)?;
        let f16 = sys::cudaDataType::CUDA_R_16F;
        let layouts = [
            lt::create_matrix_layout(f16, k as u64, n as u64, k as i64).map_err(err)?,
            lt::create_matrix_layout(f16, k as u64, m as u64, k as i64).map_err(err)?,
            lt::create_matrix_layout(f16, n as u64, m as u64, n as i64).map_err(err)?,
        ];
        let pref = lt::create_matmul_pref().map_err(err)?;
        // Owned from here on, so the descriptors are freed on every error path below.
        // SAFETY: cublasLtMatmulAlgo_t is plain data; the zeroed value is replaced before use.
        let mut plan = Plan { mm, layouts, pref, algo: unsafe { std::mem::zeroed() } };
        let transpose: i32 = 1;
        let epilogue = if gelu { sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS } else { sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS };
        let bias_type = f16;
        unsafe {
            let set = |attr, p: *const c_void, size: usize| lt::set_matmul_desc_attribute(mm, attr, p, size).map_err(err);
            set(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &transpose as *const _ as *const c_void, std::mem::size_of_val(&transpose))?;
            set(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue as *const _ as *const c_void, std::mem::size_of_val(&epilogue))?;
            set(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type as *const _ as *const c_void, std::mem::size_of_val(&bias_type))?;
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
        let [a, b, d] = layouts;
        let heuristic = unsafe { lt::get_matmul_algo_heuristic(handle, mm, a, b, d, d, pref) }.map_err(err)?;
        plan.algo = heuristic.algo;
        Ok(plan)
    }
}

struct LtLinear {
    gelu: bool,
}

impl candle_core::CustomOp3 for LtLinear {
    fn name(&self) -> &'static str {
        "idol-lt-linear"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("idol-lt-linear is CUDA-only")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout, b: &CudaStorage, bl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::cublaslt::{result as lt, sys};
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if wk != k || bl.shape().elem_count() != n {
            candle_core::bail!("idol-lt-linear: x {:?}, w {:?}, b {:?}", xl.shape(), wl.shape(), bl.shape());
        }
        let dev = x.device().clone();
        let stream = dev.cuda_stream();
        let out = unsafe { dev.alloc::<half::f16>(m * n)? };
        {
            let xp = ptr::<half::f16>(x, xl, "x", &stream)?;
            let wp = ptr::<half::f16>(w, wl, "weight", &stream)?;
            let bp = ptr::<half::f16>(b, bl, "bias", &stream)?;
            let (op, _go) = out.device_ptr(&stream);
            let alignments = [wp, xp, op, op].map(|p| 1u32 << p.trailing_zeros().min(8));
            let key = (m, n, k, self.gelu, alignments);
            let mut guard = lt_gemm::STATE.lock().map_err(lt_gemm::err)?;
            if guard.as_ref().is_none_or(|s| s.device != dev.id()) {
                *guard = Some(lt_gemm::State {
                    handle: lt::create_handle().map_err(lt_gemm::err)?,
                    plans: std::collections::HashMap::new(),
                    workspaces: std::collections::HashMap::new(),
                    device: dev.id(),
                    owner: dev.clone(),
                });
            }
            stream.context().bind_to_thread().map_err(lt_gemm::err)?;
            let state = guard.as_mut().ok_or_else(|| lt_gemm::err("missing handle"))?;
            if !state.plans.contains_key(&key) {
                let p = lt_gemm::plan(m, n, k, self.gelu, alignments, state.handle)?;
                state.plans.insert(key, p);
            }
            let stream_key = stream.cu_stream() as usize;
            if let std::collections::hash_map::Entry::Vacant(slot) = state.workspaces.entry(stream_key) {
                slot.insert(unsafe { dev.alloc::<u8>(lt_gemm::WORKSPACE)? });
            }
            let plan = state.plans.get(&key).ok_or_else(|| lt_gemm::err("missing plan"))?;
            let work = state.workspaces.get(&stream_key).ok_or_else(|| lt_gemm::err("missing workspace"))?;
            let (scratch, _gs) = work.device_ptr(&stream);
            let [a, bl_, d] = plan.layouts;
            let (alpha, beta) = (1f32, 0f32);
            unsafe {
                let bias = bp as *const c_void;
                lt::set_matmul_desc_attribute(
                    plan.mm,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &bias as *const _ as *const c_void,
                    std::mem::size_of_val(&bias),
                )
                .map_err(lt_gemm::err)?;
                lt::matmul(
                    state.handle,
                    plan.mm,
                    &alpha as *const _ as *const c_void,
                    &beta as *const _ as *const c_void,
                    wp as *const c_void,
                    a,
                    xp as *const c_void,
                    bl_,
                    op as *const c_void,
                    d,
                    op as *mut c_void,
                    d,
                    &plan.algo,
                    scratch as *mut c_void,
                    lt_gemm::WORKSPACE,
                    stream.cu_stream() as sys::cudaStream_t,
                )
                .map_err(lt_gemm::err)?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, n).into()))
    }
}

/// `x (M, K) Wᵀ + b` for `weight (N, K)` and `bias (N)`, all f16 on CUDA, then the tanh GELU when
/// `gelu`: one cuBLASLt GEMM with the bias (and GELU) in its epilogue. `None` when the shapes or
/// dtypes do not fit (the caller then runs the plain ops).
pub fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor, gelu: bool) -> Result<Option<Tensor>> {
    let fits = x.device().is_cuda()
        && [x.dtype(), weight.dtype(), bias.dtype()].iter().all(|&d| d == DType::F16)
        && x.rank() == 2
        && weight.dim(0)?.is_multiple_of(8)
        && weight.dim(1)?.is_multiple_of(8);
    if !fits {
        return Ok(None);
    }
    Ok(Some(x.contiguous()?.apply_op3_no_bwd(&weight.contiguous()?, &bias.contiguous()?, &LtLinear { gelu })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    /// The fused AdaLN ops against the plain ops the block runs (candle's LayerNorm, broadcast
    /// mul/add), in f32. Skipped without CUDA.
    #[test]
    fn adaln_ops_match_the_plain_ops() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (b, l, c) = (2, 53, 1024);
        let x = Tensor::randn(0f32, 1.0, (b, l, c), &dev)?.to_dtype(DType::F16)?;
        let a = Tensor::randn(0f32, 1.0, (b, l, c), &dev)?.to_dtype(DType::F16)?;
        let m = Tensor::randn(0f32, 0.3, (b, 6 * c), &dev)?.to_dtype(DType::F16)?;
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        let chunk = |j: usize| -> Result<Tensor> { Ok(f(&m)?.narrow(1, j * c, c)?.unsqueeze(1)?) };
        let ln = |t: &Tensor| -> Result<Tensor> {
            let mean = t.mean_keepdim(D::Minus1)?;
            let d = t.broadcast_sub(&mean)?;
            Ok(d.broadcast_div(&(d.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?)
        };
        let modulate = |t: &Tensor, sh: usize, sc: usize| -> Result<Tensor> { Ok(ln(t)?.broadcast_mul(&(chunk(sc)? + 1.0)?)?.broadcast_add(&chunk(sh)?)?) };
        let check = |name: &str, got: &Tensor, want: &Tensor, tol: f32| -> Result<()> {
            let err = (f(got)? - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
            eprintln!("adaln {name}: max abs err {err} (max |y| {scale})");
            assert!(err <= scale * tol, "{name}: max abs err {err} against max |y| {scale}");
            Ok(())
        };
        check("ln_mod", &ln_mod(&x, &m, 0, 1, 1e-6)?, &modulate(&f(&x)?, 0, 1)?, 2e-3)?;
        let x2 = (f(&x)? + f(&a)?.broadcast_mul(&chunk(2)?)?)?;
        let (gx, gh) = res_ln_mod(&x, &a, &m, 2, 3, 4, 1e-6)?;
        check("res_ln_mod x", &gx, &x2, 2e-3)?;
        check("res_ln_mod h", &gh, &modulate(&f(&gx)?, 3, 4)?, 2e-3)?;
        check("gate_res", &gate_res(&x, &a, &m, 5)?, &(f(&x)? + f(&a)?.broadcast_mul(&chunk(5)?)?)?, 2e-3)?;
        Ok(())
    }

    /// Each fused AdaLN op alone at Idol's joint-block shape (CFG batch 2, 12294 tokens, 1024).
    #[test]
    #[ignore = "exclusive CUDA benchmark"]
    fn adaln_ops_time() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (b, l, c) = (2, 12294, 1024);
        let x = Tensor::randn(0f32, 1.0, (b, l, c), &dev)?.to_dtype(DType::F16)?;
        let a = Tensor::randn(0f32, 1.0, (b, l, c), &dev)?.to_dtype(DType::F16)?;
        let m = Tensor::randn(0f32, 0.3, (b, 6 * c), &dev)?.to_dtype(DType::F16)?;
        let mut bench = |name: &str, f: &mut dyn FnMut() -> Result<Tensor>| -> Result<()> {
            for _ in 0..3 {
                f()?;
            }
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..50 {
                f()?;
            }
            dev.synchronize()?;
            eprintln!("{name:<12} {:.3} ms", t.elapsed().as_secs_f64() * 1000.0 / 50.0);
            Ok(())
        };
        bench("ln_mod", &mut || ln_mod(&x, &m, 0, 1, 1e-6))?;
        bench("res_ln_mod", &mut || Ok(res_ln_mod(&x, &a, &m, 2, 3, 4, 1e-6)?.1))?;
        bench("gate_res", &mut || gate_res(&x, &a, &m, 5))?;
        bench("plain add", &mut || Ok((&x + &a)?))?;
        Ok(())
    }

    /// The cuBLASLt linear (bias, and bias + GELU) against candle's matmul, bias add and GELU in
    /// f32. Skipped without CUDA.
    #[test]
    fn lt_linear_matches_the_plain_ops() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (m, k, n) = (777, 256, 384);
        let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
        let w = (Tensor::randn(0f32, 1.0, (n, k), &dev)? / (k as f64).sqrt())?.to_dtype(DType::F16)?;
        let b = Tensor::randn(0f32, 0.5, n, &dev)?.to_dtype(DType::F16)?;
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        let want = f(&x)?.matmul(&f(&w)?.t()?)?.broadcast_add(&f(&b)?)?;
        for gelu in [false, true] {
            let got = linear_bias(&x, &w, &b, gelu)?.expect("fits");
            let want = if gelu { want.gelu()? } else { want.clone() };
            let err = (f(&got)? - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
            eprintln!("lt linear gelu={gelu}: max abs err {err} (max |y| {scale})");
            assert!(err <= scale * 4e-3, "gelu={gelu}: max abs err {err} against max |y| {scale}");
        }
        Ok(())
    }

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
