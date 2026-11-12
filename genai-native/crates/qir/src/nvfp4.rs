//! NVFP4 linears: 4-bit e2m1 weights with an e4m3 scale for each 16 values and one f32 global
//! scale, as ComfyUI's `nvfp4` repacks store them (`weight` U8 `(out, in/2)` with element `2i`
//! in the high nibble, `weight_scale`
//! e4m3 in cuBLAS's swizzled 128×4 tiles, `weight_scale_2` F32), run the way ComfyUI runs them on
//! a Blackwell card (comfy_kitchen's `TensorCoreNVFP4Layout`):
//!
//! - **matmul in FP4** (RTX 50-series and other compute 10.x/12.x cards): the activation is
//!   quantized to NVFP4 for each call (global scale `amax / (448·6)`, kernels/fp8/nvfp4.cu) and
//!   one cuBLASLt block-scaled FP4 GEMM runs with fp32 accumulation, bf16 out;
//! - **full precision** everywhere else (an older card, the CPU, a layer the file marks
//!   `full_precision_matrix_mult`, a shape cuBLASLt declines): the weight is dequantized to
//!   bf16 for each call and multiplied in bf16.
//!
//! Either way the weight stays half a byte a parameter in memory: Krea 2's DiT is 7.2 GB
//! instead of the fp8 file's 12.2 GB. The 4-bit grid is coarse (about 13 % layer error against
//! INT8's 1.2 % at MOSS's shapes, the rebuild thread's bench of 2026-10-29), so the app offers it
//! only for the Fast and Balanced qualities; Best keeps the full models.

use candle_core::{DType, Device, Tensor};

use crate::Result;

/// E2M1's magnitudes by code (the sign is bit 3).
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Bytes of the swizzled block-scale buffer for a `(rows, cols)` matrix: rows padded to 128,
/// scale columns (`cols / 16`) to 4.
pub fn scale_bytes(rows: usize, cols: usize) -> usize {
    rows.div_ceil(128) * 128 * (cols / 16).div_ceil(4) * 4
}

/// Where block scale `(row, col)` sits in the swizzled buffer (cuBLAS's 1D block-scaling layout).
pub fn swizzled(row: usize, col: usize, cols: usize) -> usize {
    let col_tiles = (cols / 16).div_ceil(4);
    let tile = (row / 128) * col_tiles + col / 4;
    let r = row % 128;
    tile * 512 + (r % 32) * 16 + (r / 32) * 4 + col % 4
}

/// The nearest e2m1 code, ties to the even code, saturating at ±6 (the kernel's rounding).
pub fn e2m1_code(x: f32) -> u8 {
    let a = x.abs();
    let c = if a.is_nan() || a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    if x < 0.0 && c != 0 {
        c | 8
    } else {
        c
    }
}

pub fn e2m1_value(code: u8) -> f32 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 {
        -v
    } else {
        v
    }
}

/// Whether this process's card can run the FP4 GEMM: a Blackwell card (compute 10.x or 12.x)
/// and a build with the kernels. Asked once.
pub fn gemm_supported() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| cfg!(feature = "fp8") && crate::device::probe().is_ok_and(|g| g.compute_cap >= 100))
}

/// Whether the layer under `vb` is held in NVFP4 (it has the global scale).
pub fn is_nvfp4(vb: &candle_nn::VarBuilder) -> bool {
    vb.contains_tensor("weight_scale_2")
}

/// The layer under `vb` as an [`Nvfp4Linear`] when the file holds it in NVFP4, else `None`.
/// `full` keeps it in full precision (the file's `full_precision_matrix_mult`).
pub fn load_linear(in_c: usize, out_c: usize, full: bool, vb: &candle_nn::VarBuilder) -> Result<Option<Nvfp4Linear>> {
    if !is_nvfp4(vb) {
        return Ok(None);
    }
    let w = vb.get_unchecked_dtype("weight", DType::U8)?;
    anyhow::ensure!(w.dims2()? == (out_c, in_c / 2), "{}: an NVFP4 weight of shape {:?}, expected ({out_c}, {})", vb.prefix(), w.dims(), in_c / 2);
    let scales = vb.get_unchecked_dtype("weight_scale", DType::F8E4M3)?.flatten_all()?;
    let global = vb.get_unchecked_dtype("weight_scale_2", DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied().unwrap_or(1.0);
    Ok(Some(Nvfp4Linear::new(w, scales, global, full).map_err(|e| anyhow::anyhow!("{}: {e}", vb.prefix()))?))
}

/// One NVFP4 linear `y = x Wᵀ` (no bias: the repacks' NVFP4 layers have none).
pub struct Nvfp4Linear {
    /// `(out, in/2)`: two e2m1 values a byte, element `2i` in the high nibble (Comfy-Org's
    /// files, checked against the fp8 file on 2026-11-11: 9.3 % from it read this way, 142 %
    /// the other). The activations are packed the same way, so the GEMM, which pairs the
    /// nibbles of both operands alike, sums the same products whichever order the tensor cores
    /// read.
    w: Tensor,
    /// The e4m3 block scales, flat, swizzled ([`scale_bytes`] of them).
    scales: Tensor,
    global: f32,
    full: bool,
}

impl Nvfp4Linear {
    pub fn new(w: Tensor, scales: Tensor, global: f32, full: bool) -> Result<Self> {
        let (n, half) = w.dims2()?;
        let k = half * 2;
        anyhow::ensure!(
            w.dtype() == DType::U8 && scales.dtype() == DType::F8E4M3,
            "NVFP4 wants U8 data and e4m3 scales, got {:?} and {:?}",
            w.dtype(),
            scales.dtype()
        );
        anyhow::ensure!(k.is_multiple_of(16), "an NVFP4 width of {k}, not a multiple of 16");
        let scales = scales.flatten_all()?;
        anyhow::ensure!(
            scales.elem_count() == scale_bytes(n, k),
            "{} block scales for a ({n}, {k}) weight, expected {} (swizzled, padded to 128×4 tiles)",
            scales.elem_count(),
            scale_bytes(n, k)
        );
        Ok(Self { w, scales, global, full })
    }

    /// `(out, in)`.
    pub fn shape(&self) -> Result<(usize, usize)> {
        let (n, half) = self.w.dims2()?;
        Ok((n, half * 2))
    }

    pub fn is_full_precision(&self) -> bool {
        self.full
    }

    /// The weight in `dtype`.
    pub fn dequantized(&self, dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "fp8")]
        if self.w.device().is_cuda() {
            return Ok(cuda::dequant(&self.w, &self.scales, self.global, Layout::default())?.to_dtype(dtype)?);
        }
        dequant_cpu(&self.w, &self.scales, self.global, Layout::default())?.to_dtype(dtype).map_err(Into::into)
    }

    /// The weight read with another layout (checking a file: [`Layout`]).
    pub fn dequantized_as(&self, layout: Layout) -> Result<Tensor> {
        #[cfg(feature = "fp8")]
        if self.w.device().is_cuda() {
            return cuda::dequant(&self.w, &self.scales, self.global, layout);
        }
        dequant_cpu(&self.w, &self.scales, self.global, layout)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (n, k) = self.shape()?;
        let x2 = x.reshape(((), k))?;
        let flops = 2.0 * x2.dim(0)? as f64 * (n * k) as f64;
        #[cfg(feature = "fp8")]
        if !self.full && x2.device().is_cuda() && x.dtype() == DType::BF16 && gemm_supported() && cuda::usable() {
            match crate::profile::scope("nvfp4 matmul", flops, x2.device(), || cuda::matmul(&x2, &self.w, &self.scales, self.global)) {
                Ok(y) => return reshape_out(y, &dims, n),
                Err(e) => {
                    tracing::warn!(error = %e, "NVFP4 matmul declined: NVFP4 layers run dequantized in bf16 from now on");
                    cuda::decline();
                }
            }
        }
        let y = crate::profile::scope("nvfp4 dequantized + bf16 matmul", flops, x2.device(), || Ok(x2.matmul(&self.dequantized(x.dtype())?.t()?)?))?;
        reshape_out(y, &dims, n)
    }
}

fn reshape_out(y: Tensor, dims: &[usize], n: usize) -> Result<Tensor> {
    let mut out = dims.to_vec();
    if let Some(last) = out.last_mut() {
        *last = n;
    }
    Ok(y.reshape(out)?)
}

/// How the bytes are read: the default is ComfyUI's layout; the others exist to check a file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// Block scales as a plain `(rows, cols/16)` matrix instead of swizzled tiles.
    pub linear_scales: bool,
    /// Element `2i` in the low nibble (ComfyUI's files hold it in the high one).
    pub low_first: bool,
}

/// [`Nvfp4Linear::dequantized`] on the CPU, in f32.
fn dequant_cpu(w: &Tensor, scales: &Tensor, global: f32, layout: Layout) -> Result<Tensor> {
    let (n, half) = w.dims2()?;
    let k = half * 2;
    let q = w.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let s = scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for col in 0..k / 16 {
            let si = if layout.linear_scales { row * (k / 16) + col } else { swizzled(row, col, k) };
            let scale = s[si] * global;
            for j in 0..8 {
                let byte = q[(row * k + col * 16) / 2 + j];
                let (a, b) = if layout.low_first { (byte & 15, byte >> 4) } else { (byte >> 4, byte & 15) };
                out[row * k + col * 16 + 2 * j] = e2m1_value(a) * scale;
                out[row * k + col * 16 + 2 * j + 1] = e2m1_value(b) * scale;
            }
        }
    }
    Ok(Tensor::from_vec(out, (n, k), &Device::Cpu)?.to_device(w.device())?)
}

/// Quantize a `(rows, cols)` weight to NVFP4 on the CPU, as comfy_kitchen's `quantize_nvfp4`
/// does (global `amax / (448·6)`): the packed data `(rows, cols/2)` U8, the swizzled e4m3 block
/// scales (flat) and the global scale. For converting a model's own weights once (and tests).
pub fn quantize_cpu(w: &Tensor) -> Result<(Tensor, Tensor, f32)> {
    let (rows, cols) = w.dims2()?;
    anyhow::ensure!(cols.is_multiple_of(16), "an NVFP4 width of {cols}, not a multiple of 16");
    let v = w.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let global = if amax > 0.0 { amax / (448.0 * 6.0) } else { 1.0 };
    let mut q = vec![0u8; rows * cols / 2];
    let mut s = vec![0f32; scale_bytes(rows, cols)];
    for row in 0..rows {
        for col in 0..cols / 16 {
            let block = &v[row * cols + col * 16..row * cols + col * 16 + 16];
            let bmax = block.iter().fold(0f32, |m, x| m.max(x.abs()));
            let scale = float8::F8E4M3::from_f32((bmax / 6.0 / global).min(448.0)).to_f32();
            s[swizzled(row, col, cols)] = scale;
            let inv = if scale > 0.0 { 1.0 / (scale * global) } else { 0.0 };
            for j in 0..8 {
                q[(row * cols + col * 16) / 2 + j] = (e2m1_code(block[2 * j] * inv) << 4) | e2m1_code(block[2 * j + 1] * inv);
            }
        }
    }
    let q = Tensor::from_vec(q, (rows, cols / 2), &Device::Cpu)?;
    let s = Tensor::from_vec(s, scale_bytes(rows, cols), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
    Ok((q, s, global))
}

#[cfg(feature = "fp8")]
mod cuda {
    //! The casts (kernels/fp8/nvfp4.cu) and cuBLASLt's block-scaled FP4 GEMM, in the "TN"
    //! layout of src/fp8.rs: row-major `X (M,K)` and `W (N,K)` are column-major `(K,M)` and
    //! `(K,N)`, `Y (M,N)` row-major is column-major `(N,M)` = `op(W) · X` with `op = T`.

    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::cublaslt::{result as lt, sys};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{CpuStorage, CudaStorage, Shape, Tensor};

    use crate::layout::dense_byte_offset;

    const WORKSPACE: usize = 32 << 20;
    /// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`: an e4m3 scale for each 16 values.
    const SCALE_VEC16_UE4M3: i32 = 1;
    /// `CUBLASLT_POINTER_MODE_DEVICE`: alpha and beta are read from the device.
    const POINTER_MODE_DEVICE: i32 = 1;

    extern "C" {
        fn qir_nvfp4_quant(
            q: *mut c_void,
            scales: *mut c_void,
            params: *mut c_void,
            x: *const c_void,
            rows: usize,
            cols: usize,
            weight_global: f32,
            stream: *mut c_void,
        ) -> i32;
        fn qir_nvfp4_dequant(
            out: *mut c_void,
            q: *const c_void,
            scales: *const c_void,
            rows: usize,
            cols: usize,
            global: f32,
            linear: i32,
            low_first: i32,
            stream: *mut c_void,
        ) -> i32;
    }

    static DECLINED: AtomicBool = AtomicBool::new(false);

    /// False once cuBLASLt has declined an FP4 GEMM in this process (an older driver or cuBLAS).
    pub(super) fn usable() -> bool {
        !DECLINED.load(Ordering::Relaxed)
    }

    pub(super) fn decline() {
        DECLINED.store(true, Ordering::Relaxed);
    }

    fn aligned(t: &Tensor, elem: usize) -> candle_core::Result<Tensor> {
        let t = t.contiguous()?;
        if (t.layout().start_offset() * elem).is_multiple_of(16) {
            Ok(t)
        } else {
            t.copy()
        }
    }

    fn err(what: &str, e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Msg(format!("nvfp4 gemm: {what}: {e:?}"))
    }

    /// e2m1 + scales → bf16.
    struct Dequant {
        global: f32,
        layout: super::Layout,
    }

    impl candle_core::CustomOp2 for Dequant {
        fn name(&self) -> &'static str {
            "nvfp4-dequant"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &candle_core::Layout, _: &CpuStorage, _: &candle_core::Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("nvfp4-dequant is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            q: &CudaStorage,
            q_l: &candle_core::Layout,
            s: &CudaStorage,
            s_l: &candle_core::Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = q.device().clone();
            let (n, half) = q_l.shape().dims2()?;
            let k = half * 2;
            let qo = dense_byte_offset::<u8>(q_l, "nvfp4-dequant q")?;
            let so = dense_byte_offset::<u8>(s_l, "nvfp4-dequant scales")?;
            let qs = q.as_cuda_slice::<u8>()?;
            let ss = s.as_cuda_slice::<float8::F8E4M3>()?;
            let stream = dev.cuda_stream();
            let out = unsafe { dev.alloc::<half::bf16>(n * k)? };
            {
                let (qp, _gq) = qs.device_ptr(&stream);
                let (sp, _gs) = ss.device_ptr(&stream);
                let (op, _go) = out.device_ptr(&stream);
                let rc = unsafe {
                    qir_nvfp4_dequant(
                        op as *mut c_void,
                        (qp + qo) as *const c_void,
                        (sp + so) as *const c_void,
                        n,
                        k,
                        self.global,
                        self.layout.linear_scales as i32,
                        self.layout.low_first as i32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("nvfp4-dequant failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), (n, k).into()))
        }
    }

    pub(super) fn dequant(w: &Tensor, scales: &Tensor, global: f32, layout: super::Layout) -> crate::Result<Tensor> {
        Ok(aligned(w, 1)?.apply_op2_no_bwd(&scales.contiguous()?, &Dequant { global, layout })?)
    }

    struct Lt {
        handle: sys::cublasLtHandle_t,
        algos: HashMap<(usize, usize, usize), sys::cublasLtMatmulAlgo_t>,
    }
    // SAFETY: the handle is only used under the mutex below.
    unsafe impl Send for Lt {}

    static LT: Mutex<Option<Lt>> = Mutex::new(None);

    pub(super) fn matmul(x: &Tensor, w: &Tensor, scales: &Tensor, global: f32) -> crate::Result<Tensor> {
        Ok(aligned(x, 2)?.apply_op3(&w.contiguous()?, &scales.contiguous()?, Gemm { global })?)
    }

    /// Quantize the activation and run the GEMM, in one op: `x` bf16 `(M,K)`, `w` U8 `(N,K/2)`,
    /// `s` the weight's swizzled e4m3 scales.
    struct Gemm {
        global: f32,
    }

    struct Desc {
        mm: sys::cublasLtMatmulDesc_t,
        layouts: Vec<sys::cublasLtMatrixLayout_t>,
        pref: Option<sys::cublasLtMatmulPreference_t>,
    }

    impl Drop for Desc {
        fn drop(&mut self) {
            unsafe {
                for l in self.layouts.drain(..) {
                    let _ = lt::destroy_matrix_layout(l);
                }
                if let Some(p) = self.pref.take() {
                    let _ = lt::destroy_matmul_pref(p);
                }
                let _ = lt::destroy_matmul_desc(self.mm);
            }
        }
    }

    impl candle_core::CustomOp3 for Gemm {
        fn name(&self) -> &'static str {
            "nvfp4-gemm"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &candle_core::Layout,
            _: &CpuStorage,
            _: &candle_core::Layout,
            _: &CpuStorage,
            _: &candle_core::Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("nvfp4-gemm is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            x_l: &candle_core::Layout,
            w: &CudaStorage,
            w_l: &candle_core::Layout,
            s: &CudaStorage,
            s_l: &candle_core::Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = x.device().clone();
            let (m, k) = x_l.shape().dims2()?;
            let (n, half) = w_l.shape().dims2()?;
            if half * 2 != k {
                candle_core::bail!("nvfp4-gemm K mismatch: {k} vs {}", half * 2);
            }
            if !k.is_multiple_of(64) || !n.is_multiple_of(32) {
                candle_core::bail!("nvfp4-gemm needs K in multiples of 64 and N of 32 ({k}, {n})");
            }
            let xo = dense_byte_offset::<half::bf16>(x_l, "nvfp4-gemm x")?;
            let wo = dense_byte_offset::<u8>(w_l, "nvfp4-gemm w")?;
            let so = dense_byte_offset::<u8>(s_l, "nvfp4-gemm scales")?;
            let xs = x.as_cuda_slice::<half::bf16>()?;
            let ws = w.as_cuda_slice::<u8>()?;
            let ss = s.as_cuda_slice::<float8::F8E4M3>()?;
            let stream = dev.cuda_stream();
            let xq = unsafe { dev.alloc::<u8>(m * k / 2)? };
            let xsc = unsafe { dev.alloc::<u8>(super::scale_bytes(m, k))? };
            let params = unsafe { dev.alloc::<f32>(4)? };
            let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
            let work = unsafe { dev.alloc::<u8>(WORKSPACE)? };
            {
                let (xp, _gx) = xs.device_ptr(&stream);
                let (wp, _gw) = ws.device_ptr(&stream);
                let (sp, _gs) = ss.device_ptr(&stream);
                let (qp, _gq) = xq.device_ptr(&stream);
                let (qsp, _gqs) = xsc.device_ptr(&stream);
                let (pp, _gp) = params.device_ptr(&stream);
                let (op, _go) = out.device_ptr(&stream);
                let (kp, _gk) = work.device_ptr(&stream);
                let (xp, wp, sp) = (xp + xo, wp + wo, sp + so);
                if !wp.is_multiple_of(16) || !sp.is_multiple_of(16) {
                    candle_core::bail!("nvfp4-gemm: misaligned view (w {wp:#x}, scales {sp:#x})");
                }
                let rc = unsafe {
                    qir_nvfp4_quant(
                        qp as *mut c_void,
                        qsp as *mut c_void,
                        pp as *mut c_void,
                        xp as *const c_void,
                        m,
                        k,
                        self.global,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("nvfp4 activation quant failed, rc={rc}");
                }
                let (alpha, beta) = (pp + 4, pp + 8);

                let mut guard = LT.lock().unwrap_or_else(|e| e.into_inner());
                if guard.is_none() {
                    *guard = Some(Lt { handle: lt::create_handle().map_err(|e| err("create handle", e))?, algos: HashMap::new() });
                }
                let Some(state) = guard.as_mut() else { candle_core::bail!("nvfp4-gemm: no cuBLASLt handle") };

                let mm =
                    lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F).map_err(|e| err("matmul desc", e))?;
                let mut desc = Desc { mm, layouts: Vec::new(), pref: None };
                let (t, nn): (i32, i32) = (1, 0);
                unsafe {
                    let set = |attr, p: *const c_void, size: usize| lt::set_matmul_desc_attribute(desc.mm, attr, p, size).map_err(|e| err("desc attribute", e));
                    let i32_attr = |attr, v: &i32| set(attr, v as *const i32 as *const c_void, std::mem::size_of::<i32>());
                    let ptr_attr = |attr, v: &u64| set(attr, v as *const u64 as *const c_void, std::mem::size_of::<u64>());
                    use sys::cublasLtMatmulDescAttributes_t as A;
                    i32_attr(A::CUBLASLT_MATMUL_DESC_TRANSA, &t)?;
                    i32_attr(A::CUBLASLT_MATMUL_DESC_TRANSB, &nn)?;
                    i32_attr(A::CUBLASLT_MATMUL_DESC_POINTER_MODE, &POINTER_MODE_DEVICE)?;
                    i32_attr(A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                    i32_attr(A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                    ptr_attr(A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sp)?;
                    ptr_attr(A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &qsp)?;
                }
                let layout =
                    |ty, rows: usize, cols: usize, ld: usize| lt::create_matrix_layout(ty, rows as u64, cols as u64, ld as i64).map_err(|e| err("layout", e));
                let a_l = layout(sys::cudaDataType::CUDA_R_4F_E2M1, k, n, k)?;
                desc.layouts.push(a_l);
                let b_l = layout(sys::cudaDataType::CUDA_R_4F_E2M1, k, m, k)?;
                desc.layouts.push(b_l);
                let d_l = layout(sys::cudaDataType::CUDA_R_16BF, n, m, n)?;
                desc.layouts.push(d_l);

                let algo = match state.algos.get(&(m, n, k)) {
                    Some(a) => *a,
                    None => {
                        let pref = lt::create_matmul_pref().map_err(|e| err("preference", e))?;
                        desc.pref = Some(pref);
                        let ws = WORKSPACE;
                        unsafe {
                            lt::set_matmul_pref_attribute(
                                pref,
                                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                &ws as *const _ as *const c_void,
                                std::mem::size_of_val(&ws),
                            )
                            .map_err(|e| err("preference attribute", e))?;
                        }
                        let h = unsafe { lt::get_matmul_algo_heuristic(state.handle, desc.mm, a_l, b_l, d_l, d_l, pref) }
                            .map_err(|e| err("no algorithm for this shape", e))?;
                        state.algos.insert((m, n, k), h.algo);
                        h.algo
                    }
                };
                unsafe {
                    lt::matmul(
                        state.handle,
                        desc.mm,
                        alpha as *const c_void,
                        beta as *const c_void,
                        wp as *const c_void,
                        a_l,
                        qp as *const c_void,
                        b_l,
                        op as *const c_void,
                        d_l,
                        op as *mut c_void,
                        d_l,
                        &algo,
                        kp as *mut c_void,
                        WORKSPACE,
                        stream.cu_stream() as sys::cudaStream_t,
                    )
                    .map_err(|e| err("matmul", e))?;
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), (m, n).into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_rounds_to_nearest_even_and_saturates() {
        let cases = [
            (0.0, 0.0),
            (0.25, 0.0),
            (0.26, 0.5),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.3, 1.5),
            (1.75, 2.0),
            (2.5, 2.0),
            (2.6, 3.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.1, 6.0),
            (100.0, 6.0),
        ];
        for (x, want) in cases {
            assert_eq!(e2m1_value(e2m1_code(x)), want, "{x}");
            assert_eq!(e2m1_value(e2m1_code(-x)), -want, "-{x}");
        }
        for c in 0u8..16 {
            if c != 8 {
                assert_eq!(e2m1_code(e2m1_value(c)), c);
            }
        }
    }

    #[test]
    fn the_swizzle_is_a_permutation_of_each_tile() {
        let (rows, cols) = (256, 16 * 8);
        let mut seen = vec![false; scale_bytes(rows, cols)];
        for r in 0..rows {
            for c in 0..cols / 16 {
                let i = swizzled(r, c, cols);
                assert!(!seen[i], "({r}, {c}) lands on {i} twice");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
        // Row 32 of a tile follows row 0's four scales; row 1 starts the next 16.
        assert_eq!(swizzled(32, 0, cols), 4);
        assert_eq!(swizzled(1, 0, cols), 16);
        assert_eq!(swizzled(0, 4, cols), 512);
    }

    #[test]
    fn a_quantized_weight_comes_back_within_the_4_bit_grid() -> Result<()> {
        let w = Tensor::randn(0f32, 1.0, (128, 256), &Device::Cpu)?;
        let (q, s, g) = quantize_cpu(&w)?;
        let l = Nvfp4Linear::new(q, s, g, false)?;
        let back = l.dequantized(DType::F32)?;
        let err = ((&back - &w)?.sqr()?.sum_all()?.to_scalar::<f32>()? / w.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
        // NVFP4 on Gaussian weights: about 9-10 % relative RMS error.
        assert!(err < 0.13, "relative error {err}");
        // And the full-precision forward is the dequantized matmul.
        let x = Tensor::randn(0f32, 1.0, (3, 256), &Device::Cpu)?;
        let y = l.forward(&x)?;
        let want = x.matmul(&back.t()?)?;
        assert!((y - want)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-3);
        Ok(())
    }

    /// The kernels against the CPU reference, and the FP4 GEMM against the dequantized matmul
    /// (on a Blackwell card).
    #[test]
    #[cfg(feature = "fp8")]
    fn the_cuda_path_matches_the_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let w = Tensor::randn(0f32, 0.05, (512, 1024), &Device::Cpu)?;
        let (q, s, g) = quantize_cpu(&w)?;
        let cpu = Nvfp4Linear::new(q.clone(), s.clone(), g, false)?.dequantized(DType::F32)?;
        let l = Nvfp4Linear::new(q.to_device(&dev)?, s.to_device(&dev)?, g, false)?;
        let gpu = l.dequantized(DType::F32)?.to_device(&Device::Cpu)?;
        let d = (&gpu - &cpu)?.abs()?.max_all()?.to_scalar::<f32>()?;
        // bf16 output against f32.
        assert!(d < 2e-3, "dequant differs by {d}");
        if gemm_supported() {
            let x = Tensor::randn(0f32, 1.0, (300, 1024), &dev)?.to_dtype(DType::BF16)?;
            let y = l.forward(&x)?.to_dtype(DType::F32)?;
            let want = x.to_dtype(DType::F32)?.matmul(&cpu.to_device(&dev)?.t()?)?;
            let rel = ((&y - &want)?.sqr()?.sum_all()?.to_scalar::<f32>()? / want.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
            // Only the activation's 4-bit rounding separates them (~10 %).
            assert!(rel < 0.15, "FP4 GEMM relative error {rel}");
            assert!(cuda::usable(), "cuBLASLt declined the FP4 GEMM");
        }
        Ok(())
    }
}
