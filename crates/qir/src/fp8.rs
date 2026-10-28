//! FP8 (e4m3) linears, run the way ComfyUI runs its `fp8_scaled` repacks on Ada and Blackwell
//! cards (`--fast fp8_matrix_mult`, `comfy/ops.py` `mixed_precision_ops` with
//! `comfy_kitchen`'s `TensorCoreFP8Layout`): the weight stays in the file's e4m3 bytes with its
//! per-tensor `weight_scale`, and per layer either
//!
//! - **matmul in fp8**: the activation clamped to ±448 and cast to e4m3 with scale 1 (the
//!   checkpoints carry no `input_scale`), one cuBLASLt FP8 GEMM with the two per-tensor scales,
//!   fp32 accumulation (`use_fast_accum=False`), bf16 out; or
//! - **full precision**: the layers the file's `_quantization_metadata` marks
//!   `full_precision_matrix_mult` (Krea 2's attention gate, `wo` and the SwiGLU's `down`) are
//!   dequantized to bf16 (`q.to(bf16) * scale.to(bf16)`) for each call and multiplied in bf16.
//!
//! The weights stay 1 byte a parameter in memory either way (Krea 2's DiT: 13 GB instead of
//! the 24 GB bf16 copy), so the encoder stays resident beside it. A text encoder's layers
//! (marked one by one in `comfy_quant` records) all take the full-precision path, as ComfyUI
//! runs them when it encodes a prompt (it switches to fp8 matmuls only to generate text). On
//! the CPU, or when cuBLASLt declines a shape, the fp8 layers take the full-precision path too.

use std::collections::HashSet;
use std::path::Path;

use candle_core::{DType, Tensor};

use crate::Result;

/// E4M3's largest finite value: activations are clamped to it before the cast.
pub const E4M3_MAX: f64 = 448.0;

/// What a transformer file says about its fp8 layers.
#[derive(Clone, Debug, Default)]
pub struct Fp8Plan {
    /// Layer prefixes (`blocks.0.attn.wo`) the file marks `full_precision_matrix_mult`.
    pub full_precision: HashSet<String>,
}

impl Fp8Plan {
    /// Read the files' headers: `Some` when any tensor is an fp8 weight with a `weight_scale`
    /// beside it (ComfyUI's `fp8_scaled` repacks), with the layers its metadata keeps in full
    /// precision; `None` for a bf16 or prequantized ConvRot file.
    pub fn from_files(files: &[impl AsRef<Path>]) -> Result<Option<Self>> {
        let mut found = false;
        let mut plan = Self::default();
        for f in files {
            let (names, meta) = header(f.as_ref())?;
            found |= names.iter().any(|n| n.ends_with(".weight_scale"));
            if let Some(q) = meta.and_then(|m| m.get("_quantization_metadata").cloned()) {
                plan.full_precision.extend(full_precision_layers(&q));
            }
        }
        Ok(found.then_some(plan))
    }

    /// Whether `prefix` (a layer's tensor name without `.weight`) is kept in full precision.
    /// The marks matter: Krea 2 with every layer an fp8 matmul is 0.11 s a step faster and 6 of
    /// 28 pictures at ComfyUI's parity gate instead of 21 (`benchmarks.md`, 2026-10-27).
    pub fn is_full_precision(&self, prefix: &str) -> bool {
        self.full_precision.contains(prefix)
    }
}

/// The layers a `_quantization_metadata` JSON marks `full_precision_matrix_mult`.
fn full_precision_layers(json: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        tracing::warn!("unreadable _quantization_metadata: every fp8 layer runs as an fp8 matmul");
        return Vec::new();
    };
    let Some(layers) = v.get("layers").and_then(|l| l.as_object()) else {
        return Vec::new();
    };
    layers.iter().filter(|(_, conf)| conf.get("full_precision_matrix_mult").and_then(|b| b.as_bool()).unwrap_or(false)).map(|(name, _)| name.clone()).collect()
}

/// A safetensors header: tensor names and `__metadata__`.
type Header = (Vec<String>, Option<std::collections::HashMap<String, String>>);

/// A safetensors file's tensor names and `__metadata__`, from its header alone.
fn header(path: &Path) -> Result<Header> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    anyhow::ensure!(n < 256 << 20, "{}: a {n}-byte safetensors header", path.display());
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    let v: serde_json::Value = serde_json::from_slice(&buf)?;
    let obj = v.as_object().ok_or_else(|| anyhow::anyhow!("{}: the header is not an object", path.display()))?;
    let names = obj.keys().filter(|k| *k != "__metadata__").cloned().collect();
    let meta =
        obj.get("__metadata__").and_then(|m| m.as_object()).map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect());
    Ok((names, meta))
}

/// The layer under `vb` as an [`Fp8Linear`] when the file holds it in e4m3 (a `weight` with a
/// `weight_scale` beside it), else `None`. Whether it is kept in full precision comes from its
/// own `comfy_quant` record when the file has one (the text-encoder repacks: a JSON string in a
/// U8 tensor), else from the file-wide `plan` (the DiT repacks' `_quantization_metadata`).
pub fn load_linear(in_c: usize, out_c: usize, plan: Option<&Fp8Plan>, vb: &candle_nn::VarBuilder) -> Result<Option<Fp8Linear>> {
    if !vb.contains_tensor("weight_scale") {
        return Ok(None);
    }
    let w = vb.get_unchecked_dtype("weight", DType::F8E4M3)?;
    anyhow::ensure!(w.dims2()? == (out_c, in_c), "{}: an fp8 weight of shape {:?}, expected ({out_c}, {in_c})", vb.prefix(), w.dims());
    let scale = vb.get_unchecked_dtype("weight_scale", DType::F32)?;
    let full = if vb.contains_tensor("comfy_quant") {
        let bytes = vb.get_unchecked_dtype("comfy_quant", DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
        comfy_quant_full_precision(&bytes)
    } else {
        plan.is_some_and(|p| p.is_full_precision(&vb.prefix()))
    };
    Ok(Some(Fp8Linear::new(w, &scale, full)?))
}

/// A `comfy_quant` record's `full_precision_matrix_mult` (false when absent or unreadable).
fn comfy_quant_full_precision(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes).ok().and_then(|v| v.get("full_precision_matrix_mult").and_then(|b| b.as_bool())).unwrap_or(false)
}

/// One fp8 linear `y = x Wᵀ` (no bias: the fp8 layers of the repacks have none).
pub struct Fp8Linear {
    /// `(out, in)` in e4m3.
    w: Tensor,
    /// The per-tensor weight scale.
    scale: f32,
    /// `[scale, 1]` in f32 on the device: cuBLASLt's A and B scale pointers.
    #[cfg_attr(not(feature = "fp8"), allow(dead_code))]
    scales: Tensor,
    /// Dequantize and multiply in bf16 (the file's `full_precision_matrix_mult`); without the
    /// `fp8` kernels every layer does.
    #[cfg_attr(not(feature = "fp8"), allow(dead_code))]
    full: bool,
}

/// Whether this build can run e4m3 weights on `dev` (the `fp8` feature's kernels on CUDA; the
/// CPU always can, through candle's casts).
pub fn supported_on(dev: &candle_core::Device) -> bool {
    !dev.is_cuda() || cfg!(feature = "fp8")
}

impl Fp8Linear {
    /// From the file's e4m3 `weight` and f32 `weight_scale`.
    pub fn new(w: Tensor, weight_scale: &Tensor, full: bool) -> Result<Self> {
        anyhow::ensure!(w.dtype() == DType::F8E4M3, "an fp8 linear's weight must be e4m3, got {:?}", w.dtype());
        let scale =
            weight_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied().ok_or_else(|| anyhow::anyhow!("an empty weight_scale"))?;
        let scales = Tensor::new(&[scale, 1.0f32], w.device())?;
        Ok(Self { w, scale, scales, full })
    }

    /// Full-precision callers never use cuBLASLt scale pointers; keep them on the host.
    pub fn new_full_precision(w: Tensor, weight_scale: &Tensor) -> Result<Self> {
        anyhow::ensure!(w.dtype() == DType::F8E4M3, "an fp8 linear's weight must be e4m3, got {:?}", w.dtype());
        let scale =
            weight_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied().ok_or_else(|| anyhow::anyhow!("an empty weight_scale"))?;
        let scales = Tensor::new(&[scale, 1.0f32], &candle_core::Device::Cpu)?;
        Ok(Self { w, scale, scales, full: true })
    }

    /// This layer multiplying in bf16 whatever its file asked.
    pub fn full_precision(self) -> Self {
        Self { full: true, ..self }
    }

    /// The weight in `dtype`, ComfyUI's dequantization (`q.to(bf16) * scale.to(bf16)`).
    pub fn dequantized(&self, dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "fp8")]
        if self.w.device().is_cuda() {
            return Ok(cuda::dequant(&self.w, self.scale)?.to_dtype(dtype)?);
        }
        let s = half::bf16::from_f32(self.scale).to_f64();
        Ok(self.w.to_dtype(DType::F32)?.to_dtype(DType::BF16)?.affine(s, 0.0)?.to_dtype(dtype)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (n, k) = self.w.dims2()?;
        let x2 = x.reshape(((), k))?;
        let flops = 2.0 * x2.dim(0)? as f64 * (n * k) as f64;
        #[cfg(feature = "fp8")]
        if !self.full && x2.device().is_cuda() && x.dtype() == DType::BF16 {
            match crate::profile::scope("fp8 matmul", flops, x2.device(), || cuda::fp8_matmul(&x2, &self.w, &self.scales)) {
                Ok(y) => return reshape_out(y, &dims, n),
                Err(e) => tracing::debug!(error = %e, "fp8 matmul declined: the layer runs in bf16"),
            }
        }
        let y = crate::profile::scope("fp8 dequantized + bf16 matmul", flops, x2.device(), || Ok(x2.matmul(&self.dequantized(x.dtype())?.t()?)?))?;
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

#[cfg(feature = "fp8")]
mod cuda {
    //! The casts (kernels/fp8/fp8_cast.cu) and the cuBLASLt FP8 GEMM. Row-major `X (M,K)` and
    //! `W (N,K)` are column-major `(K,M)` and `(K,N)`; `Y (M,N)` row-major is column-major
    //! `(N,M)` = `op(W) · X` with `op = T`, the "TN" layout cuBLASLt's FP8 kernels require.

    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::Mutex;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::cublaslt::{result as lt, sys};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{CpuStorage, CudaStorage, Layout, Shape, Tensor};

    use crate::layout::dense_byte_offset;

    const WORKSPACE: usize = 32 << 20;

    extern "C" {
        fn qir_fp8_dequant_bf16(out: *mut c_void, q: *const c_void, scale: f32, n: usize, stream: *mut c_void) -> i32;
        fn qir_bf16_quant_fp8(out: *mut c_void, x: *const c_void, n: usize, stream: *mut c_void) -> i32;
    }

    /// e4m3 → bf16 times the scale.
    struct Dequant(f32);

    impl candle_core::CustomOp1 for Dequant {
        fn name(&self) -> &'static str {
            "fp8-dequant"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("fp8-dequant is CUDA-only")
        }

        fn cuda_fwd(&self, q: &CudaStorage, l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = q.device().clone();
            let n = l.shape().elem_count();
            let qo = dense_byte_offset::<u8>(l, "fp8-dequant q")?;
            let qs = q.as_cuda_slice::<float8::F8E4M3>()?;
            let stream = dev.cuda_stream();
            let out = unsafe { dev.alloc::<half::bf16>(n)? };
            {
                let (qp, _gq) = qs.device_ptr(&stream);
                let (op, _go) = out.device_ptr(&stream);
                let rc = unsafe { qir_fp8_dequant_bf16(op as *mut c_void, (qp + qo) as *const c_void, self.0, n, stream.cu_stream() as *mut c_void) };
                if rc != 0 {
                    candle_core::bail!("fp8-dequant failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), l.shape().clone()))
        }
    }

    /// bf16 → e4m3, clamped to ±448.
    struct Quant;

    impl candle_core::CustomOp1 for Quant {
        fn name(&self) -> &'static str {
            "fp8-quant"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("fp8-quant is CUDA-only")
        }

        fn cuda_fwd(&self, x: &CudaStorage, l: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = x.device().clone();
            let n = l.shape().elem_count();
            let xo = dense_byte_offset::<half::bf16>(l, "fp8-quant x")?;
            let xs = x.as_cuda_slice::<half::bf16>()?;
            let stream = dev.cuda_stream();
            let out = unsafe { dev.alloc::<float8::F8E4M3>(n)? };
            {
                let (xp, _gx) = xs.device_ptr(&stream);
                let (op, _go) = out.device_ptr(&stream);
                let rc = unsafe { qir_bf16_quant_fp8(op as *mut c_void, (xp + xo) as *const c_void, n, stream.cu_stream() as *mut c_void) };
                if rc != 0 {
                    candle_core::bail!("fp8-quant failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), l.shape().clone()))
        }
    }

    /// A dense view whose start the vector kernels can read (16-byte aligned), copying if not.
    fn aligned(t: &Tensor, elem: usize) -> candle_core::Result<Tensor> {
        let t = t.contiguous()?;
        if (t.layout().start_offset() * elem).is_multiple_of(16) {
            Ok(t)
        } else {
            t.copy()
        }
    }

    pub(super) fn dequant(w: &Tensor, scale: f32) -> crate::Result<Tensor> {
        Ok(aligned(w, 1)?.apply_op1_no_bwd(&Dequant(scale))?)
    }

    /// The process's cuBLASLt handle and the algorithm chosen per `(m, n, k)`.
    struct Lt {
        handle: sys::cublasLtHandle_t,
        algos: HashMap<(usize, usize, usize), sys::cublasLtMatmulAlgo_t>,
    }
    // SAFETY: the handle is only used under the mutex below.
    unsafe impl Send for Lt {}

    static LT: Mutex<Option<Lt>> = Mutex::new(None);

    pub(super) fn fp8_matmul(x: &Tensor, w: &Tensor, scales: &Tensor) -> crate::Result<Tensor> {
        let xq = aligned(x, 2)?.apply_op1_no_bwd(&Quant)?;
        Ok(xq.apply_op3(&w.contiguous()?, scales, Fp8Gemm)?)
    }

    struct Fp8Gemm;

    /// Descriptors destroyed when the call ends, whatever happens.
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

    fn err(what: &str, e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Msg(format!("fp8 gemm: {what}: {e:?}"))
    }

    impl candle_core::CustomOp3 for Fp8Gemm {
        fn name(&self) -> &'static str {
            "fp8-gemm"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("fp8-gemm is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            x_l: &Layout,
            w: &CudaStorage,
            w_l: &Layout,
            s: &CudaStorage,
            s_l: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = x.device().clone();
            let (m, k) = x_l.shape().dims2()?;
            let (n, k2) = w_l.shape().dims2()?;
            if k != k2 {
                candle_core::bail!("fp8-gemm K mismatch: {k} vs {k2}");
            }
            if !k.is_multiple_of(16) || !n.is_multiple_of(16) {
                candle_core::bail!("fp8-gemm needs K and N in multiples of 16 ({k}, {n})");
            }
            let xo = dense_byte_offset::<u8>(x_l, "fp8-gemm x")?;
            let wo = dense_byte_offset::<u8>(w_l, "fp8-gemm w")?;
            let so = dense_byte_offset::<f32>(s_l, "fp8-gemm scales")?;
            let xs = x.as_cuda_slice::<float8::F8E4M3>()?;
            let ws = w.as_cuda_slice::<float8::F8E4M3>()?;
            let ss = s.as_cuda_slice::<f32>()?;
            let stream = dev.cuda_stream();
            let out = unsafe { dev.alloc::<half::bf16>(m * n)? };
            let work = unsafe { dev.alloc::<u8>(WORKSPACE)? };
            {
                let (xp, _gx) = xs.device_ptr(&stream);
                let (wp, _gw) = ws.device_ptr(&stream);
                let (sp, _gs) = ss.device_ptr(&stream);
                let (op, _go) = out.device_ptr(&stream);
                let (kp, _gk) = work.device_ptr(&stream);
                let (xp, wp, sp) = (xp + xo, wp + wo, sp + so);
                if !xp.is_multiple_of(16) || !wp.is_multiple_of(16) {
                    candle_core::bail!("fp8-gemm: misaligned view (x {xp:#x}, w {wp:#x})");
                }
                let a_scale = sp;
                let b_scale = sp + 4;

                let mut guard = LT.lock().unwrap_or_else(|e| e.into_inner());
                if guard.is_none() {
                    *guard = Some(Lt { handle: lt::create_handle().map_err(|e| err("create handle", e))?, algos: HashMap::new() });
                }
                let Some(state) = guard.as_mut() else { candle_core::bail!("fp8-gemm: no cuBLASLt handle") };

                let mm =
                    lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType::CUDA_R_32F).map_err(|e| err("matmul desc", e))?;
                let mut desc = Desc { mm, layouts: Vec::new(), pref: None };
                // cublasOperation_t: CUBLAS_OP_N = 0, CUBLAS_OP_T = 1.
                let (t, nn): (i32, i32) = (1, 0);
                let fast_accum: i8 = 0;
                unsafe {
                    let set = |attr, p: *const c_void, size: usize| lt::set_matmul_desc_attribute(desc.mm, attr, p, size).map_err(|e| err("desc attribute", e));
                    set(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &t as *const _ as *const c_void, std::mem::size_of_val(&t))?;
                    set(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &nn as *const _ as *const c_void, std::mem::size_of_val(&nn))?;
                    set(
                        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                        &a_scale as *const _ as *const c_void,
                        std::mem::size_of_val(&a_scale),
                    )?;
                    set(
                        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                        &b_scale as *const _ as *const c_void,
                        std::mem::size_of_val(&b_scale),
                    )?;
                    set(
                        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                        &fast_accum as *const _ as *const c_void,
                        std::mem::size_of_val(&fast_accum),
                    )?;
                }
                let layout =
                    |ty, rows: usize, cols: usize, ld: usize| lt::create_matrix_layout(ty, rows as u64, cols as u64, ld as i64).map_err(|e| err("layout", e));
                let a_l = layout(sys::cudaDataType::CUDA_R_8F_E4M3, k, n, k)?;
                desc.layouts.push(a_l);
                let b_l = layout(sys::cudaDataType::CUDA_R_8F_E4M3, k, m, k)?;
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
                let (alpha, beta) = (1.0f32, 0.0f32);
                unsafe {
                    lt::matmul(
                        state.handle,
                        desc.mm,
                        &alpha as *const f32 as *const c_void,
                        &beta as *const f32 as *const c_void,
                        wp as *const c_void,
                        a_l,
                        xp as *const c_void,
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
    fn the_metadata_names_the_full_precision_layers() {
        let json = r#"{"layers": {"blocks.0.attn.wo": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true}, "blocks.0.attn.wq": {"format": "float8_e4m3fn"}}}"#;
        assert_eq!(full_precision_layers(json), vec!["blocks.0.attn.wo".to_string()]);
        assert!(full_precision_layers("not json").is_empty());
        assert!(comfy_quant_full_precision(br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}"#));
        assert!(!comfy_quant_full_precision(br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": false}"#));
        assert!(!comfy_quant_full_precision(b"\xff"));
    }

    #[test]
    fn a_full_precision_layer_is_the_dequantized_matmul() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let w = Tensor::new(&[[1.0f32, -2.0], [0.5, 4.0]], &dev)?.to_dtype(DType::F8E4M3)?;
        let l = Fp8Linear::new(w, &Tensor::new(&[0.5f32], &dev)?, true)?;
        // f32 activations: candle's CPU matmul has no bf16.
        let x = Tensor::new(&[[[1.0f32, 1.0]]], &dev)?;
        let y = l.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(y, vec![-0.5, 2.25]);
        Ok(())
    }
    #[test]
    #[cfg(feature = "fp8")]
    fn host_only_scales_preserve_full_precision_cuda() -> Result<()> {
        let dev = candle_core::Device::new_cuda(0)?;
        let w = Tensor::randn(0f32, 0.5, (128, 128), &candle_core::Device::Cpu)?.to_dtype(DType::F8E4M3)?.to_device(&dev)?;
        let scale = Tensor::new(&[0.321f32], &candle_core::Device::Cpu)?;
        let old = Fp8Linear::new(w.clone(), &scale, true)?;
        let new = Fp8Linear::new_full_precision(w, &scale)?;
        assert!(new.scales.device().is_cpu());
        let x = Tensor::randn(0f32, 0.5, (2, 37, 128), &dev)?.to_dtype(DType::BF16)?;
        let diff = (old.forward(&x)?.to_dtype(DType::F32)? - new.forward(&x)?.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);
        Ok(())
    }
}
