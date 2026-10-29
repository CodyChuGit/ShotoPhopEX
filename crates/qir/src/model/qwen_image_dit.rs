//! Shared Layered/Edit-2511 Qwen-Image dual-stream blocks, without sampler state.
//! Checkpoint builders are unprefixed ComfyUI model keys; CPU uses f32, CUDA bf16.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::Result;

#[cfg(feature = "cuda")]
#[path = "layered_gemm.rs"]
mod cuda_gemm;

/// Map immutable checkpoints once; keep quantization metadata on the host to avoid upload fences.
pub fn checkpoint_builder(path: impl AsRef<std::path::Path>, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: installed checkpoints must not be modified while a split is running.
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(VarBuilder::from_backend(
        Box::new(Checkpoint(tensors, std::collections::BTreeMap::new())),
        if device.is_cuda() { DType::BF16 } else { DType::F32 },
        device.clone(),
    ))
}

struct Checkpoint(candle_core::safetensors::MmapedSafetensors, std::collections::BTreeMap<String, Tensor>);

/// Overlay a bound CPU FP8 snapshot when staged residency reloads the same adapters.
pub fn checkpoint_builder_patched(
    path: impl AsRef<std::path::Path>,
    device: &Device,
    patches: std::collections::BTreeMap<String, Tensor>,
) -> Result<VarBuilder<'static>> {
    // SAFETY: installed checkpoints remain immutable during inference.
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(VarBuilder::from_backend(Box::new(Checkpoint(tensors, patches)), if device.is_cuda() { DType::BF16 } else { DType::F32 }, device.clone()))
}

impl candle_nn::var_builder::SimpleBackend for Checkpoint {
    fn get(&self, shape: candle_core::Shape, name: &str, _: candle_nn::Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("{name}: expected {shape:?}, got {:?}", tensor.shape());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let host = name.ends_with(".comfy_quant") || name.ends_with(".weight_scale");
        let device = if host { &Device::Cpu } else { device };
        if let Some(tensor) = self.1.get(name) {
            return tensor.to_device(device)?.to_dtype(dtype);
        }
        self.0.load(name, device)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.get(name).is_ok()
    }
}

pub const WIDTH: usize = 3072;
pub const TEXT_WIDTH: usize = 3584;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const BLOCKS: usize = 60;

/// Biased mixed-file linear. FP8 multiplication follows the checkpoint's metadata.
struct LayeredLinear {
    dense: Option<candle_nn::Linear>,
    fp8: Option<(Tensor, Tensor)>,
    /// On CUDA, the same fp8 weight through qir's own dequantization kernel (candle has no CUDA
    /// cast from e4m3), multiplied in bf16 as the file's `full_precision_matrix_mult` asks.
    gpu: Option<crate::fp8::Fp8Linear>,
    /// `QIR_LAYERED_FP8_GEMM=1` (an experiment, off by default): the same weight through the
    /// e4m3 × e4m3 tensor-core GEMM instead of the file's full-precision bf16 multiply.
    fast: Option<crate::fp8::Fp8Linear>,
    /// `QIR_LAYERED_INT8=1` (an experiment, off by default): the fp8 weight dequantized once and
    /// requantized to ConvRot INT8 (so double-quantized), replacing the fp8 copy on the device.
    #[cfg(feature = "convrot")]
    int8: Option<crate::convrot::ConvRotLinear>,
    bias: Tensor,
    original: Option<(Tensor, Option<Tensor>)>,
    #[cfg(feature = "cuda")]
    lt: Option<std::sync::Arc<cuda_gemm::Handle>>,
}

impl LayeredLinear {
    fn load(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let fp8 = if vb.contains_tensor("weight_scale") {
            let bytes = vb.get_unchecked_dtype("comfy_quant", DType::U8)?.to_vec1::<u8>()?;
            let quant: serde_json::Value = serde_json::from_slice(&bytes)?;
            anyhow::ensure!(quant["full_precision_matrix_mult"] == true && quant["format"] == "float8_e4m3fn", "Layered requires full-precision E4M3 matmuls");
            let weight = vb.get_unchecked_dtype("weight", DType::F8E4M3)?;
            anyhow::ensure!(weight.dims2()? == (output, input), "incorrect Layered linear weight shape");
            let scale = vb.get_unchecked_dtype("weight_scale", DType::F32)?.reshape(())?;
            Some((weight, scale))
        } else {
            None
        };
        let dense = if fp8.is_none() { Some(candle_nn::linear_no_bias(input, output, vb.clone())?) } else { None };
        let gpu = match &fp8 {
            Some((w, s)) if w.device().is_cuda() => {
                anyhow::ensure!(crate::fp8::supported_on(w.device()), "Layered's fp8 weights on CUDA need the `fp8` feature's kernels");
                Some({
                    let linear = crate::fp8::Fp8Linear::new_full_precision(w.clone(), s)?;
                    if vb.contains_tensor("input_scale") {
                        linear.precise_scale()
                    } else {
                        linear
                    }
                })
            }
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let lt = if gpu.is_some() && std::env::var("QIR_LAYERED_GEMM").map_or(true, |v| v != "cublas") { Some(cuda_gemm::handle(vb.device())?) } else { None };
        let fast = fast_fp8(&fp8)?;
        #[cfg(feature = "convrot")]
        let (int8, fp8, gpu) = match gpu {
            Some(l) if layered_int8() && input.is_multiple_of(crate::model::rotation::GROUP) && output.is_multiple_of(8) => {
                let r = crate::model::rotation::regular_hadamard_256(vb.device())?;
                (Some(crate::convrot::ConvRotLinear::from_weight(&l.dequantized(DType::BF16)?, &r)?), None, None)
            }
            gpu => (None, fp8, gpu),
        };
        Ok(Self {
            original: None,
            dense,
            fp8,
            gpu,
            fast,
            #[cfg(feature = "convrot")]
            int8,
            bias: vb.get(output, "bias")?,
            #[cfg(feature = "cuda")]
            lt,
        })
    }

    fn bind(&mut self, module: &str, pairs: &[crate::lora::LoraPair]) -> Result<()> {
        if pairs.is_empty() && self.original.is_none() {
            return Ok(());
        }
        #[cfg(feature = "convrot")]
        anyhow::ensure!(self.int8.is_none(), "{module}: LoRAs are not supported with QIR_LAYERED_INT8=1");
        let device = self.bias.device();
        if self.original.is_none() {
            let (w, scale) = if let Some((w, s)) = &self.fp8 {
                (w.to_device(&Device::Cpu)?, Some(s.clone()))
            } else {
                (self.dense.as_ref().ok_or_else(|| anyhow::anyhow!("missing weight"))?.weight().to_device(&Device::Cpu)?, None)
            };
            self.original = Some((w, scale));
        }
        let (original, scale) = self.original.as_ref().ok_or_else(|| anyhow::anyhow!("missing original weight"))?;
        let original = original.to_device(device)?;
        let mut weight = if let Some(scale) = scale {
            if device.is_cuda() {
                crate::fp8::Fp8Linear::new_full_precision(original.clone(), scale)?.precise_scale().dequantized(self.bias.dtype())?
            } else {
                original.to_dtype(self.bias.dtype())?.broadcast_mul(&scale.to_dtype(self.bias.dtype())?)?
            }
        } else {
            original.clone()
        };
        for pair in pairs {
            weight = pair.merge_weight(weight)?;
        }
        if let Some(scale) = scale {
            let (w, scale) = if pairs.is_empty() {
                (original, scale.clone())
            } else {
                crate::fp8::requantize_full_precision(&weight, crate::edit2511_pipeline::key_seed(module))?
            };
            self.gpu = if device.is_cuda() { Some(crate::fp8::Fp8Linear::new_full_precision(w.clone(), &scale)?.precise_scale()) } else { None };
            self.fp8 = Some((w, scale));
            self.fast = fast_fp8(&self.fp8)?;
        } else {
            self.dense = Some(candle_nn::Linear::new(weight, None));
        }
        Ok(())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_impl(x, false)
    }

    // Wider accumulation limits CPU error amplified by the edit model's large MLP gates.
    fn forward_precise(&self, x: &Tensor) -> Result<Tensor> {
        if !x.device().is_cpu() || x.dtype() != DType::F32 {
            return self.forward(x);
        }
        let weight = match (&self.dense, &self.fp8) {
            (Some(dense), _) => dense.weight().clone(),
            (_, Some((weight, scale))) => weight.to_dtype(DType::F32)?.broadcast_mul(scale)?,
            _ => anyhow::bail!("missing DiT linear weight"),
        };
        let mut shape = x.dims().to_vec();
        let (output, input) = weight.dims2()?;
        if let Some(last) = shape.last_mut() {
            *last = output;
        }
        let y = x.reshape(((), input))?.to_dtype(DType::F64)?.matmul(&weight.to_dtype(DType::F64)?.t()?)?.to_dtype(DType::F32)?.reshape(shape)?;
        linear_bias(y, &self.bias, false)
    }

    fn forward_gelu_precise(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() && x.dtype() == DType::F32 {
            return Ok(self.forward_precise(x)?.gelu()?);
        }
        self.forward_gelu(x)
    }

    fn forward_gelu(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_impl(x, true)
    }

    fn forward_impl(&self, x: &Tensor, gelu: bool) -> Result<Tensor> {
        #[cfg(feature = "convrot")]
        if let Some(c) = &self.int8 {
            let flops = 2.0 * x.elem_count() as f64 * self.bias.elem_count() as f64;
            let y = crate::profile::scope("layered int8 convrot", flops, x.device(), || c.forward(x))?;
            return linear_bias(y, &self.bias, gelu);
        }
        if let Some(f) = &self.fast {
            let k = x.dim(D::Minus1)?;
            if x.dtype() == DType::BF16 && x.elem_count() / k >= 1024 {
                // The GEMM casts x to e4m3 at scale 1, which saturates Qwen-Image's activations
                // (the first measurement came out as noise): scale x's largest magnitude to e4m3's
                // 448 first and the output back. Two wide reductions, not one over the tensor.
                let rows = x.reshape(((), k))?;
                let s = (rows.abs()?.max_keepdim(1)?.max_keepdim(0)?.to_dtype(DType::F32)? / 448.0)?.maximum(1e-12)?.to_dtype(x.dtype())?;
                let y = f.forward(&x.broadcast_div(&s.reshape(())?)?)?.broadcast_mul(&s.reshape(())?)?;
                return linear_bias(y, &self.bias, gelu);
            }
        }
        if let Some(l) = &self.gpu {
            #[cfg(feature = "cuda")]
            if let Some(lt) = &self.lt {
                let k = x.dim(D::Minus1)?;
                if x.dtype() == DType::BF16 && x.elem_count() / k >= 1024 {
                    let flops = 2.0 * x.elem_count() as f64 * self.bias.elem_count() as f64;
                    let y = crate::profile::scope("layered bf16 Lt", flops, x.device(), || cuda_gemm::forward(x, &l.dequantized(x.dtype())?, lt))?;
                    return linear_bias(y, &self.bias, gelu);
                }
            }
            return linear_bias(l.forward(x)?, &self.bias, gelu);
        }
        let y = match (&self.dense, &self.fp8) {
            (Some(dense), _) => crate::profile::scope("layered dense linear", 0.0, x.device(), || Ok(dense.forward(x)?))?,
            (_, Some((weight, scale))) => {
                // Comfy's eager dequantization casts BOTH operands to the compute dtype.
                // The shared fp8 reader rounds scales to bf16 even for CPU f32 inputs.
                let weight = weight.to_dtype(x.dtype())?.broadcast_mul(&scale.to_dtype(x.dtype())?)?;
                let mut shape = x.dims().to_vec();
                let output = weight.dim(0)?;
                let input = weight.dim(1)?;
                if let Some(last) = shape.last_mut() {
                    *last = output;
                }
                x.reshape(((), input))?.matmul(&weight.t()?)?.reshape(shape)?
            }
            _ => anyhow::bail!("missing Layered linear weight"),
        };
        linear_bias(y, &self.bias, gelu)
    }
}

/// `QIR_LAYERED_INT8=1`: ConvRot INT8 for the Layered DiT's fp8 linears (see `LayeredLinear::int8`).
#[cfg(feature = "convrot")]
fn layered_int8() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_LAYERED_INT8").is_ok_and(|v| v.trim() == "1"))
}

/// The e4m3 tensor-core linear for a CUDA fp8 weight when `QIR_LAYERED_FP8_GEMM=1`.
fn fast_fp8(fp8: &Option<(Tensor, Tensor)>) -> Result<Option<crate::fp8::Fp8Linear>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("QIR_LAYERED_FP8_GEMM").is_ok_and(|v| v.trim() == "1"));
    match fp8 {
        Some((w, s)) if on && w.device().is_cuda() && cfg!(feature = "fp8") => Ok(Some(crate::fp8::Fp8Linear::new(w.clone(), s, false)?)),
        _ => Ok(None),
    }
}

/// Reuse a fresh GEMM allocation; preserve separate bf16 GEMM/bias rounding.
fn linear_bias(y: Tensor, bias: &Tensor, gelu: bool) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    if y.device().is_cuda() && y.dtype() == DType::BF16 {
        crate::profile::scope("layered bias", 0.0, y.device(), || Ok(y.inplace_op2(bias, &cuda_ops::Bias { gelu })?))?;
        return Ok(y);
    }
    let y = y.broadcast_add(bias)?;
    Ok(if gelu { y.gelu()? } else { y })
}

/// Non-affine LayerNorm accumulates in f32, including on CUDA.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    crate::profile::scope("layered layer_norm", 0.0, x.device(), || layer_norm_impl(x))
}

fn layer_norm_impl(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() && x.dtype() == DType::BF16 {
        let n = x.dim(D::Minus1)?;
        return Ok(candle_nn::ops::layer_norm(x, &Tensor::ones(n, x.dtype(), x.device())?, &Tensor::zeros(n, x.dtype(), x.device())?, 1e-6)?);
    }
    let f = x.to_dtype(DType::F32)?;
    let centered = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
    Ok(centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?)
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    crate::profile::scope("layered rms_norm", 0.0, x.device(), || rms_norm_impl(x, weight))
}

fn rms_norm_impl(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() && x.dtype() == DType::BF16 {
        let n = x.dim(D::Minus1)?;
        let normalized = candle_nn::ops::rms_norm(x, &Tensor::ones(n, x.dtype(), x.device())?, 1e-6)?;
        return Ok(normalized.broadcast_mul(weight)?);
    }
    let f = x.to_dtype(DType::F32)?;
    // PyTorch RMSNorm casts the normalized result before multiplying the weight.
    Ok(f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?.broadcast_mul(weight)?)
}

/// Match addcmul's single bf16 rounding at modulation and gated residual boundaries.
fn add_product(base: &Tensor, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    crate::profile::scope("layered add_product", 0.0, base.device(), || add_product_impl(base, left, right))
}

fn add_product_impl(base: &Tensor, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    if base.device().is_cuda() && base.dtype() == DType::BF16 && base.rank() == 3 {
        if left.dim(1)? == 1 {
            let gate = left.broadcast_as((base.dim(0)?, 1, base.dim(2)?))?;
            return crate::krea2_fused::gate_add(base, right, &gate, 0);
        }
        if right.dim(1)? == 1 {
            let base = base.broadcast_as(left.shape())?.contiguous()?;
            let gate = right.broadcast_as((base.dim(0)?, 1, base.dim(2)?))?;
            return crate::krea2_fused::gate_add(&base, left, &gate, 0);
        }
    }
    Ok(left.to_dtype(DType::F32)?.broadcast_mul(&right.to_dtype(DType::F32)?)?.broadcast_add(&base.to_dtype(DType::F32)?)?.to_dtype(base.dtype())?)
}

/// Spatial 2x2 patches in (frame,row,column,channel,patch-row,patch-column) order.
/// Padding belongs to the caller; Layered's pipeline supplies even latent dimensions.
pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    anyhow::ensure!(c == 16 && h.is_multiple_of(2) && w.is_multiple_of(2), "Layered patches need 16 channels and even spatial dimensions");
    Ok(x.reshape(&[b, c, t, h / 2, 2, w / 2, 2])?.permute(&[0, 2, 3, 5, 1, 4, 6][..])?.reshape((b, t * (h / 2) * (w / 2), 64))?)
}

/// Inverse of [`patchify`], useful at the velocity boundary.
pub fn unpatchify(x: &Tensor, frames: usize, height: usize, width: usize) -> Result<Tensor> {
    anyhow::ensure!(height.is_multiple_of(2) && width.is_multiple_of(2), "even latent dimensions required");
    let b = x.dim(0)?;
    Ok(x.reshape(&[b, frames, height / 2, width / 2, 16, 2, 2])?.permute(&[0, 4, 1, 2, 5, 3, 6][..])?.reshape((b, 16, frames, height, width))?)
}

/// Centered token-grid positions. A single-frame reference uses index -1, -2, ...;
/// multiple generated frames use 0..frames irrespective of index, like ComfyUI.
pub fn image_positions(batch: usize, frames: usize, rows: usize, cols: usize, index: i32, device: &Device) -> Result<Tensor> {
    anyhow::ensure!(batch > 0 && frames > 0 && rows > 0 && cols > 0, "empty image grid");
    let mut ids = Vec::with_capacity(frames * rows * cols * 3);
    for t in 0..frames {
        for y in 0..rows {
            for x in 0..cols {
                ids.extend([if frames > 1 { t as f32 } else { index as f32 }, y as f32 - (rows / 2) as f32, x as f32 - (cols / 2) as f32]);
            }
        }
    }
    Ok(Tensor::from_vec(ids, (1, frames * rows * cols, 3), device)?.broadcast_as((batch, frames * rows * cols, 3))?.contiguous()?)
}

/// Text starts at the integer half of the larger target token-grid dimension.
pub fn text_positions(batch: usize, tokens: usize, rows: usize, cols: usize, device: &Device) -> Result<Tensor> {
    let start = (rows / 2).max(cols / 2);
    let ids: Vec<f32> = (0..tokens).flat_map(|i| [(start + i) as f32; 3]).collect();
    Ok(Tensor::from_vec(ids, (1, tokens, 3), device)?.broadcast_as((batch, tokens, 3))?.contiguous()?)
}

/// Three-axis adjacent-pair rotations, computed in f64 then stored in the model dtype.
pub struct LayeredRope {
    cos: Tensor,
    sin: Tensor,
}

impl LayeredRope {
    pub fn new(ids: &Tensor, dtype: DType) -> Result<Self> {
        let (b, s, axes) = ids.dims3()?;
        anyhow::ensure!(axes == 3, "RoPE needs three position axes");
        let positions = ids.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut cos = Vec::with_capacity(b * s * 64);
        let mut sin = Vec::with_capacity(b * s * 64);
        for pos in positions.as_chunks::<3>().0 {
            for (axis, dim) in [16, 56, 56].into_iter().enumerate() {
                for i in 0..dim / 2 {
                    let angle = f64::from(pos[axis]) * 10000f64.powf(-(2 * i) as f64 / dim as f64);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (b, 1, s, 64), ids.device())?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (b, 1, s, 64), ids.device())?.to_dtype(dtype)?,
        })
    }

    /// ComfyUI's rotation matrix layout (B,1,S,64,2,2), for fixture inspection.
    pub fn matrices(&self) -> Result<Tensor> {
        let row0 = Tensor::stack(&[&self.cos, &self.sin.neg()?], D::Minus1)?;
        let row1 = Tensor::stack(&[&self.sin, &self.cos], D::Minus1)?;
        Ok(Tensor::stack(&[row0, row1], D::Minus2)?)
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        crate::profile::scope("layered rope", 0.0, x.device(), || self.apply_impl(x))
    }

    fn apply_impl(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return Ok(x.apply_op3_no_bwd(&self.cos, &self.sin, &cuda_ops::Rope)?);
        }
        self.apply_unfused(x)
    }

    fn apply_unfused(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        anyhow::ensure!(d == HEAD_DIM, "Layered RoPE needs 128-wide heads");
        let pairs = x.reshape((b, h, s, 64, 2))?;
        let a = pairs.narrow(4, 0, 1)?.squeeze(4)?;
        let z = pairs.narrow(4, 1, 1)?.squeeze(4)?;
        let even = add_product(&a.broadcast_mul(&self.cos)?, &z, &self.sin.neg()?)?;
        let odd = add_product(&a.broadcast_mul(&self.sin)?, &z, &self.cos)?;
        Ok(Tensor::stack(&[even, odd], 4)?.reshape((b, h, s, d))?)
    }
}

/// ComfyUI Timesteps(256, flip_sin_to_cos=true, scale=1000).
pub fn timestep_projection(timestep: &Tensor) -> Result<Tensor> {
    let frequencies: Vec<f32> = (0..128).map(|i| (-(10000f32).ln() * i as f32 / 128.0).exp()).collect();
    let frequencies = Tensor::from_vec(frequencies, (1, 128), timestep.device())?;
    let phase = (timestep.to_dtype(DType::F32)?.reshape(((), 1))?.broadcast_mul(&frequencies)? * 1000.0)?;
    Ok(Tensor::cat(&[phase.cos()?, phase.sin()?], 1)?)
}

/// Fixed CPU float32 frequencies avoid scalar/vector exp drift before the 1000x phase.
/// These are ComfyUI Timesteps(256)'s exp(-ln(10000)*arange(128)/128) bit patterns.
pub fn edit_timestep_projection(timestep: &Tensor) -> Result<Tensor> {
    if !timestep.device().is_cpu() {
        return timestep_projection(timestep);
    }
    const FREQUENCIES: [u32; 128] = [
        0x3f800000, 0x3f6e39f8, 0x3f5dafd6, 0x3f4e4bac, 0x3f3ff911, 0x3f32a506, 0x3f263de0, 0x3f1ab32b, 0x3f0ff59a, 0x3f05f6ee, 0x3ef953ce, 0x3ee80460,
        0x3ed7e89b, 0x3ec8eb23, 0x3ebaf81b, 0x3eadfcff, 0x3ea1e89b, 0x3e96aaea, 0x3e8c3503, 0x3e827908, 0x3e72d422, 0x3e61f836, 0x3e5247ed, 0x3e43ae7c,
        0x3e361887, 0x3e297409, 0x3e1db03f, 0x3e12bd90, 0x3e088d77, 0x3dfe24df, 0x3dec7fd5, 0x3ddc1464, 0x3dcccccc, 0x3dbe94c7, 0x3db15978, 0x3da50957,
        0x3d99940d, 0x3d8eea6c, 0x3d84fe4c, 0x3d778511, 0x3d6655c1, 0x3d5657e3, 0x3d477640, 0x3d399d18, 0x3d2cba15, 0x3d20bc1b, 0x3d159348, 0x3d0b30cb,
        0x3d0186e2, 0x3cf11177, 0x3ce054d1, 0x3cd0c1a8, 0x3cc2434e, 0x3cb4c691, 0x3ca83989, 0x3c9c8b96, 0x3c91ad39, 0x3c879009, 0x3c7c4d30, 0x3c6ac8e6,
        0x3c5a7bf1, 0x3c4b50b5, 0x3c3d330e, 0x3c301050, 0x3c23d70a, 0x3c187706, 0x3c0de12e, 0x3c040777, 0x3bf5b9ad, 0x3be4aa45, 0x3bd4ca15, 0x3bc6040a,
        0x3bb8449a, 0x3bab7982, 0x3b9f91cc, 0x3b947daf, 0x3b8a2e75, 0x3b80967c, 0x3b6f520d, 0x3b5eb47a, 0x3b4f3e33, 0x3b40dac2, 0x3b33770d, 0x3b270153,
        0x3b1b690d, 0x3b109ed8, 0x3b06946d, 0x3afa78f0, 0x3ae91528, 0x3ad8e675, 0x3ac9d759, 0x3abbd3eb, 0x3aaec98e, 0x3aa2a6f7, 0x3a975c0b, 0x3a8cd9d9,
        0x3a83126e, 0x3a73f1a2, 0x3a6301e3, 0x3a533f24, 0x3a44948a, 0x3a36ee9d, 0x3a2a3b44, 0x3a1e69a2, 0x3a136a14, 0x3a092e01, 0x39ff4fad, 0x39ed95e4,
        0x39dd1721, 0x39cdbd93, 0x39bf74d6, 0x39b229fb, 0x39a5cb61, 0x399a48a0, 0x398f9275, 0x39859aa4, 0x3978a80d, 0x3967648c, 0x395753e2, 0x394860bf,
        0x393a7753, 0x392d852a, 0x39217919, 0x39164326, 0x390bd46d, 0x39021f28, 0x38f22cdd, 0x38e15c8e,
    ];
    let frequencies = Tensor::from_vec(FREQUENCIES.map(f32::from_bits).to_vec(), (1, 128), timestep.device())?;
    let phase = (timestep.to_dtype(DType::F32)?.reshape(((), 1))?.broadcast_mul(&frequencies)? * 1000.0)?;
    Ok(Tensor::cat(&[phase.cos()?, phase.sin()?], 1)?)
}

/// Input projections and learned timestep conditioning, independently loadable for parity.
pub struct LayeredInput {
    img: LayeredLinear,
    txt: LayeredLinear,
    txt_norm: Tensor,
    time1: LayeredLinear,
    time2: LayeredLinear,
    addition: Option<Tensor>,
}

impl LayeredInput {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            img: LayeredLinear::load(64, WIDTH, vb.pp("img_in"))?,
            txt: LayeredLinear::load(TEXT_WIDTH, WIDTH, vb.pp("txt_in"))?,
            txt_norm: vb.pp("txt_norm").get(TEXT_WIDTH, "weight")?,
            time1: LayeredLinear::load(256, WIDTH, vb.pp("time_text_embed.timestep_embedder.linear_1"))?,
            time2: LayeredLinear::load(WIDTH, WIDTH, vb.pp("time_text_embed.timestep_embedder.linear_2"))?,
            addition: if vb.contains_tensor("time_text_embed.addition_t_embedding.weight") {
                Some(vb.pp("time_text_embed.addition_t_embedding").get((2, WIDTH), "weight")?)
            } else {
                None
            },
        })
    }

    pub fn timestep(&self, timestep: &Tensor, additional_t_cond: Option<&Tensor>) -> Result<Tensor> {
        let time = if self.addition.is_none() { edit_timestep_projection(timestep)? } else { timestep_projection(timestep)? };
        let time = time.to_dtype(self.time1.bias.dtype())?;
        let linear = |layer: &LayeredLinear, x: &Tensor| if self.addition.is_none() { layer.forward_precise(x) } else { layer.forward(x) };
        let time = linear(&self.time2, &candle_nn::ops::silu(&linear(&self.time1, &time)?)?)?;
        let Some(addition) = &self.addition else { return Ok(time) };
        let batch = time.dim(0)?;
        let ids = match additional_t_cond {
            Some(ids) => ids.to_dtype(DType::U32)?.reshape(batch)?,
            None => Tensor::zeros(batch, DType::U32, time.device())?,
        };
        Ok((time + addition.index_select(&ids, 0)?)?)
    }

    pub fn project(&self, img: &Tensor, txt: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.img.forward(img)?, self.txt.forward(&rms_norm(txt, &self.txt_norm)?)?))
    }
}

struct Stream {
    image: bool,
    modulation: LayeredLinear,
    q: LayeredLinear,
    k: LayeredLinear,
    v: LayeredLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    out: LayeredLinear,
    up: LayeredLinear,
    down: LayeredLinear,
}

impl Stream {
    fn load(text: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let stream = if text { "txt" } else { "img" };
        let attn = vb.pp("attn");
        Ok(Self {
            image: !text,
            modulation: LayeredLinear::load(WIDTH, 6 * WIDTH, vb.pp(format!("{stream}_mod.1")))?,
            q: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_q_proj" } else { "to_q" }))?,
            k: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_k_proj" } else { "to_k" }))?,
            v: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_v_proj" } else { "to_v" }))?,
            q_norm: attn.pp(if text { "norm_added_q" } else { "norm_q" }).get(HEAD_DIM, "weight")?,
            k_norm: attn.pp(if text { "norm_added_k" } else { "norm_k" }).get(HEAD_DIM, "weight")?,
            out: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "to_add_out" } else { "to_out.0" }))?,
            up: LayeredLinear::load(WIDTH, 4 * WIDTH, vb.pp(format!("{stream}_mlp.net.0.proj")))?,
            down: LayeredLinear::load(4 * WIDTH, WIDTH, vb.pp(format!("{stream}_mlp.net.2")))?,
        })
    }

    fn modulate(x: &Tensor, params: &Tensor, offset: usize) -> Result<Tensor> {
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return crate::profile::scope("layered modulate", 0.0, x.device(), || {
                Ok(x.apply_op2_no_bwd(params, &cuda_ops::Modulate { shift: offset, scale: offset + 1 })?)
            });
        }
        let shift = params.narrow(1, offset * WIDTH, WIDTH)?.unsqueeze(1)?;
        let scale = (params.narrow(1, (offset + 1) * WIDTH, WIDTH)? + 1.0)?.unsqueeze(1)?;
        add_product(&shift, &layer_norm(x)?, &scale)
    }

    // Keep the established Layered arithmetic; editing uses wider CPU norm statistics.
    fn modulate_edit(x: &Tensor, params: &Tensor, offset: usize) -> Result<Tensor> {
        if !x.device().is_cpu() || x.dtype() != DType::F32 {
            return Self::modulate(x, params, offset);
        }
        let f = x.to_dtype(DType::F64)?;
        let centered = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
        let norm = centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?;
        let shift = params.narrow(1, offset * WIDTH, WIDTH)?.unsqueeze(1)?;
        let scale = (params.narrow(1, (offset + 1) * WIDTH, WIDTH)? + 1.0)?.unsqueeze(1)?;
        add_product(&shift, &norm, &scale)
    }

    fn modulate_split(x: &Tensor, params: &Tensor, offset: usize, split: Option<usize>) -> Result<Tensor> {
        let Some(n) = split else { return Self::modulate(x, params, offset) };
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return crate::profile::scope("edit2511 split modulate", 0.0, x.device(), || {
                Ok(x.apply_op2_no_bwd(params, &cuda_ops::EditModulate { shift: offset, split: n })?)
            });
        }
        let b = x.dim(0)?;
        Ok(Tensor::cat(
            &[
                Self::modulate_edit(&x.narrow(1, 0, n)?.contiguous()?, &params.narrow(0, 0, b)?, offset)?,
                Self::modulate_edit(&x.narrow(1, n, x.dim(1)? - n)?.contiguous()?, &params.narrow(0, b, b)?, offset)?,
            ],
            1,
        )?)
    }

    fn finish_split(&self, x: &Tensor, attention: &Tensor, params: &Tensor, split: Option<usize>) -> Result<Tensor> {
        let Some(n) = split else { return self.finish(x, attention, params) };
        // Project jointly; select the two gates at each residual boundary.
        let x = Self::residual_split(x, &self.out.forward_precise(attention)?, params, 2, n)?;
        let mlp = self.down.forward_precise(&self.up.forward_gelu_precise(&Self::modulate_split(&x, params, 3, split)?)?)?;
        Self::residual_split(&x, &mlp, params, 5, n)
    }

    fn residual_split(x: &Tensor, value: &Tensor, params: &Tensor, gate: usize, split: usize) -> Result<Tensor> {
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return crate::profile::scope("edit2511 split residual", 0.0, x.device(), || {
                Ok(x.apply_op3_no_bwd(&value.contiguous()?, params, &cuda_ops::EditResidual { gate, split })?)
            });
        }
        Ok((x + value.broadcast_mul(&Self::selected_gate(x, params, gate, split)?)?)?)
    }

    fn selected_gate(x: &Tensor, params: &Tensor, offset: usize, n: usize) -> Result<Tensor> {
        let b = x.dim(0)?;
        let p = params.narrow(1, offset * WIDTH, WIDTH)?.unsqueeze(1)?;
        Ok(Tensor::cat(&[p.narrow(0, 0, b)?.broadcast_as((b, n, WIDTH))?, p.narrow(0, b, b)?.broadcast_as((b, x.dim(1)? - n, WIDTH))?], 1)?)
    }

    fn qkv(&self, x: &Tensor, editing: bool) -> Result<(Tensor, Tensor, Tensor)> {
        #[cfg(feature = "fusednorm")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            let norm = |x: Tensor, w: &Tensor| -> Result<Tensor> {
                crate::profile::scope("layered head_norm", 0.0, x.device(), || Ok(x.apply_op2_no_bwd(w, &cuda_ops::HeadNorm)?))
            };
            let (b, s, _) = x.dims3()?;
            return Ok((
                norm(self.q.forward(x)?, &self.q_norm)?,
                norm(self.k.forward(x)?, &self.k_norm)?,
                self.v.forward(x)?.reshape((b, s, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?,
            ));
        }
        let (b, s, _) = x.dims3()?;
        let heads = |x: Tensor| -> Result<Tensor> { Ok(x.reshape((b, s, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?) };
        let norm = |x: Tensor, weight: &Tensor| -> Result<Tensor> {
            if editing && x.device().is_cpu() && x.dtype() == DType::F32 {
                let f = x.to_dtype(DType::F64)?;
                return Ok(f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?.broadcast_mul(weight)?);
            }
            rms_norm(&x, weight)
        };
        let linear = |l: &LayeredLinear| if editing { l.forward_precise(x) } else { l.forward(x) };
        Ok((norm(heads(linear(&self.q)?)?, &self.q_norm)?, norm(heads(linear(&self.k)?)?, &self.k_norm)?, heads(linear(&self.v)?)?))
    }

    fn finish(&self, x: &Tensor, attention: &Tensor, params: &Tensor) -> Result<Tensor> {
        self.finish_mode(x, attention, params, false)
    }

    fn finish_mode(&self, x: &Tensor, attention: &Tensor, params: &Tensor, editing: bool) -> Result<Tensor> {
        let projected = if editing { self.out.forward_precise(attention)? } else { self.out.forward(attention)? };
        let gate = params.narrow(1, 2 * WIDTH, WIDTH)?.unsqueeze(1)?;
        let x = if self.image { add_product(x, &gate, &projected)? } else { (x + projected.broadcast_mul(&gate)?)? };
        let normalized = if editing { Self::modulate_edit(&x, params, 3)? } else { Self::modulate(&x, params, 3)? };
        let hidden = if editing { self.up.forward_gelu_precise(&normalized)? } else { self.up.forward_gelu(&normalized)? };
        let mlp = if editing { self.down.forward_precise(&hidden)? } else { self.down.forward(&hidden)? };
        add_product(&x, &params.narrow(1, 5 * WIDTH, WIDTH)?.unsqueeze(1)?, &mlp)
    }
}

/// One independently loadable dual-stream block. Outputs are (image, text).
pub struct LayeredBlock {
    image: Stream,
    text: Stream,
}

impl LayeredBlock {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self { image: Stream::load(false, vb.clone())?, text: Stream::load(true, vb)? })
    }

    /// Inspect image conditioning at either attention (0) or MLP (1) boundary.
    pub fn image_conditioning(&self, image: &Tensor, temb: &Tensor, half: usize, split: usize) -> Result<(Tensor, Tensor)> {
        anyhow::ensure!(half < 2 && split > 0 && split < image.dim(1)? && temb.dim(0)? == 2 * image.dim(0)?, "invalid image conditioning split");
        let params = self.image.modulation.forward_precise(&candle_nn::ops::silu(temb)?)?;
        Ok((Stream::modulate_split(image, &params, half * 3, Some(split))?, Stream::selected_gate(image, &params, half * 3 + 2, split)?))
    }

    pub fn forward(&self, image: &Tensor, text: &Tensor, temb: &Tensor, rope: &LayeredRope) -> Result<(Tensor, Tensor)> {
        self.forward_masked(image, text, temb, rope, None)
    }

    /// Mask padded text keys in either CFG lane; padded queries are discarded.
    pub fn forward_masked(
        &self,
        image: &Tensor,
        text: &Tensor,
        temb: &Tensor,
        rope: &LayeredRope,
        padding: Option<(usize, usize)>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_split(image, text, temb, rope, padding, None)
    }

    /// Select t for target/text and zero for reference tokens in both modulation halves.
    pub fn forward_split(
        &self,
        image: &Tensor,
        text: &Tensor,
        temb: &Tensor,
        rope: &LayeredRope,
        padding: Option<(usize, usize)>,
        split: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_block(image, text, temb, rope, padding, split, true)
    }

    /// [`Self::forward_split`]; `sage_ok` lets the attention take SageAttention2 when it is on
    /// (the first block's queries and keys are too wide for its INT8: see [`sage_block`]).
    #[allow(clippy::too_many_arguments)]
    fn forward_block(
        &self,
        image: &Tensor,
        text: &Tensor,
        temb: &Tensor,
        rope: &LayeredRope,
        padding: Option<(usize, usize)>,
        split: Option<usize>,
        sage_ok: bool,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(n) = split {
            anyhow::ensure!(n > 0 && n < image.dim(1)? && temb.dim(0)? == 2 * image.dim(0)?, "invalid zero-timestep split");
        }
        let activated = candle_nn::ops::silu(temb)?;
        let im = if split.is_some() { self.image.modulation.forward_precise(&activated)? } else { self.image.modulation.forward(&activated)? };
        let tm = if split.is_some() {
            self.text.modulation.forward_precise(&activated.narrow(0, 0, image.dim(0)?)?)?
        } else {
            self.text.modulation.forward(&activated)?
        };
        let (iq, ik, iv) = self.image.qkv(&Stream::modulate_split(image, &im, 0, split)?, split.is_some())?;
        let text_modulated = if split.is_some() { Stream::modulate_edit(text, &tm, 0)? } else { Stream::modulate(text, &tm, 0)? };
        let (tq, tk, tv) = self.text.qkv(&text_modulated, split.is_some())?;
        let q = rope.apply(&Tensor::cat(&[tq, iq], 2)?)?;
        let k = rope.apply(&Tensor::cat(&[tk, ik], 2)?)?;
        let v = Tensor::cat(&[tv, iv], 2)?;
        let (b, _, s, _) = q.dims4()?;
        let attention = joint_attention_mode(&q, &k, &v, padding, sage_ok)?.reshape((b, s, WIDTH))?;
        let nt = text.dim(1)?;
        Ok((
            self.image.finish_split(image, &attention.narrow(1, nt, image.dim(1)?)?, &im, split)?,
            self.text.finish_mode(text, &attention.narrow(1, 0, nt)?, &tm, split.is_some())?,
        ))
    }
}

/// The largest |V| the f16 conversion for SA2 may see (headroom under f16's 65504).
#[cfg(feature = "sage2")]
const V_F16_RANGE: f64 = 16384.0;

/// SageAttention2 for the Layered DiT unless `QIR_LAYERED_SAGE=0` (the bf16 flash kernel then).
/// 2026-10-29, the 3-layer split at 864x1152 on the RTX 5090: 91.6 s warm against flash's
/// 137.2–139.1 s; the layers composited back onto the input at 36.99 dB against flash's 36.91 and
/// 38.73 dB at two seeds (the per-layer comparison with flash is not the measure: the background
/// layer's invented detail moves with any numeric change, int8 and fp8 matmuls included).
#[cfg(feature = "sage2")]
fn layered_sage() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_LAYERED_SAGE").map_or(true, |v| v.trim() != "0"))
}

/// The joint attention with SageAttention2 (INT8 QK, FP8 PV). `q,k,v` are `(B,H,S,D)` with RoPE
/// already applied; SA2 quantizes `(B,S,H,D)` views and fuses a RoPE, given here as the identity.
/// A CFG lane whose text is shorter than the batch's drops its padded keys (the flash path masks
/// them) so they take no share of the softmax. Returns `(B,S,H,D)` bf16.
#[cfg(feature = "sage2")]
fn sage2_joint_attention(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>, scale: f32, accum: crate::sage2::PvAccum) -> Result<Tensor> {
    use crate::sage2::{attention, quant_v, rope_quant, Role};
    let identity = |n: usize| -> Result<(Tensor, Tensor)> {
        Ok((Tensor::ones((n, HEAD_DIM / 2), DType::BF16, q.device())?, Tensor::zeros((n, HEAD_DIM / 2), DType::BF16, q.device())?))
    };
    let lane = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
        let (q, k) = (q.transpose(1, 2)?, k.transpose(1, 2)?);
        // SA2 takes V in f16, and Qwen-Image's values outgrow f16's range (65504) where bf16
        // holds them: scale each head's V into range first and its output back (attention is
        // linear in V). The range is per (lane, head), `(B,H,1,1)`: a max over D then over S, two
        // wide reductions (one over the whole tensor ran on a single block, ~60 ms a call).
        let range = (v.abs()?.max_keepdim(3)?.max_keepdim(2)?.to_dtype(DType::F32)? / V_F16_RANGE)?.maximum(1f64)?;
        let back = range.transpose(1, 2)?.to_dtype(DType::BF16)?; // (B,1,H,1) for the (B,S,H,D) output
        let v = v.broadcast_div(&range.to_dtype(v.dtype())?)?.to_dtype(DType::F16)?.transpose(1, 2)?;
        let (qc, qs) = identity(q.dim(1)?)?;
        let (kc, ks) = identity(k.dim(1)?)?;
        let q = rope_quant(&q, &qc, &qs, Role::Query)?;
        let k = rope_quant(&k, &kc, &ks, Role::Key)?;
        let v = quant_v(&v, accum)?;
        Ok(attention(&q, &k, &v, scale, false)?.broadcast_mul(&back)?)
    };
    let Some((positive, negative)) = padding else { return lane(q, k, v) };
    let tokens = positive.max(negative);
    let mut lanes = Vec::with_capacity(2);
    for (i, valid) in [positive, negative].into_iter().enumerate() {
        let (q, k, v) = (q.narrow(0, i, 1)?, k.narrow(0, i, 1)?, v.narrow(0, i, 1)?);
        lanes.push(if valid < tokens {
            let n = k.dim(2)?;
            let keep = |x: &Tensor| -> Result<Tensor> { Ok(Tensor::cat(&[x.narrow(2, 0, valid)?, x.narrow(2, tokens, n - tokens)?], 2)?) };
            lane(&q, &keep(&k)?, &keep(&v)?)?
        } else {
            lane(&q, &k, &v)?
        });
    }
    Ok(Tensor::cat(&lanes, 0)?)
}

/// The bf16 flash kernel's joint attention; a shorter CFG lane masks its padded keys.
#[cfg(feature = "flash")]
fn flash_joint_attention(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>, scale: f32) -> Result<Tensor> {
    let Some((positive, negative)) = padding else { return crate::flash::layered_attention(q, k, v, scale, None) };
    let tokens = positive.max(negative);
    let mut lanes = Vec::with_capacity(2);
    for (lane, valid) in [positive, negative].into_iter().enumerate() {
        let bias = (valid < tokens).then_some(crate::flash::KeyBias { rows_from: 0, keys: (valid, tokens), log_boost: f32::NEG_INFINITY });
        lanes.push(crate::flash::layered_attention(&q.narrow(0, lane, 1)?, &k.narrow(0, lane, 1)?, &v.narrow(0, lane, 1)?, scale, bias)?);
    }
    Ok(Tensor::cat(&lanes, 0)?)
}

/// `QIR_LAYERED_SAGE_CHECK=<image tokens per frame>` (a debug switch): the frame size that
/// splits the joint sequence into text, the latent frames and the reference frame.
#[cfg(all(feature = "sage2", feature = "flash"))]
fn sage_check_frame() -> Option<usize> {
    static FRAME: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FRAME.get_or_init(|| std::env::var("QIR_LAYERED_SAGE_CHECK").ok().and_then(|v| v.trim().parse().ok()).filter(|n: &usize| *n > 0))
}

/// The debug comparison: on blocks 0 and 59 of steps 0, 10 and 19 (one joint attention per block
/// and step with batched CFG), SA2 against the flash kernel on the same q, k, v, the relative
/// RMS error per token range (text, each latent frame, the reference frame, per CFG lane) and
/// the largest |q|, |k|, |v|, logged at info.
#[cfg(all(feature = "sage2", feature = "flash"))]
fn sage_check(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>, scale: f32, frame: usize) -> Result<()> {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (step, block) = (call / BLOCKS, call % BLOCKS);
    if ![0, 10, 19].contains(&step) {
        return Ok(());
    }
    let amax = |t: &Tensor| -> Result<f32> { Ok(t.abs()?.max_keepdim(3)?.max_keepdim(2)?.max_keepdim(1)?.flatten_all()?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?) };
    // Every block's range; the error comparison on the first two and the last.
    if ![0, 1, BLOCKS - 1].contains(&block) {
        tracing::info!(step, block, max_q = amax(q)?, max_k = amax(k)?, max_v = amax(v)?, "layered sage range");
        return Ok(());
    }
    let crate::sage2::AttentionImpl::Sage2(accum) = crate::sage2::attention_impl()? else { return Ok(()) };
    let reference = flash_joint_attention(q, k, v, padding, scale)?.to_dtype(DType::F32)?; // (B,S,H,D)
    let sage = sage2_joint_attention(q, k, v, padding, scale, accum)?.to_dtype(DType::F32)?;
    let s = q.dim(2)?;
    let images = s / frame; // whole frames after the text: the latent frames and the reference
    let text = s - images * frame;
    let mut ranges = vec![("text".to_string(), 0, text)];
    for f in 0..images {
        let name = if f + 1 == images { "reference".to_string() } else { format!("frame{f}") };
        ranges.push((name, text + f * frame, frame));
    }
    let mut parts = Vec::new();
    for lane in 0..q.dim(0)? {
        for (name, start, len) in &ranges {
            let r = reference.narrow(0, lane, 1)?.narrow(1, *start, *len)?;
            let d = (sage.narrow(0, lane, 1)?.narrow(1, *start, *len)? - &r)?;
            let err = (d.sqr()?.sum_all()?.to_scalar::<f32>()? / r.sqr()?.sum_all()?.to_scalar::<f32>()?.max(1e-30)).sqrt();
            parts.push(format!("L{lane} {name} {err:.4}"));
        }
    }
    tracing::info!(step, block, max_q = amax(q)?, max_k = amax(k)?, max_v = amax(v)?, errors = parts.join(", "), "layered sage check");
    Ok(())
}

/// Whether block `i` may use SageAttention2: not the [`FLASH_BLOCKS`].
fn sage_block(i: usize) -> bool {
    !FLASH_BLOCKS.contains(&i)
}

/// The blocks that keep the flash kernel under SageAttention2: those whose queries or keys
/// pass |32|. Measured 2026-10-29 (`QIR_LAYERED_SAGE_CHECK`, the 3-layer split of an 864x1152
/// portrait, steps 0, 10 and 19, both CFG lanes): block 0 reaches |796| (q) / |636| (k), where
/// SA2's per-block INT8 was 12–89 % off the flash kernel per token range; block 58 reaches |80.5|
/// and block 52 |45.5|; every other block stays at or under |21| (most |13–19|), and SA2 was
/// within 0.4–3.1 % on blocks 1 and 59 for the text, every latent frame and the reference.
const FLASH_BLOCKS: &[usize] = &[0, 52, 58];

/// Exact bidirectional attention; cap temporary scores by chunking queries.
#[cfg(test)]
fn joint_attention(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>) -> Result<Tensor> {
    joint_attention_mode(q, k, v, padding, true)
}

/// [`joint_attention`]; `sage_ok` false keeps this call on the flash kernel whatever the switch.
fn joint_attention_mode(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>, sage_ok: bool) -> Result<Tensor> {
    crate::profile::scope("layered joint_attention", 0.0, q.device(), || joint_attention_impl(q, k, v, padding, sage_ok))
}

fn joint_attention_impl(q: &Tensor, k: &Tensor, v: &Tensor, padding: Option<(usize, usize)>, sage_ok: bool) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    if let Some((positive, negative)) = padding {
        anyhow::ensure!(q.dim(0)? == 2 && positive > 0 && negative > 0 && positive.max(negative) < q.dim(2)?, "invalid CFG padding");
    }
    #[cfg(all(feature = "sage2", feature = "flash"))]
    if q.device().is_cuda() && q.dtype() == DType::BF16 {
        if let Some(frame) = sage_check_frame() {
            sage_check(q, k, v, padding, scale as f32, frame)?;
        }
    }
    #[cfg(feature = "sage2")]
    if q.device().is_cuda() && q.dtype() == DType::BF16 && sage_ok && layered_sage() {
        if let crate::sage2::AttentionImpl::Sage2(accum) = crate::sage2::attention_impl()? {
            return sage2_joint_attention(q, k, v, padding, scale as f32, accum);
        }
    }
    #[cfg(feature = "flash")]
    if q.device().is_cuda() && q.dtype() == DType::BF16 {
        return flash_joint_attention(q, k, v, padding, scale as f32);
    }
    let keys = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let values = v.to_dtype(DType::F32)?.contiguous()?;
    let n = q.dim(2)?;
    let bias = padding
        .map(|(positive, negative)| {
            let tokens = positive.max(negative);
            let mut bias = vec![0f32; 2 * n];
            bias[positive..tokens].fill(f32::NEG_INFINITY);
            bias[n + negative..n + tokens].fill(f32::NEG_INFINITY);
            Tensor::from_vec(bias, (2, 1, 1, n), q.device())
        })
        .transpose()?;
    let mut out = Vec::new();
    for start in (0..n).step_by(128) {
        let query = q.narrow(2, start, 128.min(n - start))?.to_dtype(DType::F32)?.contiguous()?;
        let scores = (query.matmul(&keys)? * scale)?;
        let scores = match &bias {
            Some(bias) => scores.broadcast_add(bias)?,
            None => scores,
        };
        out.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?);
    }
    Ok(Tensor::cat(&out, 2)?.to_dtype(v.dtype())?.transpose(1, 2)?.contiguous()?)
}

/// Adaptive output normalization (scale then shift) and 64-channel patch velocity.
pub struct LayeredOutput {
    modulation: LayeredLinear,
    projection: LayeredLinear,
}

impl LayeredOutput {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            modulation: LayeredLinear::load(WIDTH, WIDTH * 2, vb.pp("norm_out.linear"))?,
            projection: LayeredLinear::load(WIDTH, 64, vb.pp("proj_out"))?,
        })
    }

    pub fn normalize(&self, image: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let modulation = self.modulation.forward(&candle_nn::ops::silu(temb)?)?;
        #[cfg(feature = "fusednorm")]
        if image.device().is_cuda() && image.dtype() == DType::BF16 {
            return crate::profile::scope("layered modulate", 0.0, image.device(), || {
                Ok(image.apply_op2_no_bwd(&modulation, &cuda_ops::Modulate { shift: 1, scale: 0 })?)
            });
        }
        let scale = (modulation.narrow(1, 0, WIDTH)? + 1.0)?.unsqueeze(1)?;
        let shift = modulation.narrow(1, WIDTH, WIDTH)?.unsqueeze(1)?;
        add_product(&shift, &layer_norm(image)?, &scale)
    }

    pub fn forward(&self, image: &Tensor, temb: &Tensor) -> Result<Tensor> {
        self.projection.forward(&self.normalize(image, temb)?)
    }
}

/// Shared architecture; checkpoint markers select reference and timestep conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitKind {
    Layered,
    Edit2511,
}

impl DitKind {
    pub fn detect(vb: &VarBuilder<'_>) -> Result<Self> {
        if vb.contains_tensor("__index_timestep_zero__") {
            anyhow::ensure!(!vb.contains_tensor("time_text_embed.addition_t_embedding.weight"), "conflicting DiT markers");
            Ok(Self::Edit2511)
        } else {
            anyhow::ensure!(vb.contains_tensor("time_text_embed.addition_t_embedding.weight"), "unsupported Qwen DiT markers");
            Ok(Self::Layered)
        }
    }
}

/// Independent centered grids, with target tokens first and circular patch padding.
pub struct PackedImages {
    pub tokens: Tensor,
    pub positions: Tensor,
    pub target_tokens: usize,
    pub rows: usize,
    pub cols: usize,
}

impl PackedImages {
    pub fn new(kind: DitKind, target: &Tensor, references: &[Tensor]) -> Result<Self> {
        let (b, c, frames, h, w) = target.dims5()?;
        anyhow::ensure!(c == 16 && h > 0 && w > 0 && frames > 0, "invalid target latent");
        let rows = h.div_ceil(2);
        let cols = w.div_ceil(2);
        let mut tokens = vec![patchify_padded(target)?];
        let target_tokens = tokens[0].dim(1)?;
        let mut positions = vec![image_positions(b, frames, rows, cols, 0, target.device())?];
        for (i, reference) in references.iter().enumerate() {
            let (rb, rc, rt, rh, rw) = reference.dims5()?;
            anyhow::ensure!(rb == b && rc == 16 && rt == 1 && rh > 0 && rw > 0, "invalid reference latent");
            let index = i32::try_from(i + 1)? * if kind == DitKind::Layered { -1 } else { 1 };
            tokens.push(patchify_padded(reference)?);
            positions.push(image_positions(b, 1, rh.div_ceil(2), rw.div_ceil(2), index, target.device())?);
        }
        Ok(Self { tokens: Tensor::cat(&tokens, 1)?, positions: Tensor::cat(&positions, 1)?, target_tokens, rows, cols })
    }
}

/// ComfyUI wraps the first row/column onto the trailing odd patch edge.
pub fn patchify_padded(x: &Tensor) -> Result<Tensor> {
    let (_, _, _, h, w) = x.dims5()?;
    anyhow::ensure!(h > 0 && w > 0, "empty patch grid");
    let x = if h % 2 == 1 { Tensor::cat(&[x, &x.narrow(3, 0, 1)?], 3)? } else { x.clone() };
    let x = if w % 2 == 1 { Tensor::cat(&[&x, &x.narrow(4, 0, 1)?], 4)? } else { x };
    patchify(&x)
}

/// Shared checkpoint backbone; markers select the Layered or Edit-2511 convention.
pub struct LayeredDit {
    pub kind: DitKind,
    pub input: LayeredInput,
    blocks: Vec<LayeredBlock>,
    pub output: LayeredOutput,
    streamed: Option<VarBuilder<'static>>,
    binding: Option<crate::lora::LoraStack>,
}

impl LayeredDit {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let kind = DitKind::detect(&vb)?;
        let blocks = (0..BLOCKS).map(|i| LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))).collect::<Result<Vec<_>>>()?;
        Ok(Self { kind, input: LayeredInput::load(vb.clone())?, blocks, output: LayeredOutput::load(vb)?, streamed: None, binding: None })
    }

    /// Materialize an ordered adapter stack once; rebinding starts from exact CPU originals.
    pub fn bind_loras(&mut self, stack: &crate::lora::LoraStack) -> Result<()> {
        anyhow::ensure!(self.kind == DitKind::Edit2511 && self.streamed.is_none(), "2511 LoRAs require a resident edit DiT");
        if self.binding.as_ref().is_some_and(|old| old.same_adapters(stack)) {
            return Ok(());
        }
        self.binding = None;
        let mut known = std::collections::BTreeSet::new();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            for (text, stream) in [(false, &mut block.image), (true, &mut block.text)] {
                let prefix = format!("transformer_blocks.{i}");
                let name = if text { "txt" } else { "img" };
                for (suffix, linear) in [
                    (format!("{name}_mod.1"), &mut stream.modulation),
                    (format!("attn.{}", if text { "add_q_proj" } else { "to_q" }), &mut stream.q),
                    (format!("attn.{}", if text { "add_k_proj" } else { "to_k" }), &mut stream.k),
                    (format!("attn.{}", if text { "add_v_proj" } else { "to_v" }), &mut stream.v),
                    (format!("attn.{}", if text { "to_add_out" } else { "to_out.0" }), &mut stream.out),
                    (format!("{name}_mlp.net.0.proj"), &mut stream.up),
                    (format!("{name}_mlp.net.2"), &mut stream.down),
                ] {
                    let module = format!("{prefix}.{suffix}");
                    linear.bind(&module, &stack.pairs(&module)?)?;
                    known.insert(module);
                }
            }
        }
        anyhow::ensure!(stack.unmatched(&|m| known.contains(m)).is_empty(), "unmatched 2511 LoRA modules: {:?}", stack.unmatched(&|m| known.contains(m)));
        self.binding = Some(stack.clone());
        Ok(())
    }

    /// Retain only materialized weights on CPU so equal bindings survive stage unloads.
    pub fn materialized_weights(&self) -> Result<std::collections::BTreeMap<String, Tensor>> {
        let mut weights = std::collections::BTreeMap::new();
        for (i, block) in self.blocks.iter().enumerate() {
            for (text, stream) in [(false, &block.image), (true, &block.text)] {
                let prefix = format!("transformer_blocks.{i}");
                let name = if text { "txt" } else { "img" };
                for (suffix, linear) in [
                    (format!("{name}_mod.1"), &stream.modulation),
                    (format!("attn.{}", if text { "add_q_proj" } else { "to_q" }), &stream.q),
                    (format!("attn.{}", if text { "add_k_proj" } else { "to_k" }), &stream.k),
                    (format!("attn.{}", if text { "add_v_proj" } else { "to_v" }), &stream.v),
                    (format!("attn.{}", if text { "to_add_out" } else { "to_out.0" }), &stream.out),
                    (format!("{name}_mlp.net.0.proj"), &stream.up),
                    (format!("{name}_mlp.net.2"), &stream.down),
                ] {
                    if linear.original.is_none() {
                        continue;
                    }
                    let module = format!("{prefix}.{suffix}");
                    if let Some((weight, scale)) = &linear.fp8 {
                        weights.insert(format!("{module}.weight"), weight.to_device(&Device::Cpu)?);
                        weights.insert(format!("{module}.weight_scale"), scale.to_device(&Device::Cpu)?);
                    } else if let Some(dense) = &linear.dense {
                        weights.insert(format!("{module}.weight"), dense.weight().to_device(&Device::Cpu)?);
                    }
                }
            }
        }
        Ok(weights)
    }

    /// Every block resident, read from the checkpoint as `open_streamed` reads it (fp8 kept
    /// until each matmul): about 20 GB on the GPU, for a worker that splits more than once.
    pub fn open_resident(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        Self::from_resident_file(crate::layered_files::TensorFile::open(path)?, device)
    }

    /// Reuse a prefaulted mapping so uploading weights does not repeat page faults.
    pub(crate) fn from_resident_file(file: crate::layered_files::TensorFile, device: &Device) -> Result<Self> {
        Self::load(file.resident_builder(if device.is_cuda() { DType::BF16 } else { DType::F32 }, device)?)
    }

    /// Keep only one block resident at a time, including when running on CPU.
    pub fn open_streamed(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let vb = checkpoint_builder(path, device)?;
        Ok(Self {
            kind: DitKind::detect(&vb)?,
            input: LayeredInput::load(vb.clone())?,
            blocks: Vec::new(),
            output: LayeredOutput::load(vb.clone())?,
            streamed: Some(vb),
            binding: None,
        })
    }

    /// Legacy packed-token CFG branch; returns reference velocities too.
    /// Use `forward_latents` for Edit-2511 reference timestep selection and target slicing.
    /// Positions are (B,S,3); the optional additional condition defaults to entry zero.
    pub fn forward(
        &self,
        img_tokens: &Tensor,
        txt_tokens: &Tensor,
        timestep: &Tensor,
        img_positions: &Tensor,
        txt_positions: &Tensor,
        additional_t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let rope = LayeredRope::new(&Tensor::cat(&[txt_positions, img_positions], 1)?, img_tokens.dtype())?;
        self.forward_prepared(img_tokens, txt_tokens, timestep, additional_t_cond, &rope, None)
    }

    /// Reuse split-local RoPE tables and optionally batch positive/negative CFG lanes.
    pub fn forward_prepared(
        &self,
        img_tokens: &Tensor,
        txt_tokens: &Tensor,
        timestep: &Tensor,
        additional_t_cond: Option<&Tensor>,
        rope: &LayeredRope,
        padding: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        self.forward_selected(img_tokens, txt_tokens, timestep, additional_t_cond, rope, padding, None)
    }

    /// Latent boundary for editing; returns cropped target velocity only.
    pub fn forward_latents(&self, target: &Tensor, references: &[Tensor], text: &Tensor, timestep: &Tensor) -> Result<Tensor> {
        let packed = PackedImages::new(self.kind, target, references)?;
        let tp = text_positions(target.dim(0)?, text.dim(1)?, packed.rows, packed.cols, target.device())?;
        let rope = LayeredRope::new(&Tensor::cat(&[&tp, &packed.positions], 1)?, target.dtype())?;
        let split = (self.kind == DitKind::Edit2511 && !references.is_empty()).then_some(packed.target_tokens);
        let timestep = if split.is_some() { Tensor::cat(&[timestep, &(timestep * 0.0)?], 0)? } else { timestep.clone() };
        let velocity = self.forward_selected(&packed.tokens, text, &timestep, None, &rope, None, split)?.narrow(1, 0, packed.target_tokens)?;
        Ok(unpatchify(&velocity, target.dim(2)?, packed.rows * 2, packed.cols * 2)?.narrow(3, 0, target.dim(3)?)?.narrow(4, 0, target.dim(4)?)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_selected(
        &self,
        img_tokens: &Tensor,
        txt_tokens: &Tensor,
        timestep: &Tensor,
        additional_t_cond: Option<&Tensor>,
        rope: &LayeredRope,
        padding: Option<(usize, usize)>,
        split: Option<usize>,
    ) -> Result<Tensor> {
        let (mut img, mut txt) = self.input.project(img_tokens, txt_tokens)?;
        let temb = self.input.timestep(timestep, additional_t_cond)?;
        // SageAttention2 was measured on Qwen-Image-Layered only; 2511 keeps the flash kernel.
        let layered = self.kind == DitKind::Layered;
        for (i, block) in self.blocks.iter().enumerate() {
            (img, txt) = block.forward_block(&img, &txt, &temb, rope, padding, split, layered && sage_block(i))?;
        }
        if let Some(vb) = &self.streamed {
            for i in 0..BLOCKS {
                let block = LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))?;
                (img, txt) = block.forward_block(&img, &txt, &temb, rope, padding, split, layered && sage_block(i))?;
            }
        }
        let temb = if split.is_some() { temb.narrow(0, 0, img.dim(0)?)? } else { temb };
        self.output.forward(&img, &temb)
    }
}

/// Shared fusions preserve bf16 boundaries and compile at build time.
#[cfg(feature = "fusednorm")]
mod cuda_ops {
    use std::ffi::c_void;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::{CpuStorage, CudaStorage, Layout, Shape};

    use crate::layout::dense_byte_offset;

    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn qir_layered_rope(
            y: *mut c_void,
            x: *const c_void,
            c: *const c_void,
            s: *const c_void,
            pairs: u32,
            seq: u32,
            heads: u32,
            table_batches: u32,
            stream: *mut c_void,
        ) -> i32;
    }

    extern "C" {
        fn qir_layered_head_norm(y: *mut c_void, x: *const c_void, w: *const c_void, rows: u32, seq: u32, heads: u32, stream: *mut c_void) -> i32;
    }

    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn qir_layered_modulate(
            y: *mut c_void,
            x: *const c_void,
            params: *const c_void,
            rows: u32,
            seq: u32,
            width: u32,
            param_width: u32,
            param_batches: u32,
            shift: u32,
            scale: u32,
            stream: *mut c_void,
        ) -> i32;
    }

    pub(super) struct Modulate {
        pub shift: usize,
        pub scale: usize,
    }

    impl candle_core::CustomOp2 for Modulate {
        fn name(&self) -> &'static str {
            "layered-modulate"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Layered modulation requires CUDA")
        }
        fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, p: &CudaStorage, pl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, seq, width) = xl.shape().dims3()?;
            let (pb, pw) = pl.shape().dims2()?;
            if width != super::WIDTH || (pb != 1 && pb != b) || self.shift.max(self.scale) >= pw / width || pw % width != 0 {
                candle_core::bail!("invalid Layered modulation dimensions");
            }
            let narrow = |n| u32::try_from(n).map_err(candle_core::Error::wrap);
            let rows = narrow(b * seq)?;
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            // SAFETY: the kernel writes every dense output element exactly once.
            let out = unsafe { dev.alloc::<half::bf16>(b * seq * width)? };
            {
                let xv = x.as_cuda_slice::<half::bf16>()?;
                let pv = p.as_cuda_slice::<half::bf16>()?;
                let ((xp, _gx), (pp, _gp), (op, _go)) = (xv.device_ptr(&stream), pv.device_ptr(&stream), out.device_ptr(&stream));
                let xp = xp + dense_byte_offset::<half::bf16>(xl, "modulate x")?;
                let pp = pp + dense_byte_offset::<half::bf16>(pl, "modulate params")?;
                // SAFETY: validated dense inputs and parameter bounds on this stream.
                let rc = unsafe {
                    qir_layered_modulate(
                        op as *mut c_void,
                        xp as *const c_void,
                        pp as *const c_void,
                        rows,
                        narrow(seq)?,
                        narrow(width)?,
                        narrow(pw)?,
                        narrow(pb)?,
                        narrow(self.shift)?,
                        narrow(self.scale)?,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("Layered modulation failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
        }
    }

    extern "C" {
        fn qir_edit2511_modulate(
            y: *mut c_void,
            x: *const c_void,
            p: *const c_void,
            rows: u32,
            seq: u32,
            width: u32,
            pw: u32,
            pb: u32,
            shift: u32,
            split: u32,
            stream: *mut c_void,
        ) -> i32;
        fn qir_edit2511_residual(
            y: *mut c_void,
            x: *const c_void,
            v: *const c_void,
            p: *const c_void,
            elements: usize,
            seq: u32,
            width: u32,
            pw: u32,
            batches: u32,
            gate: u32,
            split: u32,
            stream: *mut c_void,
        ) -> i32;
    }

    pub(super) struct EditModulate {
        pub shift: usize,
        pub split: usize,
    }
    impl candle_core::CustomOp2 for EditModulate {
        fn name(&self) -> &'static str {
            "edit2511-modulate"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Edit modulation requires CUDA")
        }
        fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, p: &CudaStorage, pl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, seq, width) = xl.shape().dims3()?;
            let (pb, pw) = pl.shape().dims2()?;
            if width != super::WIDTH || pb != 2 * b || pw != 6 * width || self.shift > 3 || self.split == 0 || self.split >= seq {
                candle_core::bail!("invalid Edit modulation dimensions");
            }
            let u = |v| u32::try_from(v).map_err(candle_core::Error::wrap);
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            // SAFETY: every output element is written on the owning stream.
            let out = unsafe { dev.alloc::<half::bf16>(b * seq * width)? };
            {
                let xv = x.as_cuda_slice::<half::bf16>()?;
                let pv = p.as_cuda_slice::<half::bf16>()?;
                let ((xp, _gx), (pp, _gp), (op, _go)) = (xv.device_ptr(&stream), pv.device_ptr(&stream), out.device_ptr(&stream));
                let xp = xp + dense_byte_offset::<half::bf16>(xl, "edit modulate x")?;
                let pp = pp + dense_byte_offset::<half::bf16>(pl, "edit modulate p")?;
                // SAFETY: dense bf16 views with checked dimensions and parameter bounds.
                let rc = unsafe {
                    qir_edit2511_modulate(
                        op as *mut c_void,
                        xp as *const c_void,
                        pp as *const c_void,
                        u(b * seq)?,
                        u(seq)?,
                        u(width)?,
                        u(pw)?,
                        u(pb)?,
                        u(self.shift)?,
                        u(self.split)?,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("Edit modulation failed: {rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
        }
    }

    pub(super) struct EditResidual {
        pub gate: usize,
        pub split: usize,
    }
    impl candle_core::CustomOp3 for EditResidual {
        fn name(&self) -> &'static str {
            "edit2511-residual"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Edit residual requires CUDA")
        }
        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            xl: &Layout,
            v: &CudaStorage,
            vl: &Layout,
            p: &CudaStorage,
            pl: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, seq, width) = xl.shape().dims3()?;
            let (pb, pw) = pl.shape().dims2()?;
            if vl.shape() != xl.shape() || width != super::WIDTH || pb != 2 * b || pw != 6 * width || self.gate >= 6 || self.split == 0 || self.split >= seq {
                candle_core::bail!("invalid Edit residual dimensions");
            }
            let u = |v| u32::try_from(v).map_err(candle_core::Error::wrap);
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            // SAFETY: every output element is written on the owning stream.
            let out = unsafe { dev.alloc::<half::bf16>(b * seq * width)? };
            {
                let xv = x.as_cuda_slice::<half::bf16>()?;
                let vv = v.as_cuda_slice::<half::bf16>()?;
                let pv = p.as_cuda_slice::<half::bf16>()?;
                let ((xp, _gx), (vp, _gv), (pp, _gp), (op, _go)) =
                    (xv.device_ptr(&stream), vv.device_ptr(&stream), pv.device_ptr(&stream), out.device_ptr(&stream));
                let xp = xp + dense_byte_offset::<half::bf16>(xl, "edit residual x")?;
                let vp = vp + dense_byte_offset::<half::bf16>(vl, "edit residual v")?;
                let pp = pp + dense_byte_offset::<half::bf16>(pl, "edit residual p")?;
                // SAFETY: dense bf16 views with checked dimensions and parameter bounds.
                let rc = unsafe {
                    qir_edit2511_residual(
                        op as *mut c_void,
                        xp as *const c_void,
                        vp as *const c_void,
                        pp as *const c_void,
                        b * seq * width,
                        u(seq)?,
                        u(width)?,
                        u(pw)?,
                        u(b)?,
                        u(self.gate)?,
                        u(self.split)?,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("Edit residual failed: {rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
        }
    }

    extern "C" {
        fn qir_layered_bias(x: *mut c_void, bias: *const c_void, elements: usize, width: u32, gelu: bool, stream: *mut c_void) -> i32;
    }

    pub(super) struct Bias {
        pub gelu: bool,
    }

    impl candle_core::InplaceOp2 for Bias {
        fn name(&self) -> &'static str {
            "layered-bias"
        }
        fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<()> {
            candle_core::bail!("Layered bias requires CUDA")
        }
        fn cuda_fwd(&self, x: &mut CudaStorage, xl: &Layout, b: &CudaStorage, bl: &Layout) -> candle_core::Result<()> {
            let width = bl.shape().dims1()?;
            if xl.shape().dims().last() != Some(&width) || width % 8 != 0 {
                candle_core::bail!("invalid Layered bias dimensions");
            }
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            let xv = x.as_cuda_slice_mut::<half::bf16>()?;
            let bv = b.as_cuda_slice::<half::bf16>()?;
            let ((xp, _gx), (bp, _gb)) = (xv.device_ptr_mut(&stream), bv.device_ptr(&stream));
            let xp = xp + dense_byte_offset::<half::bf16>(xl, "bias output")?;
            let bp = bp + dense_byte_offset::<half::bf16>(bl, "bias values")?;
            let width = u32::try_from(width).map_err(candle_core::Error::wrap)?;
            // SAFETY: only fresh GEMM outputs are mutated, with no live aliases.
            let rc = unsafe {
                qir_layered_bias(xp as *mut c_void, bp as *const c_void, xl.shape().elem_count(), width, self.gelu, stream.cu_stream() as *mut c_void)
            };
            if rc != 0 {
                candle_core::bail!("Layered bias failed, rc={rc}");
            }
            Ok(())
        }
    }

    pub(super) struct HeadNorm;

    impl candle_core::CustomOp2 for HeadNorm {
        fn name(&self) -> &'static str {
            "layered-head-norm"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Layered head norm requires CUDA")
        }
        fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, seq, width) = xl.shape().dims3()?;
            if width != super::WIDTH || wl.shape().dims1()? != 128 {
                candle_core::bail!("invalid Layered head norm dimensions");
            }
            let heads = width / 128;
            let rows = u32::try_from(b * seq * heads).map_err(candle_core::Error::wrap)?;
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            // SAFETY: the kernel writes one value for every dense output element.
            let out = unsafe { dev.alloc::<half::bf16>(b * seq * width)? };
            {
                let xv = x.as_cuda_slice::<half::bf16>()?;
                let wv = w.as_cuda_slice::<half::bf16>()?;
                let ((xp, _gx), (wp, _gw), (op, _go)) = (xv.device_ptr(&stream), wv.device_ptr(&stream), out.device_ptr(&stream));
                let xp = xp + dense_byte_offset::<half::bf16>(xl, "head norm x")?;
                let wp = wp + dense_byte_offset::<half::bf16>(wl, "head norm w")?;
                // SAFETY: validated dense input and weight, with stream-ordered lifetimes.
                let rc = unsafe {
                    qir_layered_head_norm(
                        op as *mut c_void,
                        xp as *const c_void,
                        wp as *const c_void,
                        rows,
                        seq as u32,
                        heads as u32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("Layered head norm failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), (b, heads, seq, 128).into()))
        }
    }

    pub(super) struct Rope;

    impl candle_core::CustomOp3 for Rope {
        fn name(&self) -> &'static str {
            "layered-rope"
        }
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Layered fused RoPE requires CUDA")
        }
        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            xl: &Layout,
            c: &CudaStorage,
            cl: &Layout,
            s: &CudaStorage,
            sl: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let (b, h, seq, d) = xl.shape().dims4()?;
            let (cb, ch, cs, cd) = cl.shape().dims4()?;
            if d != 128 || ch != 1 || cs != seq || cd != 64 || (cb != 1 && cb != b) || cl.shape() != sl.shape() {
                candle_core::bail!("invalid Layered RoPE dimensions");
            }
            for l in [xl, cl, sl] {
                if !l.is_contiguous() {
                    candle_core::bail!("Layered RoPE requires dense inputs");
                }
            }
            let narrow = |v: usize| u32::try_from(v).map_err(|e| candle_core::Error::Msg(e.to_string()));
            let pairs = narrow(b * h * seq * 64)?;
            let dev = x.device().clone();
            let stream = dev.cuda_stream();
            // SAFETY: the kernel writes each of the allocated output elements exactly once.
            let out = unsafe { dev.alloc::<half::bf16>(b * h * seq * d)? };
            {
                let (xv, cv, sv) = (x.as_cuda_slice::<half::bf16>()?, c.as_cuda_slice::<half::bf16>()?, s.as_cuda_slice::<half::bf16>()?);
                let ((xp, _gx), (cp, _gc), (sp, _gs)) = (xv.device_ptr(&stream), cv.device_ptr(&stream), sv.device_ptr(&stream));
                let xp = xp + dense_byte_offset::<half::bf16>(xl, "rope x")?;
                let cp = cp + dense_byte_offset::<half::bf16>(cl, "rope cos")?;
                let sp = sp + dense_byte_offset::<half::bf16>(sl, "rope sin")?;
                let (op, _go) = out.device_ptr(&stream);
                // SAFETY: dense bf16 views of the shapes checked above, on this stream.
                let rc = unsafe {
                    qir_layered_rope(
                        op as *mut c_void,
                        xp as *const c_void,
                        cp as *const c_void,
                        sp as *const c_void,
                        pairs,
                        narrow(seq)?,
                        narrow(h)?,
                        narrow(cb)?,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle_core::bail!("layered rope failed, rc={rc}");
                }
            }
            Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
        }
    }
}

#[cfg(test)]
mod layered_tests {
    use super::*;

    fn error(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?)
    }

    #[test]
    #[cfg(feature = "fusednorm")]
    fn edit2511_split_fusions_preserve_bf16_boundaries() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for b in [1, 2] {
            let x = Tensor::randn(0f32, 2.0, (b + 1, 17, WIDTH), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, b)?;
            let v = Tensor::randn(0f32, 2.0, (b + 1, 17, WIDTH), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, b)?;
            let p = Tensor::randn(0f32, 0.5, (2 * b + 1, WIDTH * 6), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2 * b)?;
            for split in [1, 8, 16] {
                for offset in [0, 3] {
                    let old = Tensor::cat(
                        &[
                            Stream::modulate_edit(&x.narrow(1, 0, split)?.contiguous()?, &p.narrow(0, 0, b)?, offset)?,
                            Stream::modulate_edit(&x.narrow(1, split, 17 - split)?.contiguous()?, &p.narrow(0, b, b)?, offset)?,
                        ],
                        1,
                    )?;
                    let new = Stream::modulate_split(&x, &p, offset, Some(split))?;
                    assert_eq!(error(&old, &new)?, 0.0);
                }
                for gate in [2, 5] {
                    let old = (&x + v.broadcast_mul(&Stream::selected_gate(&x, &p, gate, split)?)?)?;
                    let new = Stream::residual_split(&x, &v, &p, gate, split)?;
                    assert_eq!(error(&old, &new)?, 0.0);
                }
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "fusednorm")]
    fn layered_head_norm_matches_candle_rounding() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::randn(0f32, 2.0, (3, 17, WIDTH), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2)?;
        let w = Tensor::randn(0f32, 0.5, 256, &dev)?.to_dtype(DType::BF16)?.narrow(0, 128, 128)?;
        let old = rms_norm(&x.reshape((2, 17, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?, &w)?;
        let new = x.apply_op2_no_bwd(&w, &cuda_ops::HeadNorm)?;
        assert_eq!(error(&old, &new)?, 0.0);
        Ok(())
    }

    #[test]
    #[cfg(feature = "fusednorm")]
    fn layered_modulation_matches_candle_rounding() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::randn(0f32, 2.0, (3, 17, WIDTH), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2)?;
        for pb in [1, 2] {
            let p = Tensor::randn(0f32, 0.5, (pb + 1, WIDTH * 6), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, pb)?;
            for (shift, scale) in [(0, 1), (3, 4), (1, 0)] {
                let a = p.narrow(1, shift * WIDTH, WIDTH)?.unsqueeze(1)?;
                let b = (p.narrow(1, scale * WIDTH, WIDTH)? + 1.0)?.unsqueeze(1)?;
                let old = add_product(&a, &layer_norm(&x)?, &b)?;
                let new = x.apply_op2_no_bwd(&p, &cuda_ops::Modulate { shift, scale })?;
                assert_eq!(error(&old, &new)?, 0.0);
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "fusednorm")]
    fn layered_bias_gelu_matches_candle_rounding() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for gelu in [false, true] {
            let x = Tensor::randn(0f32, 4.0, (3, 17, WIDTH), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2)?;
            let bias = Tensor::randn(0f32, 0.5, WIDTH * 2, &dev)?.to_dtype(DType::BF16)?.narrow(0, WIDTH, WIDTH)?;
            let old = x.broadcast_add(&bias)?;
            let old = if gelu { old.gelu()? } else { old };
            dev.synchronize()?;
            let new = linear_bias(x, &bias, gelu)?;
            assert_eq!(error(&old, &new)?, 0.0);
        }
        Ok(())
    }

    #[test]
    #[ignore = "exclusive CUDA GEMM benchmark"]
    #[cfg(feature = "cuda")]
    fn layered_gemm_shapes() -> Result<()> {
        use candle_core::{
            cuda_backend::cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig},
            Storage,
        };
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(d) = &dev else { unreachable!() };
        let lt = CudaBlasLT::new(d.cuda_stream())?;
        for m in [12000, 38880] {
            for (k, n) in [(3072, 3072), (3072, 12288), (12288, 3072)] {
                let x = Tensor::randn(0f32, 0.1, (m, k), &dev)?.to_dtype(DType::BF16)?;
                let w = Tensor::randn(0f32, 0.1, (n, k), &dev)?.to_dtype(DType::BF16)?;
                let (xs, _) = x.storage_and_layout();
                let (ws, _) = w.storage_and_layout();
                let Storage::Cuda(xs) = &*xs else { unreachable!() };
                let Storage::Cuda(ws) = &*ws else { unreachable!() };
                let xs = xs.as_cuda_slice::<half::bf16>()?;
                let ws = ws.as_cuda_slice::<half::bf16>()?;
                let mut out = unsafe { d.alloc::<half::bf16>(m * n)? };
                let cfg = MatmulConfig {
                    transa: true,
                    transb: false,
                    transc: false,
                    m: n as u64,
                    n: m as u64,
                    k: k as u64,
                    alpha: 1.,
                    beta: 0.,
                    lda: k as i64,
                    ldb: k as i64,
                    ldc: n as i64,
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                };
                for use_lt in [false, true, false, true] {
                    dev.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..5 {
                        if use_lt {
                            unsafe {
                                lt.matmul(cfg, &ws.slice(..), &xs.slice(..), &mut out, None, None)?;
                            }
                        } else {
                            let _ = x.matmul(&w.t()?)?;
                        }
                    }
                    dev.synchronize()?;
                    eprintln!("GEMM m={m} k={k} n={n} lt={use_lt}: {:.3} ms", start.elapsed().as_secs_f64() * 200.);
                }
            }
        }
        Ok(())
    }

    fn padding_check(dev: &Device, dtype: DType) -> Result<()> {
        let q = Tensor::randn(0f32, 0.5, (2, 2, 11, 128), dev)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 0.5, (2, 2, 11, 128), dev)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 0.5, (2, 2, 11, 128), dev)?.to_dtype(dtype)?;
        for lengths in [(4, 2), (2, 4)] {
            let actual = joint_attention(&q, &k, &v, Some(lengths))?;
            for (lane, valid) in [lengths.0, lengths.1].into_iter().enumerate() {
                let compact = |t: &Tensor| -> Result<Tensor> {
                    let t = t.narrow(0, lane, 1)?;
                    Ok(Tensor::cat(&[t.narrow(2, 0, valid)?, t.narrow(2, 4, 7)?], 2)?)
                };
                let expected = joint_attention(&q.narrow(0, lane, 1)?, &compact(&k)?, &compact(&v)?, None)?;
                let err = error(&actual.narrow(0, lane, 1)?, &expected)?;
                assert!(err < 0.01, "padding lane {lane}: {err}");
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "sage2")]
    fn sage2_joint_attention_tracks_the_exact_attention() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let q = Tensor::randn(0f32, 1.0, (2, 3, 300, 128), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (2, 3, 300, 128), &dev)?.to_dtype(DType::BF16)?;
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu)?.to_dtype(DType::F32);
        // Values past f16's range too (Qwen-Image's activations reach them; bf16 holds them).
        for (v_scale, padding) in [(1.0, None), (1.0, Some((40, 25))), (2.0e5, None), (2.0e5, Some((40, 25)))] {
            let v = Tensor::randn(0f32, v_scale, (2, 3, 300, 128), &dev)?.to_dtype(DType::BF16)?;
            let actual = joint_attention(&q, &k, &v, padding)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
            let expected = joint_attention(&cpu(&q)?, &cpu(&k)?, &cpu(&v)?, padding)?.flatten_all()?;
            let dot = (&actual * &expected)?.sum_all()?.to_scalar::<f32>()?;
            let norms = actual.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() * expected.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            assert!(dot / norms > 0.995, "V x{v_scale}, {padding:?}: cosine {}", dot / norms);
        }
        Ok(())
    }

    #[test]
    fn padded_cfg_keys_match_separate_attention() -> Result<()> {
        padding_check(&Device::Cpu, DType::F32)?;
        #[cfg(feature = "flash")]
        padding_check(&Device::new_cuda(0)?, DType::BF16)?;
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn fused_rope_and_broadcast_addcmul_preserve_rounding() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for batch in [1, 2] {
            let ids = text_positions(batch, 13, 4, 6, &dev)?;
            let rope = LayeredRope::new(&ids, DType::BF16)?;
            // Include a nonzero storage offset and both shared and per-lane tables.
            let x = Tensor::randn(0f32, 1.0, (3, 2, 13, 128), &dev)?.to_dtype(DType::BF16)?.narrow(0, 1, 2)?;
            assert_eq!(error(&rope.apply(&x)?, &rope.apply_unfused(&x)?)?, 0.0);
        }
        let x = Tensor::randn(0f32, 1.0, (2, 19, 128), &dev)?.to_dtype(DType::BF16)?;
        let gate = Tensor::randn(0f32, 1.0, (1, 1, 128), &dev)?.to_dtype(DType::BF16)?;
        let value = Tensor::randn(0f32, 1.0, (2, 19, 128), &dev)?.to_dtype(DType::BF16)?;
        let expected =
            x.to_dtype(DType::F32)?.broadcast_add(&value.to_dtype(DType::F32)?.broadcast_mul(&gate.to_dtype(DType::F32)?)?)?.to_dtype(DType::BF16)?;
        assert_eq!(error(&add_product(&x, &gate, &value)?, &expected)?, 0.0);
        let expected = gate.to_dtype(DType::F32)?.broadcast_add(&x.to_dtype(DType::F32)?.broadcast_mul(&gate.to_dtype(DType::F32)?)?)?.to_dtype(DType::BF16)?;
        assert_eq!(error(&add_product(&gate, &x, &gate)?, &expected)?, 0.0);
        let f = x.to_dtype(DType::F32)?;
        let centered = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
        let norm = centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(DType::BF16)?;
        assert!(error(&layer_norm(&x)?, &norm)? <= 0.015625);
        Ok(())
    }
}

#[cfg(test)]
mod edit_lora_tests {
    use super::*;
    #[test]
    fn dense_binding_stacks_and_restores_exact_originals() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &device)?;
        let bias = Tensor::zeros(2, DType::F32, &device)?;
        let vb = VarBuilder::from_tensors(std::collections::HashMap::from([("weight".into(), weight.clone()), ("bias".into(), bias)]), DType::F32, &device);
        let mut linear = LayeredLinear::load(2, 2, vb)?;
        let a = Tensor::from_slice(&[1f32, 2.], (1, 2), &device)?;
        let b = Tensor::from_slice(&[3f32, 4.], (2, 1), &device)?;
        let p = crate::lora::LoraPair::new(&a, &b, 0.5)?;
        let x = Tensor::from_slice(&[1f32, 1.], (1, 2), &device)?;
        let original = linear.forward(&x)?.to_vec2::<f32>()?;
        linear.bind("test", &[p.clone(), p.clone()])?;
        assert_eq!(linear.forward(&x)?.to_vec2::<f32>()?, vec![vec![12., 19.]]);
        linear.bind("test", &[p])?;
        assert_eq!(linear.forward(&x)?.to_vec2::<f32>()?, vec![vec![7.5, 13.]]);
        linear.bind("test", &[])?;
        assert_eq!(linear.forward(&x)?.to_vec2::<f32>()?, original);
        assert_eq!(linear.dense.as_ref().unwrap().weight().to_vec2::<f32>()?, weight.to_vec2::<f32>()?);
        Ok(())
    }
}
