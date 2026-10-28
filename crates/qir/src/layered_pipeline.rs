//! The shipped Layered graph: reference conditioning, Euler CFG, then independent RGBA slices.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
/// The pictures split takes and returns.
pub use image::RgbaImage;
use tokenizers::Tokenizer;

use crate::comfy::{torch_randn, FlowSchedule, Shift};
use crate::layered_files::{LayeredFiles, TensorFile};
use crate::model::qwen25_text::{load_tokenizer, tokenize, Qwen25TextEncoder};
use crate::model::qwen_image_dit::{image_positions, patchify, text_positions, unpatchify, LayeredDit, LayeredRope};
use crate::model::wan_vae::WanVae;
use crate::Result;

pub const DEFAULT_STEPS: usize = 20;
pub const CFG: f32 = 2.5;

/// Draw before packing: PyTorch fills channel, slice, row, column in that order.
pub fn layered_noise(seed: u64, layers: usize, height: usize, width: usize, device: &Device) -> Result<Tensor> {
    anyhow::ensure!(layers > 0 && height > 0 && width > 0 && height.is_multiple_of(2) && width.is_multiple_of(2), "invalid Layered latent dimensions");
    let frames = layers.checked_add(1).ok_or_else(|| anyhow::anyhow!("slice count overflow"))?;
    let count = [16, frames, height, width].into_iter().try_fold(1usize, |n, d| n.checked_mul(d)).ok_or_else(|| anyhow::anyhow!("latent size overflow"))?;
    Ok(Tensor::from_vec(torch_randn(seed, count).map_err(anyhow::Error::msg)?, (1, 16, frames, height, width), device)?)
}

/// AuraFlow replaces the sampling table; simple selects its rows and appends zero.
pub fn layered_schedule(steps: usize) -> Result<FlowSchedule> {
    FlowSchedule::simple(steps, Shift::Snr { shift: 1.0, multiplier: 1.0, timesteps: 1000 }).map_err(anyhow::Error::msg)
}

/// Layered's BaseModel passes sigma as f32, even with bf16 image activations.
/// Qwen-Image-2.1's double bf16 rounding does not apply to this older DiT.
pub fn layered_timestep(sigma: f32, device: &Device) -> Result<Tensor> {
    anyhow::ensure!(sigma.is_finite() && (0.0..=1.0).contains(&sigma), "invalid flow sigma");
    Ok(Tensor::from_slice(&[sigma], 1, device)?)
}

/// Preserve ComfyUI's denoised-space CFG and to_d arithmetic, including f32 cancellation.
pub fn euler_cfg(x: &Tensor, positive: &Tensor, negative: &Tensor, sigma: f32, next: f32) -> Result<Tensor> {
    anyhow::ensure!(sigma.is_finite() && sigma > 0.0 && next.is_finite() && next >= 0.0 && next <= sigma, "invalid Euler interval");
    anyhow::ensure!(x.dims() == positive.dims() && x.dims() == negative.dims(), "CFG tensor shapes disagree");
    let x = x.to_dtype(DType::F32)?;
    let pos = (&x - (positive.to_dtype(DType::F32)? * sigma as f64)?)?;
    let neg = (&x - (negative.to_dtype(DType::F32)? * sigma as f64)?)?;
    let denoised = (&neg + ((pos - &neg)? * CFG as f64)?)?;
    let derivative = ((&x - denoised)? / sigma as f64)?;
    Ok((&x + (derivative * (next - sigma) as f64)?)?)
}

/// Drop the composite and move time to batch in LatentCutToBatch's order.
pub fn retained_slices(latent: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = latent.dims5()?;
    anyhow::ensure!(b == 1 && c == 16 && t > 1, "expected a single Layered sample with retained slices");
    Ok(latent.narrow(2, 1, t - 1)?.transpose(1, 2)?.contiguous()?.reshape((t - 1, 16, h, w))?)
}

/// SaveImage clips and truncates, preserving straight alpha without rounding to nearest.
pub fn rgba_image(decoded: &Tensor) -> Result<RgbaImage> {
    let (b, c, h, w) = decoded.dims4()?;
    anyhow::ensure!(b == 1 && c == 4, "expected one decoded RGBA image");
    let pixels = ((decoded.squeeze(0)?.permute((1, 2, 0))? + 1.0)? / 2.0)?.clamp(0.0, 1.0)?;
    let bytes = (pixels * 255.0)?.to_dtype(DType::U8)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    RgbaImage::from_raw(u32::try_from(w)?, u32::try_from(h)?, bytes).ok_or_else(|| anyhow::anyhow!("invalid RGBA byte count"))
}

/// File-backed stages avoid retaining the encoder alongside the diffusion model.
pub struct LayeredPipeline {
    files: LayeredFiles,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl LayeredPipeline {
    pub fn load(models_root: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(models_root, crate::device::best_device()?)
    }

    /// Choose CPU explicitly for parity; CUDA computes model activations in bf16.
    pub fn load_on_device(models_root: impl AsRef<Path>, device: Device) -> Result<Self> {
        let root = models_root.as_ref();
        let files = LayeredFiles::resolve(root)?;
        let tokenizer = crate::comfy_files::tokenizer_dir(root).ok_or_else(|| anyhow::anyhow!("ComfyUI Qwen tokenizer code tree not found"))?;
        Ok(Self { files, tokenizer: load_tokenizer(tokenizer)?, dtype: if device.is_cuda() { DType::BF16 } else { DType::F32 }, device })
    }

    /// The device the split runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn vae(&self) -> Result<WanVae> {
        let vae = WanVae::load(TensorFile::open(&self.files.vae)?.builder("", self.dtype, &self.device)?)?;
        anyhow::ensure!(vae.image_channels() == 4 && vae.can_encode(), "Layered requires an RGBA encoder and decoder");
        Ok(vae)
    }

    /// Dimensions are supplied at graph resolution (multiples of 16); engine resizing is step 7.
    /// Progress counts completed Euler steps. Cancellation is checked between model calls.
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &self,
        picture: &RgbaImage,
        layers: usize,
        prompt: &str,
        seed: u64,
        steps: usize,
        progress: impl FnMut(usize, usize),
        cancel: &AtomicBool,
    ) -> Result<Vec<RgbaImage>> {
        self.split_observed(picture, layers, prompt, seed, steps, progress, cancel, |_, _| Ok(()))
    }

    /// Observe complete tensors at parity boundaries; the callback owns any copies it retains.
    #[allow(clippy::too_many_arguments)]
    pub fn split_observed(
        &self,
        picture: &RgbaImage,
        layers: usize,
        prompt: &str,
        seed: u64,
        steps: usize,
        mut progress: impl FnMut(usize, usize),
        cancel: &AtomicBool,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Vec<RgbaImage>> {
        let check = || -> Result<()> {
            anyhow::ensure!(!cancel.load(Ordering::Relaxed), "Layered split cancelled");
            Ok(())
        };
        check()?;
        let (w, h) = (picture.width() as usize, picture.height() as usize);
        anyhow::ensure!(
            (1..=8).contains(&layers) && w > 0 && h > 0 && w.is_multiple_of(16) && h.is_multiple_of(16),
            "expected 1..8 layers and positive picture dimensions divisible by 16"
        );
        let schedule = layered_schedule(steps)?;
        // `QIR_PROFILE=1`: each phase's seconds on stderr (the device synchronised at each mark).
        let profile = std::env::var("QIR_PROFILE").is_ok_and(|v| v.trim() == "1");
        let timing = profile || std::env::var("QIR_LAYERED_TIMING").is_ok_and(|v| v == "1");
        let mut last = std::time::Instant::now();
        let mut mark = |what: &str| -> Result<()> {
            if timing {
                self.device.synchronize()?;
                eprintln!("layered {what}: {:.2} s", last.elapsed().as_secs_f32());
                last = std::time::Instant::now();
            }
            Ok(())
        };
        // LoadImage supplies RGB to VAEEncode; alpha is a separate, unused mask in this graph.
        let rgb: Vec<f32> = picture.pixels().flat_map(|p| p.0[..3].iter().map(|v| (*v as f32 / 255.0) * 2.0 - 1.0)).collect();
        let pixels = Tensor::from_vec(rgb, (1, h, w, 3), &self.device)?.permute((0, 3, 1, 2))?.contiguous()?;
        let reference = self.vae()?.encode(&pixels)?.unsqueeze(2)?;
        observe("reference", &reference)?;
        // Release reference scratch before loading the resident encoder.
        crate::device::trim_pool(&self.device);
        mark("reference encode")?;
        check()?;
        let (positive, negative) = {
            let encoder = Qwen25TextEncoder::from_file(&self.files.text_encoder, self.dtype, &self.device)?;
            let positive = encoder.forward(&tokenize(&self.tokenizer, prompt)?)?;
            check()?;
            let negative = encoder.forward(&tokenize(&self.tokenizer, "")?)?;
            (positive, negative)
        };
        observe("positive_text", &positive)?;
        observe("negative_text", &negative)?;
        // The encoder's memory back to the driver before the 20 GB DiT arrives: kept in the
        // pool, the two overfilled the 32 GB card and the first steps paged (50 s each, then 15
        // under the profiler, 2026-10-27).
        crate::device::trim_pool(&self.device);
        mark("text encode")?;
        crate::profile::report("layered text");
        check()?;
        let (lh, lw, frames) = (h / 8, w / 8, layers + 1);
        let reference = patchify(&reference)?.to_dtype(self.dtype)?;
        let ids = Tensor::cat(&[image_positions(1, frames, lh / 2, lw / 2, 0, &self.device)?, image_positions(1, 1, lh / 2, lw / 2, -1, &self.device)?], 1)?;
        let pos_ids = text_positions(1, positive.dim(1)?, lh / 2, lw / 2, &self.device)?;
        let neg_ids = text_positions(1, negative.dim(1)?, lh / 2, lw / 2, &self.device)?;
        let batched = self.device.is_cuda() && std::env::var("QIR_LAYERED_CFG_BATCH").map_or(true, |v| v != "0");
        let nt = positive.dim(1)?;
        let nn = negative.dim(1)?;
        let tokens = nt.max(nn);
        let pad = |x: &Tensor| -> Result<Tensor> {
            let n = x.dim(1)?;
            Ok(if n == tokens { x.clone() } else { Tensor::cat(&[x, &Tensor::zeros((1, tokens - n, x.dim(2)?), x.dtype(), x.device())?], 1)? })
        };
        let texts = if batched { Some(Tensor::cat(&[pad(&positive)?, pad(&negative)?], 0)?) } else { None };
        let rope_ids = if batched { text_positions(1, tokens, lh / 2, lw / 2, &self.device)? } else { pos_ids };
        let rope = LayeredRope::new(&Tensor::cat(&[&rope_ids, &ids], 1)?, self.dtype)?;
        let neg_rope = if batched { None } else { Some(LayeredRope::new(&Tensor::cat(&[&neg_ids, &ids], 1)?, self.dtype)?) };
        let mut latent = layered_noise(seed, layers, lh, lw, &self.device)?;
        observe("noise", &latent)?;
        progress(0, steps);
        {
            // On CUDA the whole DiT goes on the device for the split's forwards (the encoder is
            // gone by now: the two do not fit 32 GB together), else one block at a time.
            let dit = if self.device.is_cuda() {
                LayeredDit::open_resident(&self.files.diffusion, &self.device)?
            } else {
                LayeredDit::open_streamed(&self.files.diffusion, &self.device)?
            };
            // Finish uploads and release loading scratch before the denoising working set.
            crate::device::trim_pool(&self.device);
            mark("DiT load")?;
            #[cfg(feature = "cuda")]
            let pool = LayeredPool::retain(&self.device);
            for i in 0..steps {
                check()?;
                let packed = patchify(&latent)?.to_dtype(self.dtype)?;
                let count = packed.dim(1)?;
                let input = Tensor::cat(&[packed, reference.clone()], 1)?;
                let time = layered_timestep(schedule.sigma(i), &self.device)?;
                let (pos, neg) = if let Some(texts) = &texts {
                    let input = Tensor::cat(&[&input, &input], 0)?;
                    let velocities = dit.forward_prepared(&input, texts, &time, None, &rope, (nn != nt).then_some((nt, nn)))?.narrow(1, 0, count)?;
                    (velocities.narrow(0, 0, 1)?, velocities.narrow(0, 1, 1)?)
                } else {
                    let pos = dit.forward_prepared(&input, &positive, &time, None, &rope, None)?.narrow(1, 0, count)?;
                    check()?;
                    let neg_rope = neg_rope.as_ref().ok_or_else(|| anyhow::anyhow!("missing negative RoPE"))?;
                    let neg = dit.forward_prepared(&input, &negative, &time, None, neg_rope, None)?.narrow(1, 0, count)?;
                    (pos, neg)
                };
                let pos = unpatchify(&pos, frames, lh, lw)?.to_dtype(DType::F32)?;
                let neg = unpatchify(&neg, frames, lh, lw)?.to_dtype(DType::F32)?;
                observe(&format!("step_{i}.positive"), &pos)?;
                observe(&format!("step_{i}.negative"), &neg)?;
                check()?;
                latent = euler_cfg(&latent, &pos, &neg, schedule.sigma(i), schedule.sigmas[i + 1])?;
                observe(&format!("step_{i}.latent"), &latent)?;
                #[cfg(feature = "cuda")]
                if pool.is_some() {
                    self.device.synchronize()?;
                }
                progress(i + 1, steps);
                if profile {
                    mark(&format!("step {i}"))?;
                }
                if i == 0 {
                    crate::profile::report("layered step 0");
                }
            }
            // Finish checkpoint use before freeing its storage and loading the decoder.
            self.device.synchronize()?;
            mark("denoise")?;
            crate::profile::report("layered steps 1..");
        }
        crate::device::trim_pool(&self.device);
        check()?;
        let retained = retained_slices(&latent)?;
        observe("retained", &retained)?;
        let vae = self.vae()?;
        let mut images = Vec::with_capacity(layers);
        for i in 0..layers {
            check()?;
            let raw = vae.denormalize(&retained.narrow(0, i, 1)?.contiguous()?)?;
            let decoded = vae.decode(&raw)?;
            observe(&format!("decoded_{i}"), &decoded)?;
            images.push(rgba_image(&decoded)?);
        }
        mark("decode")?;
        Ok(images)
    }
}

/// Keep a bounded denoising workspace across step barriers, restoring the caller's pool policy.
#[cfg(feature = "cuda")]
struct LayeredPool {
    device: candle_core::CudaDevice,
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    previous: u64,
}

#[cfg(feature = "cuda")]
impl LayeredPool {
    fn retain(device: &Device) -> Option<Self> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use sys::CUmemPool_attribute as A;
        if std::env::var("QIR_LAYERED_POOL").is_ok_and(|v| v == "0") || std::env::var("QIR_POOL_RETAIN").is_ok_and(|v| v == "0") {
            return None;
        }
        let Device::Cuda(device) = device else {
            return None;
        };
        let ctx = device.cuda_stream().context().clone();
        ctx.bind_to_thread().ok()?;
        let mut pool = std::ptr::null_mut();
        unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }.result().ok()?;
        let get = |attribute| -> Option<u64> {
            let mut value = 0u64;
            unsafe { sys::cuMemPoolGetAttribute(pool, attribute, (&mut value as *mut u64).cast()) }.result().ok()?;
            Some(value)
        };
        let previous = get(A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)?;
        let threshold = get(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?.saturating_add(2 << 30);
        unsafe { sys::cuMemPoolSetAttribute(pool, A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&threshold as *const u64).cast_mut().cast()) }.result().ok()?;
        Some(Self { device: device.clone(), pool, previous })
    }
}

#[cfg(feature = "cuda")]
impl Drop for LayeredPool {
    fn drop(&mut self) {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let _ = self.device.cuda_stream().context().bind_to_thread();
        let _ = unsafe {
            sys::cuMemPoolSetAttribute(self.pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&self.previous as *const u64).cast_mut().cast())
        }
        .result();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod layered_pool_tests {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::sys;

    #[test]
    fn pool_policy_is_restored_on_error() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &device else { anyhow::bail!("CUDA required") };
        let mut pool = std::ptr::null_mut();
        unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, cuda.cuda_stream().context().cu_device()) }.result()?;
        let threshold = || -> Result<u64> {
            let mut value = 0u64;
            unsafe { sys::cuMemPoolGetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&mut value as *mut u64).cast()) }
                .result()?;
            Ok(value)
        };
        let before = threshold()?;
        let guard = LayeredPool::retain(&device).ok_or_else(|| anyhow::anyhow!("pool retention unavailable or opted out"))?;
        assert_eq!(guard.previous, before);
        assert!(threshold()? >= 2 << 30);
        let attempt = move || -> Result<()> {
            let _guard = guard;
            anyhow::bail!("test cancellation")
        };
        assert!(attempt().is_err());
        assert_eq!(threshold()?, before);
        Ok(())
    }
}
