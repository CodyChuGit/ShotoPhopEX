//! Edit-2511 graph execution with staged weights and seed-independent conditioning.
use crate::{
    comfy::{torch_randn, FlowSchedule, Shift},
    edit2511_files::Edit2511Files,
    edit2511_prep::{crop_vae_target, latent_noise_mask, resize_kontext, resize_vae_reference},
    layered_files::TensorFile,
    lora::LoraStack,
    model::{
        qwen25_text::{load_tokenizer, Qwen25TextEncoder},
        qwen25_vision::{resize_editplus, Qwen25Vision},
        qwen_image_dit::LayeredDit,
        wan_vae::WanVae,
    },
    Result,
};
use candle_core::{DType, Device, Tensor};
pub use image::RgbImage;
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};
use tokenizers::Tokenizer;

/// Inputs after template routing; references are already Kontext-scaled where the graph asks.
pub struct Edit2511Request {
    pub prompt: String,
    pub negative: String,
    pub references: Vec<RgbImage>,
    pub target: Option<RgbImage>,
    pub mask: Option<RgbImage>,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub steps: usize,
    pub cfg: f32,
    pub loras: LoraStack,
}

#[derive(Clone, PartialEq, Eq)]
struct ConditioningKey {
    references: Vec<RgbImage>,
    prompt: String,
    negative: Option<String>,
}
impl ConditioningKey {
    fn new(req: &Edit2511Request) -> Self {
        Self { references: req.references.clone(), prompt: req.prompt.clone(), negative: (req.cfg > 1.).then(|| req.negative.clone()) }
    }
}
#[derive(Clone)]
struct Conditioning {
    references: Vec<Tensor>,
    positive: Tensor,
    negative: Option<Tensor>,
}

/// AuraFlow's table replaces the base model's schedule.
pub fn edit2511_schedule(steps: usize) -> Result<FlowSchedule> {
    FlowSchedule::simple(steps, Shift::Snr { shift: 3.1, multiplier: 1., timesteps: 1000 }).map_err(anyhow::Error::msg)
}

/// Fractional masks mix protected values at both sampler boundaries.
pub fn masked_replace(value: &Tensor, protected: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    Ok(match mask {
        Some(mask) => (value.broadcast_mul(mask)? + protected.broadcast_mul(&(1. - mask)?)?)?,
        None => value.clone(),
    })
}

/// CFGNorm clamps channel norms in denoised space after classifier-free guidance.
pub fn denoised_cfg(x: &Tensor, positive: &Tensor, negative: Option<&Tensor>, sigma: f32, cfg: f32) -> Result<Tensor> {
    let pos = (x - (positive.to_dtype(DType::F32)? * sigma as f64)?)?;
    let Some(negative) = negative else { return Ok(pos) };
    let neg = (x - (negative.to_dtype(DType::F32)? * sigma as f64)?)?;
    let guided = (&neg + ((&pos - &neg)? * cfg as f64)?)?;
    let pn = pos.sqr()?.sum_keepdim(1)?.sqrt()?;
    let gn = (guided.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-8)?;
    Ok(guided.broadcast_mul(&pn.broadcast_div(&gn)?.clamp(0., 1.)?)?)
}

/// Convert graph RGB pixels without introducing a second quantization.
pub fn pixels(rgb: &RgbImage) -> Result<Tensor> {
    Ok(Tensor::from_vec(rgb.as_raw().iter().map(|v| *v as f32 / 255.).collect::<Vec<_>>(), (rgb.height() as usize, rgb.width() as usize, 3), &Device::Cpu)?)
}

/// Kontext uses byte RGB internally, as the graph's Pillow resizer does.
pub fn kontext(rgb: &RgbImage) -> Result<RgbImage> {
    let ((w, h), (x, y, cw, ch)) = crate::edit2511_prep::kontext_geometry(rgb.height() as usize, rgb.width() as usize)?;
    if x == 0 && y == 0 && cw == w && ch == h {
        return Ok(rgb.clone());
    }
    // A seed change repeats the same graph resize before the conditioning cache lookup.
    static CACHE: std::sync::OnceLock<Mutex<Option<(RgbImage, RgbImage)>>> = std::sync::OnceLock::new();
    let mut slot = CACHE.get_or_init(|| Mutex::new(None)).lock().map_err(|_| anyhow::anyhow!("2511 Kontext lock poisoned"))?;
    if let Some((input, output)) = slot.as_ref() {
        if input == rgb {
            return Ok(output.clone());
        }
    }
    let output = rgb_image(&resize_kontext(&pixels(rgb)?)?)?;
    if rgb.as_raw().len() <= 64 * 1024 * 1024 {
        *slot = Some((rgb.clone(), output.clone()));
    } else {
        *slot = None;
    }
    Ok(output)
}

fn rgb_image(hwc: &Tensor) -> Result<RgbImage> {
    let (h, w, _) = hwc.dims3()?;
    let bytes = (hwc.clamp(0., 1.)? * 255.)?.to_dtype(DType::U8)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    RgbImage::from_raw(w as u32, h as u32, bytes).ok_or_else(|| anyhow::anyhow!("invalid RGB bytes"))
}

/// Keep the DiT for seed changes; release it before uncached encoder work.
pub struct Edit2511Pipeline {
    files: Edit2511Files,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    conditioning: Mutex<Option<(ConditioningKey, Conditioning)>>,
    binding: Mutex<Option<(LoraStack, std::collections::BTreeMap<String, Tensor>)>>,
    target: Mutex<Option<(RgbImage, Tensor)>>,
    resident: Mutex<Option<(LoraStack, LayeredDit)>>,
}
impl Edit2511Pipeline {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(root, crate::device::best_device()?)
    }
    pub fn load_on_device(root: impl AsRef<Path>, device: Device) -> Result<Self> {
        let files = Edit2511Files::resolve(&root)?;
        let tokenizer = crate::comfy_files::tokenizer_dir(root.as_ref()).ok_or_else(|| anyhow::anyhow!("Qwen tokenizer not found"))?;
        Ok(Self {
            files,
            tokenizer: load_tokenizer(tokenizer)?,
            dtype: if device.is_cuda() { DType::BF16 } else { DType::F32 },
            device,
            conditioning: Mutex::new(None),
            binding: Mutex::new(None),
            target: Mutex::new(None),
            resident: Mutex::new(None),
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    fn vae(&self) -> Result<WanVae> {
        let vae = WanVae::load(TensorFile::open(&self.files.vae)?.builder("", self.dtype, &self.device)?)?;
        anyhow::ensure!(vae.image_channels() == 3 && vae.can_encode(), "2511 requires the RGB VAE");
        Ok(vae)
    }
    fn encode(&self, vae: &WanVae, hwc: &Tensor) -> Result<Tensor> {
        let input = ((hwc.to_device(&self.device)?.permute((2, 0, 1))?.unsqueeze(0)? * 2.)? - 1.)?;
        Ok(vae.encode(&input.contiguous()?)?.unsqueeze(2)?)
    }
    fn target(&self, rgb: &RgbImage) -> Result<Tensor> {
        let mut cached = self.target.lock().map_err(|_| anyhow::anyhow!("2511 target lock poisoned"))?;
        if let Some((old, latent)) = cached.as_ref() {
            if old == rgb {
                return Ok(latent.clone());
            }
        }
        *cached = None;
        let latent = self.encode(&self.vae()?, &crop_vae_target(&pixels(rgb)?)?)?.to_dtype(DType::F32)?;
        if rgb.as_raw().len() <= 64 * 1024 * 1024 {
            *cached = Some((rgb.clone(), latent.clone()));
        }
        Ok(latent)
    }

    fn conditioning(&self, req: &Edit2511Request, check: &impl Fn() -> Result<()>) -> Result<Conditioning> {
        let mut slot = self.conditioning.lock().map_err(|_| anyhow::anyhow!("2511 conditioning lock poisoned"))?;
        let key = ConditioningKey::new(req);
        if let Some((old, value)) = slot.as_ref() {
            if !req.references.is_empty() && old == &key {
                return Ok(value.clone());
            }
        }
        *slot = None;
        let mut mark = phase_marks(&self.device);
        let references = {
            let vae = self.vae()?;
            req.references
                .iter()
                .map(|rgb| {
                    check()?;
                    self.encode(&vae, &resize_vae_reference(&pixels(rgb)?)?)
                })
                .collect::<Result<Vec<_>>>()?
        };
        crate::device::trim_pool(&self.device);
        mark("reference VAE load + encode")?;
        let pictures = {
            let vision = Qwen25Vision::from_file(&self.files.text_encoder)?;
            req.references
                .iter()
                .map(|rgb| {
                    check()?;
                    vision.encode_pixels_on_device(&resize_editplus(rgb)?, &self.device)
                })
                .collect::<Result<Vec<_>>>()?
        };
        crate::device::trim_pool(&self.device);
        mark("vision load + encode")?;
        check()?;
        let (positive, negative) = {
            let text = Qwen25TextEncoder::from_edit_file(&self.files.text_encoder, self.dtype, &self.device)?;
            let positive = text.encode_edit(&self.tokenizer, &req.prompt, &pictures)?.0;
            check()?;
            let negative = if req.cfg > 1. { Some(text.encode_edit(&self.tokenizer, &req.negative, &pictures)?.0) } else { None };
            (positive, negative)
        };
        drop(pictures);
        crate::device::trim_pool(&self.device);
        mark("text load + encode")?;
        let value = Conditioning { references, positive, negative };
        let bytes = req.references.iter().map(|r| r.as_raw().len()).sum::<usize>().saturating_add(req.prompt.len()).saturating_add(req.negative.len());
        if !req.references.is_empty() && bytes <= 64 * 1024 * 1024 {
            *slot = Some((key, value.clone()));
        }
        Ok(value)
    }
    pub fn generate(&self, req: &Edit2511Request, progress: impl FnMut(usize, usize), cancel: &AtomicBool) -> Result<RgbImage> {
        self.generate_observed(req, progress, cancel, |_, _| Ok(()))
    }
    /// Expose sampler boundaries for oracle-velocity and full-route comparisons.
    pub fn generate_observed(
        &self,
        req: &Edit2511Request,
        mut progress: impl FnMut(usize, usize),
        cancel: &AtomicBool,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<RgbImage> {
        let check = || {
            anyhow::ensure!(!cancel.load(Ordering::Relaxed), "2511 cancelled");
            Ok(())
        };
        check()?;
        anyhow::ensure!(req.references.len() <= 3 && req.cfg.is_finite() && req.cfg > 0., "invalid 2511 references or CFG");
        let schedule = edit2511_schedule(req.steps)?;
        let mut mark = phase_marks(&self.device);
        let mut resident = self.resident.lock().map_err(|_| anyhow::anyhow!("2511 resident lock poisoned"))?;
        let conditioning_hit = self
            .conditioning
            .lock()
            .map_err(|_| anyhow::anyhow!("2511 conditioning lock poisoned"))?
            .as_ref()
            .is_some_and(|(key, _)| !req.references.is_empty() && key == &ConditioningKey::new(req));
        let target_hit = match &req.target {
            None => true,
            Some(rgb) => self.target.lock().map_err(|_| anyhow::anyhow!("2511 target lock poisoned"))?.as_ref().is_some_and(|(old, _)| old == rgb),
        };
        if std::env::var("QIR_EDIT2511_RESIDENT").is_ok_and(|v| v == "0")
            || !conditioning_hit
            || !target_hit
            || resident.as_ref().is_some_and(|(stack, _)| !stack.same_adapters(&req.loras))
        {
            *resident = None;
            crate::device::trim_pool(&self.device);
        }
        mark("residency check/release")?;
        let target = if let Some(rgb) = &req.target {
            self.target(rgb)?
        } else {
            anyhow::ensure!(
                req.width >= 16 && req.height >= 16 && req.width.is_multiple_of(16) && req.height.is_multiple_of(16),
                "empty target requires dimensions divisible by 16"
            );
            // ComfyUI deliberately skips process_in when the latent is entirely zero.
            Tensor::zeros((1, 16, 1, req.height / 8, req.width / 8), DType::F32, &self.device)?
        };
        crate::device::trim_pool(&self.device);
        mark("target VAE encode/cache")?;
        let cond = self.conditioning(req, &check)?;
        mark("conditioning/cache")?;
        let noise = Tensor::from_vec(torch_randn(req.seed, target.elem_count()).map_err(anyhow::Error::msg)?, target.shape(), &self.device)?;
        observe("noise", &noise)?;
        observe("target", &target)?;
        let mask = req
            .mask
            .as_ref()
            .map(|m| latent_noise_mask(m, &[1, 16, target.dim(3)?, target.dim(4)?])?.unsqueeze(2)?.to_device(&self.device).map_err(anyhow::Error::from))
            .transpose()?;
        let initial = schedule.sigma(0);
        let mut latent = ((&noise * initial as f64)? + (&target * (1. - initial) as f64)?)?;
        observe("initial", &latent)?;
        progress(0, req.steps);
        {
            check()?;
            mark("sampler prepare")?;
            if resident.is_none() {
                let mut binding = self.binding.lock().map_err(|_| anyhow::anyhow!("2511 binding lock poisoned"))?;
                let hit = binding.as_ref().is_some_and(|(stack, _)| stack.same_adapters(&req.loras));
                if !hit {
                    *binding = None;
                }
                let patches = if let Some((_, weights)) = binding.as_ref() { weights.clone() } else { std::collections::BTreeMap::new() };
                let mut dit = LayeredDit::load(crate::model::qwen_image_dit::checkpoint_builder_patched(&self.files.diffusion, &self.device, patches)?)?;
                if !hit {
                    mark("DiT load")?;
                    dit.bind_loras(&req.loras)?;
                    let weights = dit.materialized_weights()?;
                    // Repack materialized allocations before retaining the model across requests.
                    drop(dit);
                    crate::device::trim_pool(&self.device);
                    dit = LayeredDit::load(crate::model::qwen_image_dit::checkpoint_builder_patched(&self.files.diffusion, &self.device, weights.clone())?)?;
                    *binding = Some((req.loras.clone(), weights));
                }
                mark(if hit { "DiT load" } else { "LoRA materialize + snapshot" })?;
                drop(binding);
                *resident = Some((req.loras.clone(), dit));
            } else {
                mark("DiT resident hit")?;
            }
            let dit = &resident.as_ref().ok_or_else(|| anyhow::anyhow!("missing 2511 DiT"))?.1;
            crate::device::trim_pool(&self.device);
            let refs = cond.references.iter().map(|r| r.to_dtype(self.dtype).map_err(Into::into)).collect::<Result<Vec<_>>>()?;
            #[cfg(feature = "cuda")]
            let pool = crate::layered_pipeline::LayeredPool::retain(&self.device);
            mark("denoise prepare")?;
            for i in 0..req.steps {
                check()?;
                let sigma = schedule.sigma(i);
                let protected = ((&noise * sigma as f64)? + (&target * (1. - sigma) as f64)?)?;
                let input = masked_replace(&latent, &protected, mask.as_ref())?;
                observe(&format!("step_{i}.input"), &input)?;
                let t = Tensor::from_slice(&[sigma], 1, &self.device)?;
                let x = input.to_dtype(self.dtype)?;
                let positive = dit.forward_latents(&x, &refs, &cond.positive, &t)?;
                observe(&format!("step_{i}.positive"), &positive)?;
                check()?;
                let negative = cond.negative.as_ref().map(|n| dit.forward_latents(&x, &refs, n, &t)).transpose()?;
                if let Some(n) = &negative {
                    observe(&format!("step_{i}.negative"), n)?;
                }
                let denoised = denoised_cfg(&input, &positive, negative.as_ref(), sigma, req.cfg)?;
                let denoised = masked_replace(&denoised, &target, mask.as_ref())?;
                observe(&format!("step_{i}.denoised"), &denoised)?;
                // KSampler's derivative uses the unmasked outer state.
                let derivative = ((&latent - denoised)? / sigma as f64)?;
                latent = (&latent + (derivative * (schedule.sigmas[i + 1] - sigma) as f64)?)?;
                observe(&format!("step_{i}.latent"), &latent)?;
                #[cfg(feature = "cuda")]
                if pool.is_some() {
                    self.device.synchronize()?;
                }
                mark(&format!("step {i}"))?;
                crate::profile::report(&format!("2511 step {i}"));
                progress(i + 1, req.steps);
            }
            self.device.synchronize()?;
        }
        crate::device::trim_pool(&self.device);
        check()?;
        mark("denoise workspace release")?;
        let vae = self.vae()?;
        let decoded = vae.decode(&vae.denormalize(&latent.squeeze(2)?)?)?;
        observe("decoded", &decoded)?;
        let image = rgb_image(&((decoded.squeeze(0)?.permute((1, 2, 0))? + 1.)? / 2.)?)?;
        mark("VAE load + decode + readback")?;
        Ok(image)
    }
}

/// Keep adapter files off CUDA during encoder staging.
pub fn cpu_device() -> Device {
    Device::Cpu
}
/// LowVramPatch casts adapter matrices to bf16 before merging CUDA weights.
pub fn adapter_dtype() -> DType {
    DType::BF16
}

/// Dynamic patch seeds use ComfyUI's CRC32 of the module name without `.weight`.
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
    use super::*;
    #[test]
    fn kontext_identity_and_cached_resize_preserve_rgb_bytes() -> Result<()> {
        for size in [1024u32, 768] {
            let image = RgbImage::from_fn(size, size, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]));
            let expected = rgb_image(&resize_kontext(&pixels(&image)?)?)?;
            assert_eq!(kontext(&image)?, expected);
            assert_eq!(kontext(&image)?, expected);
            let mut changed = image.clone();
            changed.put_pixel(size / 2, size / 2, image::Rgb([3, 7, 11]));
            let expected = rgb_image(&resize_kontext(&pixels(&changed)?)?)?;
            assert_eq!(kontext(&changed)?, expected);
        }
        Ok(())
    }

    #[test]
    fn conditioning_key_tracks_only_encoder_inputs() {
        let mut req = Edit2511Request {
            prompt: "change sky".into(),
            negative: "bad".into(),
            references: vec![RgbImage::new(8, 8)],
            target: None,
            mask: None,
            width: 1024,
            height: 1024,
            seed: 7,
            steps: 8,
            cfg: 1.,
            loras: LoraStack::default(),
        };
        let key = ConditioningKey::new(&req);
        req.seed = 8;
        req.steps = 40;
        req.width = 512;
        req.negative = "ignored".into();
        req.mask = Some(RgbImage::new(4, 4));
        assert!(key == ConditioningKey::new(&req));
        req.cfg = 4.;
        assert!(key != ConditioningKey::new(&req));
        let guided = ConditioningKey::new(&req);
        req.cfg = 2.;
        assert!(guided == ConditioningKey::new(&req));
        req.negative = "different".into();
        assert!(guided != ConditioningKey::new(&req));
        req.cfg = 1.;
        req.references[0].put_pixel(0, 0, image::Rgb([1, 0, 0]));
        assert!(key != ConditioningKey::new(&req));
    }
}

/// Synchronize only phase boundaries when requested.
fn phase_marks(device: &Device) -> impl FnMut(&str) -> Result<()> + '_ {
    let enabled = ["QIR_PROFILE", "QIR_PHASE_PROFILE"].iter().any(|key| std::env::var(key).is_ok_and(|v| v == "1"));
    let mut last = std::time::Instant::now();
    move |phase| {
        if enabled {
            device.synchronize()?;
            eprintln!("2511 phase {phase}: {:.6} s", last.elapsed().as_secs_f64());
            last = std::time::Instant::now();
        }
        Ok(())
    }
}
