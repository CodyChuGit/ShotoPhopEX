//! MOSS on MLX against qir on the CPU (f32, the reference), on the real weights.
//!
//! `cargo run --release -p gn-mlx --example moss_check -- <check> <models dir>`, the models dir
//! being a ComfyUI `models` folder with the MOSS files. Checks:
//! - `load`: the whole Kromoss DiT (and its 8-step DMD2 LoRA) to 4-bit; time, memory, and the
//!   converted cache's write and reload (`<models>/mlx_cache`).
//! - `dit [layers]`: one forward of a cut-down DiT (its first `layers` blocks, default 4: qir's
//!   f32 reference of all 42 needs 20 GB) at 256x256, MLX f16 and 4-bit, without and with the
//!   LoRA.
//! - `text [layers]`: the encoder's hidden state for a fixed prompt, its first `layers` layers
//!   (default 4: three DeltaNet, one full attention), MLX f16, 8-bit and 4-bit; then the whole
//!   encoder in 8-bit on MLX alone (time, memory).
//! - `edit [layers]`: MOSS Edit's pieces: the FLUX.2 encoder on a synthetic picture (MLX f32
//!   and f16 against qir), then a cut-down MOSS-Image-SFT DiT with the MOSS Edit and DMD2 LoRAs
//!   attending to that picture's reference tokens (qir's `prepare_references` and
//!   `forward_refs`), MLX f16 and 8-bit.
//! - `vae`: the FLUX.2 small decoder on a fixed latent at 512x512, MLX f32 and f16 against qir
//!   (PSNR); then a 1024x1024 decode on MLX (time, memory).

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    run::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
mod run {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};
    use gn_mlx::convert::{self, Lora, Precision};
    use gn_mlx::dit::{self, MossDit};
    use gn_mlx::nn::{compare, from_candle, to_vec};
    use mlx_rs::{Array, Dtype};
    use qwen_image_rs::lora::{LoraFile, LoraStack};
    use qwen_image_rs::moss::dit::MossConfig;
    use qwen_image_rs::moss::weights::Weights;

    const DIT: &str = "diffusion_models/kromoss_int8_convrot.safetensors";
    const TEXT: &str = "text_encoders/qwen3.5_9b_int8_convrot.safetensors";
    const VAE: &str = "vae/flux2-vae_small.safetensors";
    const TOKENIZER: &str = "text_encoders/qwen35_tokenizer/tokenizer.json";
    pub const PROMPT: &str = "a red apple, a yellow rubber duck and a white mug on a wooden table";
    const SFT: &str = "diffusion_models/moss_image_sft_int8_convrot.safetensors";
    const EDIT_LORA: &str = "loras/moss-image-edit/adapter_model.safetensors";
    const LORA: &str = "loras/kromoss-8-steps-dmd2-lora.safetensors";

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (Some(check), Some(models)) = (args.first(), args.get(1)) else {
            anyhow::bail!("usage: moss_check <load|dit> <models dir> [layers]");
        };
        let models = PathBuf::from(models);
        gn_mlx::nn::limit_cache()?;
        match check.as_str() {
            "load" => load(&models),
            "vae" => vae_check(&models),
            "edit" => edit_check(&models, args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4)),
            "text" => text_check(&models, args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4)),
            "dit" => dit_check(&models, args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4)),
            other => anyhow::bail!("unknown check {other}"),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / 1e9
    }

    fn mem() -> String {
        format!("MLX active {:.2} GB, peak {:.2} GB", gb(mlx_rs::memory::active_memory().unwrap_or(0)), gb(mlx_rs::memory::peak_memory().unwrap_or(0)))
    }

    fn load(models: &Path) -> Result<()> {
        let cfg = MossConfig::default();
        let t0 = Instant::now();
        let lora = Lora::open(&models.join(LORA), 1.0)?;
        let p = dit::convert(&models.join(DIT), &cfg, Dtype::Float16, Precision::Q4, &[lora])?;
        let bytes: usize = p.values().map(Array::nbytes).sum();
        println!("convert Kromoss + DMD2 to 4-bit: {:.1} s, {} arrays, {:.2} GB; {}", t0.elapsed().as_secs_f64(), p.len(), gb(bytes), mem());
        let cache = models.join("mlx_cache").join("kromoss_int8_convrot.dmd2-1.q4g64.safetensors");
        let t0 = Instant::now();
        convert::save(&p, &cache)?;
        println!("cache write: {:.1} s, {:.2} GB on disk ({})", t0.elapsed().as_secs_f64(), gb(std::fs::metadata(&cache)?.len() as usize), cache.display());
        drop(p);
        mlx_rs::memory::clear_cache()?;
        let t0 = Instant::now();
        let p = convert::load(&cache)?;
        let model = MossDit::load(&cfg, &p, Dtype::Float16)?;
        println!("cache reload: {:.1} s; {}", t0.elapsed().as_secs_f64(), mem());
        drop(model);
        Ok(())
    }

    /// The check's inputs: seed-42 noise `(1, 128, 16, 16)` and a fixed `ctx (32, 4096)`.
    fn inputs() -> Result<(Tensor, Tensor)> {
        let noise = qwen_image_rs::comfy::torch_randn(42, 128 * 16 * 16).map_err(anyhow::Error::msg)?;
        let ctx = qwen_image_rs::comfy::torch_randn(7, 32 * 4096).map_err(anyhow::Error::msg)?;
        Ok((Tensor::from_vec(noise, (1, 128, 16, 16), &Device::Cpu)?, Tensor::from_vec(ctx, (32, 4096), &Device::Cpu)?))
    }

    fn dit_check(models: &Path, layers: usize) -> Result<()> {
        let cfg = MossConfig { layers, ..MossConfig::default() };
        let (x, ctx) = inputs()?;
        let path = models.join(DIT);
        let t = 500.0; // σ = 0.5
        // `MOSS_CHECK_LORA_STRENGTH` exaggerates the LoRA (its own effect is small) to check the merge.
        let strength: f32 = std::env::var("MOSS_CHECK_LORA_STRENGTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let mut plain: Option<Vec<f32>> = None;
        for with_lora in [false, true] {
            let label = if with_lora { format!("with DMD2 LoRA x{strength}") } else { "no LoRA".to_string() };
            // qir, f32 on the CPU.
            let t0 = Instant::now();
            let w = Weights::open(&path, &Device::Cpu, DType::F32, false)?;
            let reference = qwen_image_rs::moss::dit::MossDit::load(&cfg, &w)?;
            if with_lora {
                let file = LoraFile::load(&models.join(LORA), &Device::Cpu, DType::F32)?;
                let unmatched = reference.set_loras(&LoraStack { layers: vec![(Arc::new(file), strength)] })?;
                // The cut-down model lacks the file's later blocks.
                let foreign = unmatched
                    .iter()
                    .filter(|m| !m.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok()).is_some_and(|i| i >= layers))
                    .count();
                anyhow::ensure!(foreign == 0, "{foreign} LoRA modules qir does not know");
            }
            let want = reference.forward(&x, t, &ctx, &[])?.flatten_all()?.to_vec1::<f32>()?;
            println!("[{label}] qir CPU f32, {layers} blocks: {:.1} s", t0.elapsed().as_secs_f64());
            match &plain {
                Some(p) => {
                    let (cos, max) = compare(p, &want)?;
                    println!("[{label}] the LoRA's effect on qir's output: cosine {cos:.6} to no LoRA, max abs {max:.4}");
                }
                None => plain = Some(want.clone()),
            }
            drop(reference);
            drop(w);
            let lora = if with_lora { vec![Lora::open(&models.join(LORA), strength)?] } else { Vec::new() };
            for precision in [Precision::F16, Precision::Q4] {
                let p = dit::convert(&path, &cfg, Dtype::Float16, precision, &lora)?;
                let model = MossDit::load(&cfg, &p, Dtype::Float16)?;
                let (xm, cm) = (from_candle(&x)?, from_candle(&ctx)?);
                let v = model.forward(&xm, t, &cm, &[], None)?;
                v.eval()?;
                let t0 = Instant::now();
                let got = to_vec(&model.forward(&xm, t, &cm, &[], None)?)?;
                let (cos, max) = compare(&got, &want)?;
                let finite = got.iter().all(|v| v.is_finite());
                println!(
                    "[{label}] MLX {}: cosine {cos:.6}, max abs {max:.4} (|v| max {:.3}), finite {finite}, warm forward {:.3} s",
                    precision.tag(),
                    want.iter().fold(0f32, |m, v| m.max(v.abs())),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        Ok(())
    }

    fn text_check(models: &Path, layers: usize) -> Result<()> {
        use qwen_image_rs::moss::text::{self as qtext, TextConfig};
        let tok = tokenizers::Tokenizer::from_file(models.join(TOKENIZER)).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let ids = tok.encode(qtext::template(PROMPT), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec();
        println!("prompt: {} tokens", ids.len());
        let path = models.join(TEXT);
        let cfg = TextConfig { layers, ..TextConfig::qwen35_9b() };
        let t0 = Instant::now();
        let w = Weights::open(&path, &Device::Cpu, DType::F32, false)?;
        let reference = qtext::Qwen35Text::load(&cfg, &w)?;
        let want = reference.forward(&ids)?.flatten_all()?.to_vec1::<f32>()?;
        println!("qir CPU f32, {layers} layers: {:.1} s", t0.elapsed().as_secs_f64());
        drop(reference);
        drop(w);
        for precision in [Precision::F16, Precision::Q8, Precision::Q6, Precision::Q5, Precision::Q4] {
            let label = precision.tag();
            let model = gn_mlx::text::Qwen35Text::load(&cfg, gn_mlx::text::convert(&path, &cfg, Dtype::Float16, precision)?, Dtype::Float16)?;
            to_vec(&model.forward(&ids)?)?;
            let t0 = Instant::now();
            let got = to_vec(&model.forward(&ids)?)?;
            let (cos, max) = compare(&got, &want)?;
            println!(
                "MLX {label}: cosine {cos:.6}, max abs {max:.4} (|h| max {:.1}), finite {}, warm encode {:.3} s",
                want.iter().fold(0f32, |m, v| m.max(v.abs())),
                got.iter().all(|v| v.is_finite()),
                t0.elapsed().as_secs_f64()
            );
        }
        mlx_rs::memory::clear_cache()?;
        mlx_rs::memory::reset_peak_memory()?;
        let full = TextConfig::qwen35_9b();
        let t0 = Instant::now();
        let model = gn_mlx::text::Qwen35Text::load(&full, gn_mlx::text::convert(&path, &full, Dtype::Float16, Precision::Q8)?, Dtype::Float16)?;
        println!("whole encoder to 8-bit: {:.1} s; {}", t0.elapsed().as_secs_f64(), mem());
        let t0 = Instant::now();
        let h = to_vec(&model.forward(&ids)?)?;
        println!("whole encoder, cold encode: {:.3} s, finite {}", t0.elapsed().as_secs_f64(), h.iter().all(|v| v.is_finite()));
        let t0 = Instant::now();
        to_vec(&model.forward(&ids)?)?;
        println!("whole encoder, warm encode: {:.3} s; {}", t0.elapsed().as_secs_f64(), mem());
        Ok(())
    }

    fn vae_check(models: &Path) -> Result<()> {
        use gn_mlx::nn::psnr;
        use gn_mlx::vae::{self, Flux2Decoder};
        let path = models.join(VAE);
        let z = Tensor::from_vec(qwen_image_rs::comfy::torch_randn(3, 128 * 32 * 32).map_err(anyhow::Error::msg)?, (1, 128, 32, 32), &Device::Cpu)?;
        let t0 = Instant::now();
        let vb = candle_nn::VarBuilder::from_tensors(candle_core::safetensors::load(&path, &Device::Cpu)?, DType::F32, &Device::Cpu);
        let want = qwen_image_rs::moss::vae::Flux2Decoder::load(vb)?.decode(&z)?.flatten_all()?.to_vec1::<f32>()?;
        println!("qir CPU f32 decode 512x512: {:.1} s", t0.elapsed().as_secs_f64());
        for dt in [Dtype::Float32, Dtype::Float16] {
            let dec = Flux2Decoder::load(&vae::convert(&path, dt)?, dt)?;
            let zm = from_candle(&z)?;
            to_vec(&dec.decode(&zm)?)?;
            let t0 = Instant::now();
            let got = to_vec(&dec.decode(&zm)?)?;
            let (cos, max) = compare(&got, &want)?;
            println!(
                "MLX {dt:?} 512x512: PSNR {:.1} dB, cosine {cos:.6}, max abs {max:.4}, warm decode {:.3} s",
                psnr(&got, &want)?,
                t0.elapsed().as_secs_f64()
            );
        }
        for dt in [Dtype::Float32, Dtype::Float16] {
            mlx_rs::memory::clear_cache()?;
            mlx_rs::memory::reset_peak_memory()?;
            let dec = Flux2Decoder::load(&vae::convert(&path, dt)?, dt)?;
            let z = Array::from_slice(&qwen_image_rs::comfy::torch_randn(5, 128 * 64 * 64).map_err(anyhow::Error::msg)?, &[1, 128, 64, 64]);
            let t0 = Instant::now();
            to_vec(&dec.decode(&z)?)?;
            println!("MLX {dt:?} 1024x1024 cold decode {:.2} s; {}", t0.elapsed().as_secs_f64(), mem());
            let t0 = Instant::now();
            to_vec(&dec.decode(&z)?)?;
            println!("MLX {dt:?} 1024x1024 warm decode {:.2} s; {}", t0.elapsed().as_secs_f64(), mem());
        }
        Ok(())
    }

    fn edit_check(models: &Path, layers: usize) -> Result<()> {
        use gn_mlx::vae::{self, Flux2Encoder};
        // A synthetic 256x256 picture: gradients and a disc.
        let (w, h) = (256usize, 256usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let disc = ((x as f32 - 150.0).powi(2) + (y as f32 - 110.0).powi(2)) < 60.0f32.powi(2);
                rgba.extend([if disc { 220 } else { (x * 255 / w) as u8 }, (y * 255 / h) as u8, if disc { 40 } else { 128 }, 255]);
            }
        }
        let path = models.join(VAE);
        let vb = candle_nn::VarBuilder::from_tensors(candle_core::safetensors::load(&path, &Device::Cpu)?, DType::F32, &Device::Cpu);
        let qenc = qwen_image_rs::moss::vae::Flux2Encoder::load(vb)?;
        let qimg = qwen_image_rs::moss::vae::from_rgba_u8(&rgba, w, h, &Device::Cpu)?;
        let latent = qenc.encode(&qimg)?; // (1, 128, 16, 16)
        let want = latent.flatten_all()?.to_vec1::<f32>()?;
        for dt in [Dtype::Float32, Dtype::Float16] {
            let enc = Flux2Encoder::load(&vae::convert(&path, dt)?, dt)?;
            let got = to_vec(&enc.encode(&vae::from_rgba_u8(&rgba, w, h)?)?)?;
            let (cos, max) = compare(&got, &want)?;
            println!("encoder MLX {dt:?}: cosine {cos:.6}, max abs {max:.4} (|z| max {:.2})", want.iter().fold(0f32, |m, v| m.max(v.abs())));
        }
        // The DiT with the edit's two LoRAs, attending to the picture.
        let cfg = MossConfig { layers, ..MossConfig::default() };
        let (x, ctx) = inputs()?;
        let t = 500.0;
        let sft = models.join(SFT);
        let t0 = Instant::now();
        let w8 = Weights::open(&sft, &Device::Cpu, DType::F32, false)?;
        let reference = qwen_image_rs::moss::dit::MossDit::load(&cfg, &w8)?;
        let files = [models.join(EDIT_LORA), models.join(LORA)].map(|p| LoraFile::load(&p, &Device::Cpu, DType::F32).map(|f| (Arc::new(f), 1.0f32)));
        let [a, b] = files;
        reference.set_loras(&LoraStack { layers: vec![a?, b?] })?;
        let cache = reference.prepare_references(std::slice::from_ref(&latent))?;
        let want = reference.forward_refs(&x, t, &ctx, &[], Some(&cache))?.flatten_all()?.to_vec1::<f32>()?;
        let plain = reference.forward(&x, t, &ctx, &[])?.flatten_all()?.to_vec1::<f32>()?;
        let (cos, max) = compare(&plain, &want)?;
        println!(
            "qir CPU f32, SFT {layers} blocks + edit + DMD2: {:.1} s; the reference's effect: cosine {cos:.6} to none, max abs {max:.3}",
            t0.elapsed().as_secs_f64()
        );
        drop(cache);
        drop(reference);
        drop(w8);
        let loras = vec![Lora::open(&models.join(EDIT_LORA), 1.0)?, Lora::open(&models.join(LORA), 1.0)?];
        for precision in [Precision::F16, Precision::Q8] {
            // The LoRAs unmerged, as the pipeline runs them.
            let p = dit::convert(&sft, &cfg, Dtype::Float16, precision, &[])?;
            let mut model = MossDit::load(&cfg, &p, Dtype::Float16)?;
            model.set_loras(&loras)?;
            let refs = model.prepare_references(&[from_candle(&latent)?])?;
            let got = to_vec(&model.forward(&from_candle(&x)?, t, &from_candle(&ctx)?, &[], Some(&refs))?)?;
            let (cos, max) = compare(&got, &want)?;
            println!(
                "MLX {}: cosine {cos:.6}, max abs {max:.4} (|v| max {:.3}), finite {}",
                precision.tag(),
                want.iter().fold(0f32, |m, v| m.max(v.abs())),
                got.iter().all(|v| v.is_finite())
            );
        }
        Ok(())
    }
}
