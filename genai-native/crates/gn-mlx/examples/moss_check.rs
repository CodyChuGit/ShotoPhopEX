//! MOSS on MLX against qir on the CPU (f32, the reference), on the real weights.
//!
//! `cargo run --release -p gn-mlx --example moss_check -- <check> <models dir>`, the models dir
//! being a ComfyUI `models` folder with the MOSS files. Checks:
//! - `load`: the whole Kromoss DiT (and its 8-step DMD2 LoRA) to 4-bit; time, memory, and the
//!   converted cache's write and reload (`<models>/mlx_cache`).
//! - `dit [layers]`: one forward of a cut-down DiT (its first `layers` blocks, default 4: qir's
//!   f32 reference of all 42 needs 20 GB) at 256x256, MLX f16 and 4-bit, without and with the
//!   LoRA.

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
    const LORA: &str = "loras/kromoss-8-steps-dmd2-lora.safetensors";

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (Some(check), Some(models)) = (args.first(), args.get(1)) else {
            anyhow::bail!("usage: moss_check <load|dit> <models dir> [layers]");
        };
        let models = PathBuf::from(models);
        match check.as_str() {
            "load" => load(&models),
            "dit" => dit_check(&models, args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4)),
            other => anyhow::bail!("unknown check {other}"),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / 1e9
    }

    fn mem() -> String {
        format!(
            "MLX active {:.2} GB, peak {:.2} GB",
            gb(mlx_rs::memory::active_memory().unwrap_or(0)),
            gb(mlx_rs::memory::peak_memory().unwrap_or(0))
        )
    }

    fn load(models: &Path) -> Result<()> {
        let cfg = MossConfig::default();
        let t0 = Instant::now();
        let lora = Lora::open(&models.join(LORA), 1.0)?;
        let p = dit::convert(&models.join(DIT), &cfg, Dtype::Float16, Precision::Q4, Some(&lora))?;
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
                let foreign = unmatched.iter().filter(|m| !m.strip_prefix("layers.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok()).is_some_and(|i| i >= layers)).count();
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
            let lora = if with_lora { Some(Lora::open(&models.join(LORA), strength)?) } else { None };
            for precision in [Precision::F16, Precision::Q4] {
                let p = dit::convert(&path, &cfg, Dtype::Float16, precision, lora.as_ref())?;
                let model = MossDit::load(&cfg, &p, Dtype::Float16)?;
                let (xm, cm) = (from_candle(&x)?, from_candle(&ctx)?);
                let v = model.forward(&xm, t, &cm, &[])?;
                v.eval()?;
                let t0 = Instant::now();
                let got = to_vec(&model.forward(&xm, t, &cm, &[])?)?;
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
}
