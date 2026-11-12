//! FLUX 3 Image Klein on CUDA: text to image, an edit, a fill or an expand (with AusBoss's
//! outpaint LoRA), from ComfyUI's files in a `models` folder.
//!
//! `cargo run --release -p qwen-image-rs --features <cuda features> --example klein -- <models dir> <out dir> <seed> <size>... -- <prompt>`
//! writes `<out dir>/klein-<size>-<seed>.png` for each square size. `<models dir>` holds
//! `diffusion_models/flux3-image-klein-int8-convrot.safetensors` (or the bf16 file),
//! `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (or `qwen3vl_4b_bf16`),
//! `vae/flux2-vae_small.safetensors` and Qwen's tokenizer folder. Optional settings:
//! - `KLEIN_EDIT=<a.png>[,<b.png>…]` edits those pictures. The first sets the canvas, and a size
//!   of 0 keeps its aspect at about 1 MP.
//! - `KLEIN_LORA=<file>[:<strength>]` adds a LoRA, unmerged.
//! - `KLEIN_MASK=<mask.png>` fills the mask's white on the first picture (Generative Fill).
//! - `KLEIN_OUTPAINT=1` with a mask runs the editor's Expand instead (the canvas greyed under
//!   the mask as a full-size reference, no noise mask; use it with the Outpaint LoRA at 1.5).
//! - `KLEIN_STEPS=<n>` sets the step count (0, the default, takes the checkpoint's own).
//! - `KLEIN_BASE=1` runs Base (50 steps, CFG 4) instead of the distilled model.
//!
//! Each picture's timings and the card's peak memory go to stdout.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use qwen_image_rs::klein::pipeline::{find_files, KleinPipeline};
use qwen_image_rs::lora::{LoraFile, LoraStack};
use qwen_image_rs::pipeline::{RefImage, Request, Schedule};

fn picture(path: &str) -> anyhow::Result<RefImage> {
    let img = image::open(path)?.to_rgba8();
    Ok(RefImage { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw(), offset: None })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let (head, prompt) = (&args[..split], args.get(split + 1..).unwrap_or_default().join(" "));
    let [models, out, seed, sizes @ ..] = head else {
        anyhow::bail!("usage: klein <models dir> <out dir> <seed> <size>... -- <prompt>");
    };
    anyhow::ensure!(!prompt.is_empty() && !sizes.is_empty(), "a prompt and at least one size");
    let (models, out, seed) = (PathBuf::from(models), PathBuf::from(out), seed.parse::<u64>()?);
    std::fs::create_dir_all(&out)?;
    let base = std::env::var("KLEIN_BASE").is_ok_and(|v| v.trim() == "1");
    let files = find_files(&models, base).ok_or_else(|| anyhow::anyhow!("FLUX 3 Image Klein's files are not all in {}", models.display()))?;
    println!("files: {files:?}");
    let t0 = Instant::now();
    let klein = KleinPipeline::load(&files, &mut |_, m| println!("  {m}"))?;
    println!("loaded in {:.1} s (8 GB mode: {})", t0.elapsed().as_secs_f32(), klein.low_vram());

    let references = match std::env::var("KLEIN_EDIT") {
        Ok(list) => list.split(',').map(picture).collect::<anyhow::Result<Vec<_>>>()?,
        Err(_) => Vec::new(),
    };
    let mut loras = LoraStack::default();
    if let Ok(spec) = std::env::var("KLEIN_LORA") {
        let (path, strength) = match spec.rsplit_once(':') {
            Some((p, s)) if s.parse::<f32>().is_ok() => (p.to_string(), s.parse::<f32>()?),
            _ => (spec, 1.0),
        };
        loras.layers.push((Arc::new(LoraFile::load(&PathBuf::from(path), klein.device(), klein.lora_dtype())?), strength));
    }
    let steps = std::env::var("KLEIN_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mask = std::env::var("KLEIN_MASK").ok().map(|p| picture(&p)).transpose()?;
    let outpaint = std::env::var("KLEIN_OUTPAINT").is_ok_and(|v| v.trim() == "1");
    let kind = match (&mask, references.is_empty()) {
        (Some(_), _) if outpaint => "klein-expand",
        (Some(_), _) => "klein-fill",
        (None, true) => "klein",
        (None, false) => "klein-edit",
    };

    for size in sizes {
        let side: usize = size.parse()?;
        let req = Request {
            prompt: prompt.clone(),
            negative: String::new(),
            width: side,
            height: side,
            steps,
            seed,
            guidance: 4.0,
            schedule: Schedule::Comfy,
            references: references.clone(),
            loras: loras.clone(),
            ref_boost: 1.0,
        };
        let t0 = Instant::now();
        let pic = match &mask {
            Some(m) if outpaint => klein.outpaint(&req, m, &mut |_, _| {}, &|| false)?,
            _ => klein.generate(&req, mask.as_ref(), &mut |_, _| {}, &|| false)?,
        };
        let total = t0.elapsed().as_secs_f32();
        let path = out.join(format!("{kind}-{size}-{seed}.png"));
        image::RgbaImage::from_raw(pic.width, pic.height, pic.rgba).ok_or_else(|| anyhow::anyhow!("a picture of the wrong size"))?.save(&path)?;
        let t = pic.timings;
        let peak = qwen_image_rs::device::free_vram(klein.device()).map(|(free, total)| (total - free) as f64 / f64::from(1u32 << 30));
        println!(
            "{} ({}x{}): total {total:.1} s = encode {:.1} + steps {:.1} + decode {:.1}; card in use now {:.1} GiB",
            path.display(),
            pic.width,
            pic.height,
            t.encode_s,
            t.denoise_s,
            t.decode_s,
            peak.unwrap_or(0.0)
        );
    }
    Ok(())
}
