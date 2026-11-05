//! Run MOSS-Image (text to image, Kromoss turbo, MOSS Edit) from the installed files, outside the
//! app: a check of the native port. `cargo run --release -p qwen-image-rs --features
//! cuda,convrot,fusednorm,flash,sage2 --example moss_run -- --root <models> --variant sft
//! --prompt "…" [--picture in.png] [--lora file.safetensors[:strength]]… --out out.png`.
//! `QIR_MOSS_DUMP=v.safetensors` saves the first step's velocity; `QIR_MOSS_FUSED=0` takes the
//! candle chains instead of the fused kernels (to compare the two).
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use qwen_image_rs::lora::{LoraFile, LoraStack};
use qwen_image_rs::moss::pipeline::{find_files, MossPipeline, Variant};
use qwen_image_rs::pipeline::{RefImage, Request};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    /// sft, base or kromoss.
    #[arg(long, default_value = "sft")]
    variant: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value = "")]
    negative: String,
    #[arg(long)]
    picture: Vec<PathBuf>,
    /// `path[:strength]`.
    #[arg(long)]
    lora: Vec<String>,
    #[arg(long, default_value_t = 0)]
    width: usize,
    #[arg(long, default_value_t = 0)]
    height: usize,
    #[arg(long, default_value_t = 50)]
    steps: usize,
    #[arg(long, default_value_t = 3.0)]
    cfg: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
    /// Runs of the same request (the first is cold).
    #[arg(long, default_value_t = 1)]
    runs: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let a = Args::parse();
    let variant = match a.variant.as_str() {
        "sft" => Variant::Sft,
        "base" => Variant::Base,
        "kromoss" => Variant::Kromoss,
        v => anyhow::bail!("variant {v}"),
    };
    let files = find_files(&a.root, variant).context("MOSS files not found under the root")?;
    println!("files: {files:?}");
    let t0 = Instant::now();
    let p = MossPipeline::load(&files, &mut |_, _| {})?;
    println!("load {:.2} s, can_edit {}", t0.elapsed().as_secs_f64(), p.can_edit());
    let mut stack = LoraStack::default();
    for l in &a.lora {
        let (path, strength) = match l.rsplit_once(':').filter(|(_, s)| s.parse::<f32>().is_ok()) {
            Some((path, s)) => (path.to_string(), s.parse::<f32>()?),
            None => (l.clone(), 1.0),
        };
        let f = LoraFile::load(std::path::Path::new(&path), p.device(), p.lora_dtype())?;
        println!("lora {path} @ {strength}: {} modules", f.entries.len());
        stack.layers.push((Arc::new(f), strength));
    }
    let mut references = Vec::new();
    for pic in &a.picture {
        let img = image::open(pic)?.to_rgba8();
        references.push(RefImage { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw(), offset: None });
    }
    let req = Request {
        negative: a.negative.clone(),
        width: a.width,
        height: a.height,
        steps: a.steps,
        seed: a.seed,
        guidance: a.cfg,
        references,
        loras: stack,
        ..Request::new(a.prompt.clone())
    };
    let req = if req.references.is_empty() && req.width == 0 { Request { width: 1024, height: 1024, ..req } } else { req };
    for run in 0..a.runs {
        let t = Instant::now();
        let pic = p.generate(&req, &mut |_, _| {}, &|| false)?;
        let tm = &pic.timings;
        println!(
            "run {run}: {:.2} s (encode {:.2}, sampling {:.2}, decode {:.2}) {}x{}  free/total VRAM {:?}",
            t.elapsed().as_secs_f64(),
            tm.encode_s,
            tm.denoise_s,
            tm.decode_s,
            pic.width,
            pic.height,
            qwen_image_rs::device::free_vram(p.device()).map(|(f, t)| (f >> 20, t >> 20))
        );
        if run + 1 == a.runs {
            image::RgbaImage::from_raw(pic.width, pic.height, pic.rgba).context("picture")?.save(&a.out)?;
        }
    }
    Ok(())
}
