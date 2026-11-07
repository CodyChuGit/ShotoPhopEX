//! MOSS Turbo on MLX: Kromoss with its 8-step DMD2 LoRA, 8 Euler steps, CFG 1, shift 5.
//!
//! `cargo run --release -p gn-mlx --example moss -- <models dir> <out dir> <seed> <size>... -- <prompt>`
//! writes `<out dir>/moss-<size>-<seed>.png` for each square size (a size may repeat: the
//! second is warm); converted weights are cached
//! in `<models dir>/mlx_cache`. Each picture's timings go to stdout; the first is cold (the
//! encoder and the DiT loaded), the rest warm (the DiT resident, the prompt encoded).

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::time::Instant;

    use gn_mlx::pipeline::{LoraSpec, MossMlx, Request};
    use qwen_image_rs::moss::pipeline::Variant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let (head, prompt) = (&args[..split], args.get(split + 1..).unwrap_or_default().join(" "));
    let [models, out, seed, sizes @ ..] = head else {
        anyhow::bail!("usage: moss <models dir> <out dir> <seed> <size>... -- <prompt>");
    };
    anyhow::ensure!(!prompt.is_empty() && !sizes.is_empty(), "a prompt and at least one size");
    let (models, out, seed) = (PathBuf::from(models), PathBuf::from(out), seed.parse::<u64>()?);
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let lora = LoraSpec { path: models.join("loras/kromoss-8-steps-dmd2-lora.safetensors"), strength: 1.0 };
    let mut moss = MossMlx::new(&models, Variant::Kromoss, Some(lora), &models.join("mlx_cache"))?;
    // `MOSS_DIT=f16|q4`: the DiT unquantized (10 GB, the reference) or 4-bit (3 GB); 8-bit else.
    use gn_mlx::convert::Precision;
    let precision = match std::env::var("MOSS_DIT").as_deref() {
        Ok("f16") => Precision::F16,
        Ok("q4") => Precision::Q4,
        _ => Precision::Q8,
    };
    moss.set_dit_precision(precision);
    let tag = if precision == Precision::Q8 { String::new() } else { format!("-{}", precision.tag()) };
    println!("pipeline ready (tokenizer, VAE): {:.1} s", t0.elapsed().as_secs_f64());
    for size in sizes {
        let size: usize = size.parse()?;
        let req = Request { prompt: prompt.clone(), negative: String::new(), width: size, height: size, steps: 8, guidance: 1.0, seed };
        let t0 = Instant::now();
        let mut last = Instant::now();
        let mut per_step = Vec::new();
        let pic = moss.generate(&req, &mut |i, _| {
            if i > 0 {
                per_step.push(last.elapsed().as_secs_f32());
            }
            last = Instant::now();
        })?;
        let total = t0.elapsed().as_secs_f64();
        let t = pic.timings;
        println!(
            "{size}x{size}: total {total:.1} s (encode {:.1} s, DiT load {:.1} s, 8 steps {:.1} s, decode {:.1} s); steps {:?} s; MLX peak {:.2} GB",
            t.encode_s,
            t.load_s,
            t.steps_s,
            t.decode_s,
            per_step.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>(),
            mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9
        );
        let path = out.join(format!("moss-{size}-{seed}{tag}.png"));
        image::RgbaImage::from_raw(pic.width as u32, pic.height as u32, pic.rgba).ok_or_else(|| anyhow::anyhow!("a picture's pixels do not match its size"))?.save(&path)?;
        println!("  wrote {}", path.display());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
