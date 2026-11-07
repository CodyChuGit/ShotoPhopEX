//! MOSS Turbo and MOSS Edit Turbo on MLX: Kromoss with its 8-step DMD2 LoRA (or, editing,
//! MOSS-Image-SFT with the MOSS Edit LoRA and the DMD2 LoRA stacked), 8 Euler steps, CFG 1.
//!
//! `cargo run --release -p gn-mlx --example moss -- <models dir> <out dir> <seed> <size>... -- <prompt>`
//! writes `<out dir>/moss-<size>-<seed>.png` for each square size (a size may repeat: the
//! second is warm; with `MOSS_EDIT=<picture.png>` it edits that picture and a size of 0 keeps
//! its aspect at about 1 MP). Converted weights are cached in `<models dir>/.qir-cache/mlx` (where the app's worker keeps them). Each
//! picture's timings go to stdout; the first is cold (the encoder and the DiT loaded), the rest
//! warm (the DiT resident, the prompt encoded).

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::time::Instant;

    use gn_mlx::convert::Precision;
    use gn_mlx::pipeline::{LoraSpec, MossMlx, RefPicture, Request};
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
    let mut moss = MossMlx::new(&models, &models.join(".qir-cache").join("mlx"))?;
    // `MOSS_DIT=f16|q4`: the DiT unquantized (10 GB, the reference) or 4-bit (3 GB); 8-bit else.
    let precision = match std::env::var("MOSS_DIT").as_deref() {
        Ok("f16") => Precision::F16,
        Ok("q4") => Precision::Q4,
        _ => Precision::Q8,
    };
    moss.set_dit_precision(precision);
    // `MOSS_TURBO_LORA=<path>`: another few-step LoRA in place of Kromoss' DMD2.
    let turbo = std::env::var("MOSS_TURBO_LORA").map_or_else(|_| models.join("loras/kromoss-8-steps-dmd2-lora.safetensors"), PathBuf::from);
    let dmd2 = LoraSpec { path: turbo, strength: 1.0 };
    let (variant, loras, references, kind) = match std::env::var("MOSS_EDIT") {
        Ok(picture) => {
            let img = image::open(&picture)?.to_rgba8();
            let r = RefPicture { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw() };
            // `MOSS_EDIT_LORA=<path>`: another edit LoRA in place of chinoll's.
            let edit_path = std::env::var("MOSS_EDIT_LORA").map_or_else(|_| models.join("loras/moss-image-edit/adapter_model.safetensors"), PathBuf::from);
            let edit = LoraSpec { path: edit_path, strength: 1.0 };
            // `MOSS_EDIT_FULL=1`: MOSS Edit alone (30 steps, CFG 4), not the turbo.
            // `MOSS_TURBO_LORA=none`: the edit LoRA alone at the turbo's 8 steps, CFG 1.
            if std::env::var("MOSS_EDIT_FULL").is_ok_and(|v| v == "1") {
                (Variant::Sft, vec![edit], vec![r], "edit-full")
            } else if std::env::var("MOSS_TURBO_LORA").is_ok_and(|v| v == "none") {
                (Variant::Sft, vec![edit], vec![r], "edit")
            } else {
                (Variant::Sft, vec![edit, dmd2], vec![r], "edit")
            }
        }
        // `MOSS_VARIANT=sft`: MOSS-Image-SFT in place of Kromoss.
        Err(_) if std::env::var("MOSS_VARIANT").is_ok_and(|v| v == "sft") => (Variant::Sft, vec![dmd2], Vec::new(), "moss-sft"),
        Err(_) => (Variant::Kromoss, vec![dmd2], Vec::new(), "moss"),
    };
    let tag = if precision == Precision::Q8 { String::new() } else { format!("-{}", precision.tag()) };
    println!("pipeline ready (tokenizer, VAE): {:.1} s", t0.elapsed().as_secs_f64());
    for size in sizes {
        let size: usize = size.parse()?;
        let req = Request {
            prompt: prompt.clone(),
            negative: String::new(),
            width: size,
            height: size,
            // `MOSS_STEPS` / `MOSS_CFG` override the template's.
            steps: std::env::var("MOSS_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(if kind == "edit-full" { 30 } else { 8 }),
            guidance: std::env::var("MOSS_CFG").ok().and_then(|v| v.parse().ok()).unwrap_or(if kind == "edit-full" { 4.0 } else { 1.0 }),
            seed,
            references: references.clone(),
            mask: None,
        };
        let t0 = Instant::now();
        let mut last = Instant::now();
        let mut per_step = Vec::new();
        let pic = moss.generate(
            variant,
            &loras,
            &req,
            &mut |i, _| {
                if i > 0 {
                    per_step.push(last.elapsed().as_secs_f32());
                }
                last = Instant::now();
            },
            &|| false,
        )?;
        let total = t0.elapsed().as_secs_f64();
        let t = pic.timings;
        println!(
            "{}x{}: total {total:.1} s (encode {:.1} s, DiT load + references {:.1} s, 8 steps {:.1} s, decode {:.1} s); steps {:?} s; MLX peak {:.2} GB",
            pic.width,
            pic.height,
            t.encode_s,
            t.load_s,
            t.steps_s,
            t.decode_s,
            per_step.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>(),
            mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9
        );
        let path = out.join(format!("{kind}-{size}-{seed}{tag}.png"));
        image::RgbaImage::from_raw(pic.width as u32, pic.height as u32, pic.rgba)
            .ok_or_else(|| anyhow::anyhow!("a picture's pixels do not match its size"))?
            .save(&path)?;
        println!("  wrote {}", path.display());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
