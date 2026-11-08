//! FLUX 3 Image Klein on MLX: text to image, an edit, or an expand (with AusBoss's outpaint
//! LoRA), from mflux's pre-quantized folder.
//!
//! `cargo run --release -p gn-mlx --example klein -- <klein dir> <vae file> <out dir> <seed> <size>... -- <prompt>`
//! writes `<out dir>/klein-<size>-<seed>.png` for each square size. `<klein dir>` is mflux's
//! folder (`transformer/`, `text_encoder/`), and `<vae file>` the FLUX.2 VAE
//! (`vae/flux2-vae_small.safetensors`). Optional settings:
//! - `KLEIN_EDIT=<a.png>[,<b.png>…]` edits those pictures. The first sets the canvas, and a size
//!   of 0 keeps its aspect at about 768².
//! - `KLEIN_LORA=<file>[:<strength>]` adds a LoRA, unmerged.
//! - `KLEIN_MASK=<mask.png>` fills the mask's white on the first picture (Generative Fill).
//! - `KLEIN_STEPS=<n>` sets the step count (0, the default, takes the checkpoint's own).
//!
//! Each picture's timings go to stdout.

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::time::Instant;

    use gn_mlx::klein::KleinMlx;
    use gn_mlx::pipeline::{LoraSpec, RefPicture, Request};

    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let (head, prompt) = (&args[..split], args.get(split + 1..).unwrap_or_default().join(" "));
    let [model, vae, out, seed, sizes @ ..] = head else {
        anyhow::bail!("usage: klein <klein dir> <vae file> <out dir> <seed> <size>... -- <prompt>");
    };
    anyhow::ensure!(!prompt.is_empty() && !sizes.is_empty(), "a prompt and at least one size");
    let (model, out, seed) = (PathBuf::from(model), PathBuf::from(out), seed.parse::<u64>()?);
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let mut klein = KleinMlx::new(&PathBuf::from(vae))?;
    println!("VAE loaded in {:.1} s", t0.elapsed().as_secs_f32());

    let references = match std::env::var("KLEIN_EDIT") {
        Ok(list) => list
            .split(',')
            .map(|p| -> anyhow::Result<RefPicture> {
                let img = image::open(p)?.to_rgba8();
                Ok(RefPicture { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw() })
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        Err(_) => Vec::new(),
    };
    let loras = match std::env::var("KLEIN_LORA") {
        Ok(spec) => {
            let (path, strength) = match spec.rsplit_once(':') {
                Some((p, s)) if s.parse::<f32>().is_ok() => (p.to_string(), s.parse::<f32>()?),
                _ => (spec, 1.0),
            };
            vec![LoraSpec { path: PathBuf::from(path), strength }]
        }
        Err(_) => Vec::new(),
    };
    let steps = std::env::var("KLEIN_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    // `KLEIN_MASK=<mask.png>`: fill the white of the mask on the first picture.
    let mask = match std::env::var("KLEIN_MASK") {
        Ok(p) => {
            let img = image::open(p)?.to_rgba8();
            Some(RefPicture { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw() })
        }
        Err(_) => None,
    };
    let kind = if references.is_empty() { "klein" } else { "klein-edit" };

    for size in sizes {
        let side: usize = size.parse()?;
        let req = Request {
            prompt: prompt.clone(),
            negative: String::new(),
            width: side,
            height: side,
            steps,
            guidance: 4.0,
            seed,
            references: references.clone(),
            mask: mask.clone(),
        };
        let t0 = Instant::now();
        let pic = klein.generate(&model, &loras, &req, &mut |_, _| {}, &|| false)?;
        let total = t0.elapsed().as_secs_f32();
        let path = out.join(format!("{kind}-{size}-{seed}.png"));
        image::RgbaImage::from_raw(pic.width as u32, pic.height as u32, pic.rgba).ok_or_else(|| anyhow::anyhow!("a picture of the wrong size"))?.save(&path)?;
        let t = pic.timings;
        println!(
            "{} ({}x{}): total {total:.1} s = load {:.1} + encode {:.1} + steps {:.1} + decode {:.1}",
            path.display(),
            pic.width,
            pic.height,
            t.load_s,
            t.encode_s,
            t.steps_s,
            t.decode_s
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the klein example runs on macOS (Apple Silicon) only");
}
