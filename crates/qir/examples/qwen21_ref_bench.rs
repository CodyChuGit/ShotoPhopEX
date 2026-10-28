//! Repeated picture requests with fresh seeds, raw parity artifacts and phase timings.
use anyhow::{Context, Result};
use clap::Parser;
use qwen_image_rs::pipeline::{Options, Pipeline, RefImage, Request};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    models: PathBuf,
    #[arg(long)]
    cache: PathBuf,
    #[arg(long)]
    picture: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "image-ref,fill,edit,matte")]
    routes: String,
    #[arg(long, default_value_t = 2)]
    repeats: usize,
    #[arg(long, default_value_t = 0)]
    tile_repeats: usize,
    #[arg(long)]
    text_gguf: Option<PathBuf>,
    #[arg(long)]
    keep_text: bool,
    #[arg(long)]
    lora: Option<PathBuf>,
    #[arg(long, default_value_t = 0.5)]
    lora_strength: f32,
    #[arg(long)]
    check_lora_reset: bool,
}

const RESTORE_PROMPT: &str =
    "Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, keeping the same composition, people and colors.";

fn upscale_tiles(picture: &RefImage) -> Result<Vec<RefImage>> {
    let img = image::RgbaImage::from_raw(picture.width as u32, picture.height as u32, picture.rgba.clone()).context("picture bytes")?;
    let enlarged = image::imageops::resize(&img, 1728, 2304, image::imageops::FilterType::CatmullRom);
    Ok([(0, 0), (736, 0), (0, 864), (736, 864), (0, 1312), (736, 1312)]
        .into_iter()
        .map(|(x, y)| RefImage { width: 992, height: 992, rgba: image::imageops::crop_imm(&enlarged, x, y, 992, 992).to_image().into_raw(), offset: None })
        .collect())
}

fn read(path: &PathBuf) -> Result<RefImage> {
    let t = Instant::now();
    let img = image::open(path)?.to_rgba8();
    tracing::info!(seconds = t.elapsed().as_secs_f64(), "request image decode");
    Ok(RefImage { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw(), offset: None })
}

fn memory_report(device: &candle_core::Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(dev) = device {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let mut pool = std::ptr::null_mut();
        unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, dev.cuda_stream().context().cu_device()) }.result()?;
        let mut used = 0u64;
        let mut reserved = 0u64;
        unsafe {
            sys::cuMemPoolGetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH, (&mut used as *mut u64).cast()).result()?;
            sys::cuMemPoolGetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, (&mut reserved as *mut u64).cast()).result()?;
        }
        println!("VRAM pool_used_peak_mib={:.3} pool_reserved_peak_mib={:.3}", used as f64 / 1048576., reserved as f64 / 1048576.);
    }
    let _ = device;
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_ansi(false).with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.out)?;
    let picture = read(&args.picture)?;
    anyhow::ensure!((picture.width, picture.height) == (864, 1152), "expected the speed-table portrait");
    let mut reference = read(&args.reference)?;
    // PhotoshopEX caps uploaded references at one megapixel before the worker sees them.
    let scale = (1_048_576.0 / (reference.width * reference.height) as f64).sqrt().min(1.0);
    let (rw, rh) = ((reference.width as f64 * scale).round() as u32, (reference.height as f64 * scale).round() as u32);
    let img = image::RgbaImage::from_raw(reference.width as u32, reference.height as u32, reference.rgba).context("reference bytes")?;
    reference = RefImage {
        width: rw as usize,
        height: rh as usize,
        rgba: image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3).into_raw(),
        offset: None,
    };
    let opts = Options {
        cache: qwen_image_rs::convrot_cache::CacheSpec::from_cli(Some(&args.cache), false, false),
        text_gguf: args.text_gguf,
        evict_text_encoder: !args.keep_text,
        ..Options::default()
    };
    let p = Pipeline::load(&args.models, opts, &mut |_, m| tracing::info!(m, "load"))?;
    let loras = match args.lora {
        Some(path) => qwen_image_rs::lora::LoraStack {
            layers: vec![(std::sync::Arc::new(qwen_image_rs::lora::LoraFile::load(&path, p.device(), candle_core::DType::BF16)?), args.lora_strength)],
        },
        None => qwen_image_rs::lora::LoraStack::default(),
    };
    let tiles = upscale_tiles(&picture)?;
    let reset_reference = if args.check_lora_reset {
        anyhow::ensure!(!loras.is_empty(), "--check-lora-reset requires --lora");
        let mut request = Request::new(RESTORE_PROMPT);
        request.width = 0;
        request.height = 0;
        request.seed = 777;
        request.references = vec![tiles[0].clone()];
        let before = p.generate(&request, &mut |_, _| {}, &|| false)?;
        std::fs::write(args.out.join("reset-before.rgba"), &before.rgba)?;
        Some((request, before.rgba))
    } else {
        None
    };
    for route in args.routes.split(',') {
        let mut req = Request::new(match route {
            "image" => "a red fox sitting in fresh snow at dawn, 35mm photo",
            "image-ref" => "a portrait of a woman wearing this hat",
            "fill" => "Edit image 1. Image 2 is a mask of image 1: change only the area that is white in image 2 so that it shows a denim jacket. Keep everything outside that area exactly as it is in image 1, with the same lighting, perspective and style.",
            "edit" | "tiles" => "Make it winter: snow falling and a wool scarf around his neck.",
            "matte" => "Remove the background, keeping only the man, and output a PNG image",
            "upscale" => RESTORE_PROMPT,
            _ => anyhow::bail!("unknown route {route}"),
        });
        req.width = 0;
        req.height = 0;
        req.loras = loras.clone();
        req.references = vec![picture.clone()];
        if route == "image" {
            req.width = 864;
            req.height = 1152;
            req.references.clear();
        } else if route == "image-ref" {
            req.width = 864;
            req.height = 1152;
            req.references = vec![reference.clone()];
        } else if route == "fill" {
            let img = image::RgbaImage::from_raw(864, 1152, picture.rgba.clone()).context("portrait bytes")?;
            let crop = image::imageops::crop_imm(&img, 48, 384, 768, 672).to_image();
            let mask = image::RgbaImage::from_fn(768, 672, |x, y| {
                let v = if (132..632).contains(&x) && (136..556).contains(&y) { 255 } else { 0 };
                image::Rgba([v, v, v, 255])
            });
            req.references = vec![
                RefImage { width: 768, height: 672, rgba: crop.into_raw(), offset: None },
                RefImage { width: 768, height: 672, rgba: mask.into_raw(), offset: None },
            ];
        }
        let repeats = if route == "upscale" {
            args.repeats * 6
        } else if route == "tiles" && args.tile_repeats > 0 {
            args.tile_repeats
        } else {
            args.repeats
        };
        for i in 0..repeats {
            req.seed = 42 + i as u64;
            if route == "upscale" {
                req.references = vec![tiles[i % 6].clone()];
                image::save_buffer(args.out.join(format!("input-{}.png", i % 6)), &req.references[0].rgba, 992, 992, image::ColorType::Rgba8)?;
            }
            // A different picture each tile forces conditioning misses.
            if route == "tiles" {
                req.references[0].rgba[i % picture.rgba.len()] ^= 1;
            }
            let t = Instant::now();
            let pic = p.generate(&req, &mut |_, _| {}, &|| false)?;
            let elapsed = t.elapsed().as_secs_f64();
            let raw = args.out.join(format!("{route}-{i}.rgba"));
            std::fs::write(raw, &pic.rgba)?;
            let t = Instant::now();
            image::save_buffer(args.out.join(format!("{route}-{i}.png")), &pic.rgba, pic.width, pic.height, image::ColorType::Rgba8)?;
            println!(
                "BENCH route={route} run={i} seed={} seconds={elapsed:.6} encode={:.6} denoise={:.6} decode={:.6} png={:.6} te_resident={}",
                req.seed,
                pic.timings.encode_s,
                pic.timings.denoise_s,
                pic.timings.decode_s,
                t.elapsed().as_secs_f64(),
                p.text_encoder_resident()
            );
            qwen_image_rs::profile::report(route);
            qwen_image_rs::device::trim_pool(p.device());
        }
    }
    if let Some((request, before)) = reset_reference {
        let after = p.generate(&request, &mut |_, _| {}, &|| false)?;
        std::fs::write(args.out.join("reset-after.rgba"), &after.rgba)?;
        anyhow::ensure!(before == after.rgba, "clearing LoRAs changed the original image");
        println!("RESET byte_identical=true");
    }
    memory_report(p.device())?;
    Ok(())
}
