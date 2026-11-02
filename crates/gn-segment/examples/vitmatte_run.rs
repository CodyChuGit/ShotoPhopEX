//! Run ViTMatte on a picture and a trimap: `vitmatte_run <model dir> <picture> <trimap> <out.png> [cpu]`.
//! Prints the time; writes the alpha as a grey PNG (for comparing against the reference).

use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use gn_segment::vitmatte::VitMatte;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, picture, trimap, out, rest @ ..] = args.as_slice() else {
        anyhow::bail!("usage: vitmatte_run <model dir> <picture> <trimap> <out.png> [cpu]");
    };
    let device = if rest.first().is_some_and(|a| a == "cpu") { Device::Cpu } else { Device::cuda_if_available(0)? };
    let img = image::open(picture).context("picture")?.to_rgba8();
    let tri = image::open(trimap).context("trimap")?.to_luma8();
    let (w, h) = img.dimensions();
    anyhow::ensure!(tri.dimensions() == (w, h), "trimap size differs");
    let t = Instant::now();
    let model = VitMatte::load(dir, &device)?;
    println!("load {} ms", t.elapsed().as_millis());
    for run in 0..3 {
        let t = Instant::now();
        let alpha = model.matte(img.as_raw(), tri.as_raw(), w as usize, h as usize)?;
        println!("run {run}: {} ms", t.elapsed().as_millis());
        if run == 2 {
            let px: Vec<u8> = alpha.iter().map(|a| (a.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
            image::GrayImage::from_raw(w, h, px).context("alpha")?.save(out)?;
        }
    }
    Ok(())
}
