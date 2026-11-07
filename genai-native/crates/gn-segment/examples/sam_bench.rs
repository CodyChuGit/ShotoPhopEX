//! Time SAM 3.1's stages on one picture: load, embed, a text prompt twice (cold and warm), a click.
//! `cargo run --release --features metal --example sam_bench -- <checkpoint> <tokenizer.json> <picture> [f16]`
use anyhow::Result;
use candle_core::{DType, Device};
use gn_segment::{
    profile,
    sam::{Options, Sam31},
    weights::Checkpoint,
};
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dtype = if a.get(4).is_some_and(|s| s == "f16") { DType::F16 } else { DType::F32 };
    let device = if std::env::var("CPU").is_ok() { Device::Cpu } else { Device::new_metal(0).or_else(|_| Device::new_cuda(0))? };
    let t = Instant::now();
    let ck = Checkpoint::open(&a[1])?.with_compute_dtype(dtype)?;
    let mut sam = Sam31::from_checkpoint(&ck, &device)?;
    sam.set_tokenizer(&a[2])?;
    println!("load {dtype:?}: {:.1} s", t.elapsed().as_secs_f32());
    let img = image::open(&a[3])?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let t = Instant::now();
    let frame = sam.encode(img.as_raw(), w, h)?;
    device.synchronize()?;
    println!("embed: {:.1} s", t.elapsed().as_secs_f32());
    for prompt in ["the rubber duck", "the mug", "the apple"] {
        let mut runs = Vec::new();
        for refine in [2usize, 1, 0] {
            let t = Instant::now();
            let (found, _) = sam.detect_text(&frame, prompt, &[], Options { refine_iterations: refine, ..Options::default() })?;
            let secs = t.elapsed().as_secs_f32();
            let mask = found.first().map(|f| f.mask.clone()).unwrap_or_default();
            let iou = runs.first().map_or(1.0, |m: &Vec<f32>| {
                let (mut i, mut u) = (0usize, 0usize);
                for (a, b) in m.iter().zip(&mask) {
                    let (a, b) = (*a > 0.5, *b > 0.5);
                    i += usize::from(a && b);
                    u += usize::from(a || b);
                }
                i as f32 / u.max(1) as f32
            });
            println!("{prompt:?} refine {refine}: {secs:.1} s, IoU vs refine 2 {iou:.4}");
            runs.push(mask);
        }
    }
    let t = Instant::now();
    let (inst, _) = sam.segment_points(&frame, &[[250.0, 330.0]], &[], Options::default())?;
    println!("click: {:.2} s bounds {:?}", t.elapsed().as_secs_f32(), inst.bounds);
    Ok(())
}
