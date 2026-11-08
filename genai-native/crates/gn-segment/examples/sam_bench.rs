//! Time SAM 3.1's stages on one picture: load, embed, a text prompt with 2, 1 and 0 refinement
//! passes (each mask's IoU against two passes), a click.
//! `cargo run --release --features metal --example sam_bench -- <checkpoint> <tokenizer.json> <picture> [f16]`
//! With `SAM_COMPARE=1` it instead compares f16 against f32 masks (one model held at a time).
use anyhow::Result;
use candle_core::{DType, Device};
use gn_segment::{
    sam::{Options, Sam31},
    weights::Checkpoint,
};
use std::time::Instant;

const PROMPTS: [&str; 3] = ["the rubber duck", "the mug", "the apple"];
/// On the 512² duck picture: the duck, the mug, the apple, the table.
const CLICKS: [[f32; 2]; 4] = [[200.0, 371.0], [360.0, 230.0], [117.0, 258.0], [256.0, 480.0]];
/// Around the duck.
const BOX: [f32; 4] = [118.0, 287.0, 282.0, 455.0];

fn iou(a: &[f32], b: &[f32]) -> (f32, usize) {
    let (mut inter, mut uni) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        inter += usize::from(*x > 0.5 && *y > 0.5);
        uni += usize::from(*x > 0.5 || *y > 0.5);
    }
    (inter as f32 / uni.max(1) as f32, uni)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dtype = if a.get(4).is_some_and(|s| s == "f16") { DType::F16 } else { DType::F32 };
    let device = if std::env::var("CPU").is_ok() { Device::Cpu } else { Device::new_metal(0).or_else(|_| Device::new_cuda(0))? };
    let img = image::open(&a[3])?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    if std::env::var("SAM_COMPARE").is_ok() {
        let mut masks: Vec<Vec<Vec<f32>>> = Vec::new();
        for dt in [DType::F32, DType::F16] {
            let ck = Checkpoint::open(&a[1])?.with_compute_dtype(dt)?;
            let mut sam = Sam31::from_checkpoint(&ck, &device)?;
            drop(ck);
            sam.set_tokenizer(&a[2])?;
            let frame = sam.encode(img.as_raw(), w, h)?;
            let mut m = Vec::new();
            for prompt in PROMPTS {
                for refine in [0usize, 2] {
                    let (found, _) = sam.detect_text(&frame, prompt, &[], Options { refine_iterations: refine, ..Options::default() })?;
                    m.push(found.first().map(|f| f.mask.clone()).unwrap_or_default());
                }
            }
            // Clicks (Object Selection) and a box, where the masks come from the interactive heads.
            for pt in CLICKS {
                let (inst, _) = sam.segment_points(&frame, &[pt], &[], Options::default())?;
                println!("{dt:?} click {pt:?}: bounds {:?} px {}", inst.bounds, inst.pixels);
                m.push(inst.mask);
            }
            let (inst, _) = sam.segment_box(&frame, BOX, Options::default())?;
            println!("{dt:?} box {BOX:?}: bounds {:?} px {}", inst.bounds, inst.pixels);
            m.push(inst.mask);
            masks.push(m);
            // The model and its embedding go before the next dtype loads.
        }
        let names: Vec<String> = PROMPTS
            .iter()
            .flat_map(|p| [format!("{p:?} refine 0"), format!("{p:?} refine 2")])
            .chain(CLICKS.iter().map(|c| format!("click {c:?}")))
            .chain(std::iter::once(format!("box {BOX:?}")))
            .collect();
        for (i, name) in names.iter().enumerate() {
            let (v, px) = iou(&masks[0][i], &masks[1][i]);
            println!("{name}: f16 vs f32 IoU {v:.4} ({px} px)");
        }
        return Ok(());
    }

    let t = Instant::now();
    let ck = Checkpoint::open(&a[1])?.with_compute_dtype(dtype)?;
    let mut sam = Sam31::from_checkpoint(&ck, &device)?;
    sam.set_tokenizer(&a[2])?;
    println!("load {dtype:?}: {:.1} s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let frame = sam.encode(img.as_raw(), w, h)?;
    device.synchronize()?;
    println!("embed: {:.1} s", t.elapsed().as_secs_f32());
    for prompt in PROMPTS {
        let mut first: Option<Vec<f32>> = None;
        for refine in [2usize, 1, 0] {
            let t = Instant::now();
            if std::env::var_os("GN_SAM_PROFILE").is_some() {
                gn_segment::profile::start();
            }
            let (found, _) = sam.detect_text(&frame, prompt, &[], Options { refine_iterations: refine, ..Options::default() })?;
            let secs = t.elapsed().as_secs_f32();
            if std::env::var_os("GN_SAM_PROFILE").is_some() {
                println!("stages (ms) {:?}", gn_segment::profile::finish());
            }
            let mask = found.first().map(|f| f.mask.clone()).unwrap_or_default();
            let v = first.as_ref().map_or(1.0, |m| iou(m, &mask).0);
            println!("{prompt:?} refine {refine}: {secs:.1} s, IoU vs refine 2 {v:.4}");
            first.get_or_insert(mask);
        }
    }
    let t = Instant::now();
    let (inst, _) = sam.segment_points(&frame, &[[250.0, 330.0]], &[], Options::default())?;
    println!("click: {:.2} s bounds {:?}", t.elapsed().as_secs_f32(), inst.bounds);
    Ok(())
}
