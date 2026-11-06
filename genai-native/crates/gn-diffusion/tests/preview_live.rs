//! Live previews on the real models (ignored: needs the CUDA build, a card and the weights).
//! For each case: a warm-up run, then Off, Fast, Sharp and Off again at 864×1152, each timed
//! with the card's peak memory (nvidia-smi, every 100 ms); the pictures must be byte-identical
//! across the four runs. One table row per run on stdout; a few previews and the result as PNG
//! under `GN_PREVIEW_OUT` when set.
//!
//! ```text
//! GN_MODELS=<ComfyUI models folder> GN_PREVIEW_OUT=<folder> \
//!   cargo test -p gn-diffusion --features cuda --release --test preview_live -- --ignored --nocapture
//! ```
//! `GN_PREVIEW_CASES=krea2,qwen21` runs only those (`qwen21 krea2 edit2511 layered upscale`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use gn_diffusion::DiffusionEngine;
use gn_engine::protocol::{Blob, PreviewMode, RunRequest};
use gn_engine::server::Engine;
use serde_json::json;

const W: u32 = 864;
const H: u32 = 1152;

fn request(model: &str, task: &str, prompt: &str) -> RunRequest {
    RunRequest {
        model: model.into(),
        task: task.into(),
        prompt: prompt.into(),
        negative: String::new(),
        seed: 7,
        steps: 0,
        guidance: 0.0,
        width: W,
        height: H,
        image: None,
        mask: None,
        references: Vec::new(),
        params: BTreeMap::new(),
        preview: PreviewMode::Off,
    }
}

/// The card's memory in use (MiB), sampled every 100 ms while it runs; the peak on `stop`.
struct PeakVram {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
}

impl PeakVram {
    fn start() -> Self {
        let (peak, stop) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicBool::new(false)));
        let (p, s) = (Arc::clone(&peak), Arc::clone(&stop));
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                if let Some(m) = used_mib() {
                    p.fetch_max(m, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Self { peak, stop, thread: Some(thread) }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.peak.load(Ordering::SeqCst)
    }
}

fn fnv(bytes: &[Vec<u8>]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes.iter().flatten() {
        h = (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn save(out: &Option<PathBuf>, name: &str, w: u32, h: u32, rgba: &[u8]) {
    if let Some(dir) = out
        && let Some(img) = image::RgbaImage::from_raw(w, h, rgba.to_vec())
    {
        let _ = img.save(dir.join(format!("{name}.png")));
    }
}

struct Case {
    name: &'static str,
    req: RunRequest,
    blobs: Vec<Vec<u8>>,
}

/// Run `case` four times (Off, Fast, Sharp, Off) after a warm-up; print the rows; check the
/// pictures agree. Returns the first result's pictures.
fn bench(engine: &mut DiffusionEngine, case: &Case, out: &Option<PathBuf>) -> Vec<Vec<u8>> {
    let cancel = AtomicBool::new(false);
    let t0 = Instant::now();
    engine.run(&case.req, &case.blobs, &mut |_, _| {}, &cancel).unwrap_or_else(|e| panic!("{} warm-up: {e:?}", case.name));
    println!("| {} | warm-up | {:.2} s | | | |", case.name, t0.elapsed().as_secs_f64());
    let mut first: Option<(u64, Vec<Vec<u8>>)> = None;
    for mode in [PreviewMode::Off, PreviewMode::Fast, PreviewMode::Sharp, PreviewMode::Off] {
        let mut req = case.req.clone();
        req.preview = mode;
        let mut previews = Vec::new();
        let mut sizes = std::collections::BTreeSet::new();
        let vram = PeakVram::start();
        let t0 = Instant::now();
        let (info, pics) = engine
            .run_previewed(
                &req,
                &case.blobs,
                &mut |_, _| {},
                &mut |p| {
                    sizes.insert((p.width, p.height));
                    previews.push(p);
                },
                &cancel,
            )
            .unwrap_or_else(|e| panic!("{} {mode:?}: {e:?}", case.name));
        let secs = t0.elapsed().as_secs_f64();
        let peak = vram.finish();
        let hash = fnv(&pics);
        let same = first.as_ref().is_none_or(|(h, _)| *h == hash);
        println!(
            "| {} | {} | {secs:.2} s | {peak} MiB | {} previews {:?} | {:016x}{} |",
            case.name,
            mode.as_str(),
            previews.len(),
            sizes,
            hash,
            if same { "" } else { " DIFFERS" }
        );
        if mode != PreviewMode::Off {
            for p in previews.iter().filter(|p| p.output == 0).step_by(previews.len().div_ceil(4).max(1)) {
                save(out, &format!("{}-{}-step{:02}", case.name, mode.as_str(), p.step), p.width, p.height, &p.rgba);
            }
        }
        match &first {
            None => {
                if let Some(b) = info.images.first() {
                    save(out, &format!("{}-result", case.name), b.width, b.height, &pics[0]);
                }
                first = Some((hash, pics));
            }
            Some((h, _)) => assert_eq!(*h, hash, "{}: the picture with previews {mode:?} differs from Off's", case.name),
        }
    }
    first.map(|(_, p)| p).unwrap_or_default()
}

fn wanted(name: &str) -> bool {
    std::env::var("GN_PREVIEW_CASES").map_or(true, |v| v.split(',').any(|c| c.trim() == name))
}

#[test]
#[ignore = "needs the CUDA build, a card and the weights (GN_MODELS)"]
fn previews_leave_every_picture_alone() {
    let Some(models) = std::env::var_os("GN_MODELS").map(PathBuf::from) else { panic!("set GN_MODELS to the models root") };
    let out = std::env::var_os("GN_PREVIEW_OUT").map(PathBuf::from);
    if let Some(d) = &out {
        std::fs::create_dir_all(d).unwrap();
    }
    let mut engine = DiffusionEngine::new(models, false);
    println!("| case | mode | warm time | peak card memory | previews | result |");
    println!("|---|---|---|---|---|---|");
    // Krea 2 first: its picture is the input of the edits and the split.
    let mut krea = Case { name: "krea2", req: request("krea2-turbo/image", "image", "a red fox in a snowy birch forest at dawn, photo"), blobs: Vec::new() };
    krea.req.steps = 8;
    let picture = bench(&mut engine, &krea, &out).into_iter().next().unwrap();
    if wanted("qwen21") {
        let mut c = Case { name: "qwen21", req: request("qwen-2.1/image", "image", "a red fox in a snowy birch forest at dawn, photo"), blobs: Vec::new() };
        c.req.steps = 25;
        bench(&mut engine, &c, &out);
    }
    if wanted("edit2511") {
        // A centred rectangle, a third of each side, filled.
        let mask: Vec<u8> =
            (0..H).flat_map(|y| (0..W).map(move |x| if (W / 3..2 * W / 3).contains(&x) && (H / 3..2 * H / 3).contains(&y) { 255 } else { 0 })).collect();
        let mut c =
            Case { name: "edit2511-fill-8", req: request("qwen-edit-2511/fill-lightning-8", "fill", "a wooden bench"), blobs: vec![picture.clone(), mask] };
        c.req.image = Some(Blob { width: W, height: H, channels: 4, index: 0 });
        c.req.mask = Some(Blob { width: W, height: H, channels: 1, index: 1 });
        c.req.width = 0;
        c.req.height = 0;
        bench(&mut engine, &c, &out);
    }
    if wanted("layered") {
        let mut c = Case { name: "layered-3", req: request("qwen-layered/split", "layers", "a red fox in a snowy forest"), blobs: vec![picture.clone()] };
        c.req.image = Some(Blob { width: W, height: H, channels: 4, index: 0 });
        c.req.params.insert("layers".into(), json!(3));
        c.req.steps = 20;
        bench(&mut engine, &c, &out);
    }
    if wanted("upscale") {
        // One Upscale tile: the 2.1 edit of a ~1 MP picture with the Restore LoRA, as
        // Generative Upscale sends each tile.
        let mut c = Case {
            name: "upscale-tile",
            req: request("qwen-2.1/edit", "edit", "Restore this picture: sharpen details and remove blur, keep everything else unchanged."),
            blobs: vec![picture],
        };
        c.req.image = Some(Blob { width: W, height: H, channels: 4, index: 0 });
        c.req.width = 0;
        c.req.height = 0;
        c.req.steps = 25;
        c.req.params.insert("loras".into(), json!([{"file": "qwen-image-2.1-restore.safetensors", "strength": 0.5}]));
        bench(&mut engine, &c, &out);
    }
    engine.free();
}
