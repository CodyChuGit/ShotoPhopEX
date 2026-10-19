//! `gn-sam`: inspect and run the SAM 3 ONNX graphs from the command line.
//!
//! ```text
//! gn-sam inspect --ort <dir> <model.onnx>...
//! gn-sam run --ort <dir> [--lib <dir>]... [--cpu] --models <dir> --image <png>
//!            (--text "fox" | --point x,y | --box x0,y0,x1,y1)... [--negative x,y]...
//!            [--threshold 0.5] [--max 1] [--refine 2] [--presence] [--no-normalize] [--out mask.png] [--id name]
//! ```
//!
//! The model directory's layout picks the path: ComfyUI's export (`vision.onnx` … `heads.onnx`,
//! the node's own text, point and box routes) or the Hugging Face export (`vision-encoder.onnx`
//! …, detector only).
//!
//! `run` prints timings to stderr and one JSON line to stdout in the shape PhotoshopEX's
//! `select.*` commands answer with (`selected`, `count`, `instances[{index, bounds, pixels}]`,
//! `ms`), so the harness can compare it with the ComfyUI references.

use std::path::{Path, PathBuf};
use std::time::Instant;

use gn_perception::graph::Instance;
use gn_perception::runtime::{Device, Runtime, RuntimeConfig, ports};
use gn_perception::sam3::{Prompt, Sam3, Scoring};
use gn_perception::sam31::{Options, Sam31};
use serde_json::json;

struct Args {
    positional: Vec<String>,
    named: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut positional = Vec::new();
        let mut named = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            if let Some(name) = raw[i].strip_prefix("--") {
                let takes_value = !matches!(name, "cpu" | "presence" | "no-normalize");
                let value = if takes_value { raw.get(i + 1).cloned().unwrap_or_default() } else { String::new() };
                named.push((name.to_string(), value));
                i += if takes_value { 2 } else { 1 };
            } else {
                positional.push(raw[i].clone());
                i += 1;
            }
        }
        Args { positional, named }
    }
    fn one(&self, key: &str) -> Option<&str> {
        self.named.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).filter(|v| !v.is_empty())
    }
    fn all(&self, key: &str) -> Vec<&str> {
        self.named.iter().filter(|(k, _)| k == key).map(|(_, v)| v.as_str()).filter(|v| !v.is_empty()).collect()
    }
    fn flag(&self, key: &str) -> bool {
        self.named.iter().any(|(k, _)| k == key)
    }
    fn need(&self, key: &str) -> Result<&str, String> {
        self.one(key).ok_or_else(|| format!("--{key} is required"))
    }
}

fn numbers(s: &str, n: usize) -> Result<Vec<f32>, String> {
    let v: Result<Vec<f32>, _> = s.split(',').map(|p| p.trim().parse::<f32>()).collect();
    match v {
        Ok(v) if v.len() == n => Ok(v),
        _ => Err(format!("`{s}` is not {n} comma-separated numbers")),
    }
}

/// Read a PNG as RGBA8.
fn read_png(path: &Path) -> Result<(Vec<u8>, usize, usize), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let px = &buf[..info.buffer_size()];
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|g| [*g, *g, *g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        other => return Err(format!("{}: unsupported colour type {other:?}", path.display())),
    };
    Ok((rgba, w, h))
}

/// Write a coverage plane as an 8-bit grey PNG.
fn write_mask(path: &Path, mask: &[f32], w: usize, h: usize) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = mask.iter().map(|v| if *v > 0.5 { 255 } else { 0 }).collect();
    writer.write_image_data(&bytes).map_err(|e| e.to_string())
}

fn runtime(args: &Args) -> Result<Runtime, String> {
    let device = if args.flag("cpu") { Device::Cpu } else { Device::Cuda { ordinal: 0 } };
    let profile_dir = args.one("profile").map(PathBuf::from);
    let cfg = RuntimeConfig { ort_dir: PathBuf::from(args.need("ort")?), library_dirs: args.all("lib").into_iter().map(PathBuf::from).collect(), device, profile_dir };
    Runtime::open(cfg).map_err(|e| e.to_string())
}

fn inspect(args: &Args) -> Result<(), String> {
    let rt = runtime(args)?;
    for model in args.positional.iter().skip(1) {
        let t0 = Instant::now();
        let session = rt.cpu_session(Path::new(model)).map_err(|e| e.to_string())?;
        let (ins, outs) = ports(&session);
        println!("{model} (opened in {:.1} s)", t0.elapsed().as_secs_f64());
        for p in ins {
            println!("  in  {}: {}", p.name, p.dtype);
        }
        for p in outs {
            println!("  out {}: {}", p.name, p.dtype);
        }
    }
    Ok(())
}

/// What the command line asked for, in picture pixels.
fn prompt_of(args: &Args) -> Result<Prompt, String> {
    let mut prompt = Prompt { text: args.one("text").unwrap_or("").to_string(), boxes: vec![], points: vec![], negatives: vec![] };
    for p in args.all("point") {
        let v = numbers(p, 2)?;
        prompt.points.push([v[0], v[1]]);
    }
    for p in args.all("negative") {
        let v = numbers(p, 2)?;
        prompt.negatives.push([v[0], v[1]]);
    }
    for b in args.all("box") {
        let v = numbers(b, 4)?;
        prompt.boxes.push([v[0], v[1], v[2], v[3]]);
    }
    if prompt.text.is_empty() && prompt.boxes.is_empty() && prompt.points.is_empty() {
        return Err("give --text, --point or --box".into());
    }
    Ok(prompt)
}

/// The ComfyUI layout (`vision.onnx` … `heads.onnx`): the node's own routes.
fn run_comfy(rt: &Runtime, models: &Path, args: &Args, rgba: &[u8], w: usize, h: usize, prompt: &Prompt) -> Result<(Vec<Instance>, u64, u64), String> {
    let t0 = Instant::now();
    let mut sam = Sam31::open(rt, models).map_err(|e| e.to_string())?;
    eprintln!("gn-sam: sessions open in {:.1} s ({:?}, ComfyUI layout)", t0.elapsed().as_secs_f64(), rt.config().device);
    let opts = Options {
        threshold: args.one("threshold").and_then(|t| t.parse().ok()).unwrap_or(0.5),
        max_detections: args.one("max").and_then(|t| t.parse().ok()).unwrap_or(1),
        refine_iterations: args.one("refine").and_then(|t| t.parse().ok()).unwrap_or(2),
    };
    // `--repeat N` runs the whole thing N times: the last run's timings are the warm ones.
    let repeat = args.one("repeat").and_then(|t| t.parse::<usize>().ok()).unwrap_or(1).max(1);
    let mut last: Option<(Vec<Instance>, u64, u64)> = None;
    for round in 0..repeat {
        let emb = sam.embed(rgba, w, h).map_err(|e| e.to_string())?;
        eprintln!("gn-sam: [{}] vision {} ms for {w}×{h} (squash {} + run {} + copy out {})", round + 1, emb.ms, emb.split_ms[0], emb.split_ms[1], emb.split_ms[2]);
        let t1 = Instant::now();
        let mut instances = Vec::new();
        // The node's order: points, then boxes (without text), then text detections.
        if !prompt.points.is_empty() || !prompt.negatives.is_empty() {
            let (inst, t) = sam.segment_points(&emb, &prompt.points, &prompt.negatives, opts).map_err(|e| e.to_string())?;
            eprintln!("gn-sam: [{}] points: heads {} ms, post {} ms", round + 1, t.heads_ms, t.post_ms);
            instances.push(inst);
        }
        if prompt.text.is_empty() {
            for b in &prompt.boxes {
                let (inst, t) = sam.segment_box(&emb, *b, opts).map_err(|e| e.to_string())?;
                eprintln!("gn-sam: [{}] box: heads {} ms, post {} ms", round + 1, t.heads_ms, t.post_ms);
                instances.push(inst);
            }
        } else {
            let (found, t) = sam.detect_text(&emb, &prompt.text, &prompt.boxes, opts).map_err(|e| e.to_string())?;
            eprintln!("gn-sam: [{}] text: text {} + decoder {} + crops {} + heads {} + post {} ms; {} detection(s)", round + 1, t.text_ms, t.decoder_ms, t.crop_ms, t.heads_ms, t.post_ms, found.len());
            instances.extend(found);
        }
        last = Some((instances, emb.ms, t1.elapsed().as_millis() as u64));
    }
    last.ok_or_else(|| "no run".to_string())
}

/// The Hugging Face layout (`vision-encoder.onnx` …): detector only.
fn run_hf(rt: &Runtime, models: &Path, args: &Args, rgba: &[u8], w: usize, h: usize, prompt: &Prompt) -> Result<(Vec<Instance>, u64, u64), String> {
    let t0 = Instant::now();
    let mut sam = Sam3::open(rt, models).map_err(|e| e.to_string())?;
    if args.flag("no-normalize") {
        sam.normalize = None;
    }
    eprintln!("gn-sam: sessions open in {:.1} s ({:?}, Hugging Face layout)", t0.elapsed().as_secs_f64(), rt.config().device);
    let emb = sam.embed(rgba, w, h).map_err(|e| e.to_string())?;
    eprintln!("gn-sam: vision encoder {} ms for {w}×{h}", emb.ms);
    let scoring = Scoring {
        threshold: args.one("threshold").and_then(|t| t.parse().ok()).unwrap_or(0.5),
        max_instances: args.one("max").and_then(|t| t.parse().ok()).unwrap_or(1),
        presence: args.flag("presence"),
    };
    let t1 = Instant::now();
    let (instances, timing) = sam.segment(&emb, prompt, scoring).map_err(|e| e.to_string())?;
    let prompt_ms = t1.elapsed().as_millis() as u64;
    eprintln!("gn-sam: prompt {prompt_ms} ms (text {} + geometry {} + decoder {} + post {}); {} instance(s)", timing.text_ms, timing.geometry_ms, timing.decoder_ms, timing.post_ms, instances.len());
    Ok((instances, emb.ms, prompt_ms))
}

fn run(args: &Args) -> Result<(), String> {
    let t_all = Instant::now();
    let rt = runtime(args)?;
    let models = PathBuf::from(args.need("models")?);
    let image = PathBuf::from(args.need("image")?);
    let (rgba, w, h) = read_png(&image)?;
    let prompt = prompt_of(args)?;
    let (instances, embed_ms, prompt_ms) = if Sam31::is_layout(&models) {
        run_comfy(&rt, &models, args, &rgba, w, h, &prompt)?
    } else {
        run_hf(&rt, &models, args, &rgba, w, h, &prompt)?
    };
    for (i, inst) in instances.iter().enumerate() {
        eprintln!("  {}: score {:.3} bounds {:?} pixels {} bbox {:?}", i + 1, inst.score, inst.bounds, inst.pixels, inst.bbox.map(|v| v.round()));
    }

    if let Some(out) = args.one("out") {
        // The union of the instances, as PhotoshopEX selects by default.
        let mut union = vec![0.0f32; w * h];
        for inst in &instances {
            for (u, m) in union.iter_mut().zip(inst.mask.iter()) {
                *u = u.max(*m);
            }
        }
        write_mask(Path::new(out), &union, w, h)?;
    }

    let list: Vec<serde_json::Value> = instances.iter().enumerate().map(|(i, inst)| json!({"index": i + 1, "bounds": inst.bounds, "pixels": inst.pixels, "score": inst.score})).collect();
    let result = json!({
        "id": args.one("id").unwrap_or("sam"),
        "result": {
            "selected": !instances.is_empty(),
            "count": instances.len(),
            "instances": list,
            "ms": embed_ms + prompt_ms,
            "embedMs": embed_ms,
            "promptMs": prompt_ms,
            "totalMs": t_all.elapsed().as_millis() as u64,
        }
    });
    println!("{result}");
    Ok(())
}

fn main() {
    let args = Args::parse();
    let r = match args.positional.first().map(String::as_str) {
        Some("inspect") => inspect(&args),
        Some("run") => run(&args),
        _ => Err("usage: gn-sam inspect --ort <dir> <model.onnx>...\n       gn-sam run --ort <dir> [--lib <dir>]... [--cpu] --models <dir> --image <png> (--text t | --point x,y | --box x0,y0,x1,y1) [--negative x,y] [--threshold 0.5] [--max 1] [--out mask.png]".into()),
    };
    if let Err(e) = r {
        eprintln!("gn-sam: {e}");
        std::process::exit(1);
    }
}
