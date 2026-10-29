//! `generate.upscale`: Generative Upscale (AusBoss's tiled restore recipe, catalogue §2.5). The
//! document is enlarged the plain way first (`image.imageSize`, bicubic), then redrawn tile by
//! tile by the edit model with the restore prompt: tiles of about a megapixel sharing a
//! [`OVERLAP`]-pixel strip, each moved back onto its source (`generate_realign`) and matched
//! to the source's colours (mean and spread per channel) so grain does not push saturation, the
//! strips faded into each other. The redraw lands as a layer above the enlarged picture.

use std::collections::BTreeMap;

use photocraft_doc::Layer;
use photocraft_genai::{Request, Rgba8, Task};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use super::loras::runs_native;
use super::{
    DEFAULT_EDIT_TEMPLATE, GenerativeInfo, JobProgress, backend_for, gen_err, loras_of, opt_num, opt_str, plan_common, request_size_within, resize_rgba8,
    set_generative_info, write_rgba8,
};
use crate::jobs::now_ms;
use crate::{EngineError, Result, Session};

pub const UPSCALE: &str = "generate.upscale";

/// AusBoss's restore wording (Tiled Upscale, Civitai, read 2026-10-26).
pub(crate) const RESTORE_PROMPT: &str =
    "Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, keeping the same composition, people and colors.";
/// The Restore LoRA (AusBoss, Qwen-Image-2.1): taken when the native engine runs the 2.1 edit
/// and the file is in its LoRA folder, at [`RESTORE_STRENGTH`] unless `loras` says otherwise.
const RESTORE_LORA: &str = "qwen-image-2.1-restore.safetensors";
const RESTORE_STRENGTH: f32 = 0.5;
/// The strip neighbouring tiles share and fade across.
pub(crate) const OVERLAP: u32 = 128;
/// The largest enlarged picture (a tile run per megapixel).
const MAX_PIXELS: u64 = 64_000_000;

/// The tiles of a `w`×`h` picture: sides of about `side` pixels (whole picture when smaller),
/// stepping by `side − OVERLAP`, the last tile of a row or column flush with the edge.
pub(crate) fn tiles(w: u32, h: u32, side: u32) -> Vec<Rect> {
    let starts = |len: u32| -> Vec<u32> {
        if len <= side {
            return vec![0];
        }
        let step = side.saturating_sub(OVERLAP).max(1);
        let mut v: Vec<u32> = (0..).map(|i| i * step).take_while(|x| x + side < len).collect();
        v.push(len - side);
        v
    };
    let (tw, th) = (side.min(w), side.min(h));
    let mut out = Vec::new();
    for y in starts(h) {
        for x in starts(w) {
            out.push(Rect::new(x as i32, y as i32, (x + tw) as i32, (y + th) as i32));
        }
    }
    out
}

/// The tiles of a `w`×`h` picture in `cols`×`rows`: equal tiles sharing at least [`OVERLAP`]
/// pixels with each neighbour, sides rounded up to multiples of 16, spread evenly from edge to
/// edge.
pub(crate) fn grid_tiles(w: u32, h: u32, cols: u32, rows: u32) -> Vec<Rect> {
    let side = |len: u32, n: u32| -> u32 {
        if n <= 1 {
            return len;
        }
        (len + (n - 1) * OVERLAP).div_ceil(n).div_ceil(16).saturating_mul(16).min(len)
    };
    let starts = |len: u32, t: u32, n: u32| -> Vec<u32> {
        if n <= 1 {
            return vec![0];
        }
        (0..n).map(|i| (u64::from(i) * u64::from(len - t) / u64::from(n - 1)) as u32).collect()
    };
    let (tw, th) = (side(w, cols.max(1)), side(h, rows.max(1)));
    let mut out = Vec::new();
    for y in starts(h, th, rows.max(1)) {
        for x in starts(w, tw, cols.max(1)) {
            out.push(Rect::new(x as i32, y as i32, (x + tw) as i32, (y + th) as i32));
        }
    }
    out
}

/// The largest tile the automatic grid makes: about the 2.1 edit's own megapixel with the strips.
const MAX_TILE_PIXELS: u64 = 1_150_000;

/// The automatic tiling of a `w`×`h` picture: the fewest [`grid_tiles`] of at most
/// [`MAX_TILE_PIXELS`] each, the smaller tiles between equal counts, never a strip of three or
/// more (~1.5 MP strips ran 3–5x slower on the RTX 5090, 2026-10-29). 864x1152, 1232x816 and
/// 1152x864 enlarged 2x take 2x2 where square megapixel tiles took six: 28–32 % faster on the
/// landscapes, the same on the portrait.
pub(crate) fn auto_grid(w: u32, h: u32) -> (u32, u32) {
    let mut best: Option<(u32, u64, (u32, u32))> = None;
    for c in 1..=MAX_GRID {
        for r in 1..=MAX_GRID {
            let n = c * r;
            if (c == 1 || r == 1) && n >= 3 {
                continue;
            }
            let Some(t) = grid_tiles(w, h, c, r).first().copied() else { continue };
            let area = t.width() as u64 * t.height() as u64;
            if area <= MAX_TILE_PIXELS && best.is_none_or(|(bn, ba, _)| (n, area) < (bn, ba)) {
                best = Some((n, area, (c, r)));
            }
        }
    }
    best.map_or((MAX_GRID, MAX_GRID), |b| b.2)
}

/// Tiles per side at most (64 MP, the largest result, takes 9x9 of the automatic tiles).
const MAX_GRID: u32 = 16;

/// A tile's weight at `(x, y)` inside it: rising over [`OVERLAP`] from each side that another
/// tile shares (`shared`: left, top, right, bottom), 1 elsewhere.
pub(crate) fn tile_weight(t: Rect, shared: [bool; 4], x: i32, y: i32) -> f32 {
    let ramp = |d: i32| ((d as f32 + 0.5) / OVERLAP as f32).clamp(0.0, 1.0);
    let mut w = 1.0f32;
    if shared[0] {
        w *= ramp(x - t.x0);
    }
    if shared[1] {
        w *= ramp(y - t.y0);
    }
    if shared[2] {
        w *= ramp(t.x1 - 1 - x);
    }
    if shared[3] {
        w *= ramp(t.y1 - 1 - y);
    }
    w
}

/// `out`'s colours matched to `src`'s: each channel moved to the source's mean and spread.
pub(crate) fn colour_matched(out: &Rgba8, src: &Rgba8) -> Rgba8 {
    if (out.width, out.height) != (src.width, src.height) || out.data.is_empty() {
        return out.clone();
    }
    let stats = |img: &Rgba8, c: usize| -> (f64, f64) {
        let n = (img.data.len() / 4).max(1) as f64;
        let mean = img.data.chunks(4).map(|p| f64::from(p.get(c).copied().unwrap_or(0))).sum::<f64>() / n;
        let var = img.data.chunks(4).map(|p| (f64::from(p.get(c).copied().unwrap_or(0)) - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    };
    let mut data = out.data.clone();
    for c in 0..3 {
        let ((mo, so), (ms, ss)) = (stats(out, c), stats(src, c));
        let gain = if so > 1.0 { ss / so } else { 1.0 };
        // Byte inputs have only 256 possible corrections; retain the same f64 arithmetic.
        let lut: [u8; 256] = std::array::from_fn(|v| ((v as f64 - mo) * gain + ms).round().clamp(0.0, 255.0) as u8);
        for p in data.chunks_mut(4) {
            if let Some(v) = p.get_mut(c) {
                *v = lut.get(usize::from(*v)).copied().unwrap_or(*v);
            }
        }
    }
    Rgba8::new(out.width, out.height, data).unwrap_or_else(|_| out.clone())
}

pub(crate) fn run_upscale(s: &mut Session, p: &Value) -> Result<Value> {
    let started = now_ms();
    let scale = opt_num(UPSCALE, p, "scale", 1.0, 4.0)?.unwrap_or(2.0);
    let tile_mp = opt_num(UPSCALE, p, "tileMegapixels", 0.25, 2.0)?.unwrap_or(1.0);
    let prompt = opt_str(UPSCALE, p, "prompt", 4000)?.map(str::trim).filter(|t| !t.is_empty()).unwrap_or(RESTORE_PROMPT).to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (w, h) = (d.doc.size.width, d.doc.size.height);
    let (tw, th) = ((f64::from(w) * scale).round() as u32, (f64::from(h) * scale).round() as u32);
    if tw == 0 || th == 0 || u64::from(tw) * u64::from(th) > MAX_PIXELS {
        return Err(EngineError::Other(format!("{tw}×{th} is beyond what Generative Upscale makes (64 megapixels)")));
    }
    // Under Auto: the 2.1 edit when the native engine serves it, else the edit default.
    let native21 = s.prefs().integrations.allow_research_models && runs_native(s, "qwen-2.1/edit");
    let default_template = if native21 { "qwen-2.1/edit" } else { DEFAULT_EDIT_TEMPLATE };
    let mut q = p.clone();
    if let Some(o) = q.as_object_mut() {
        o.insert("prompt".into(), json!(prompt));
    }
    let common = plan_common(s, UPSCALE, &q, default_template, Task::Edit, &s.prefs().integrations.default_edit_model.clone())?;
    let mut loras = loras_of(UPSCALE, p)?;
    let restore_strength = opt_num(UPSCALE, p, "restoreStrength", 0.0, 1.5)?.map_or(RESTORE_STRENGTH, |v| v as f32);
    let models_root = s.prefs().integrations.models_root();
    if loras.is_empty()
        && p.get("loras").is_none()
        && common.template.meta.family == "qwen-image-2.1"
        && runs_native(s, &common.template.meta.id)
        && !models_root.is_empty()
        && std::path::Path::new(&models_root).join("loras").join(RESTORE_LORA).is_file()
        && restore_strength > 0.0
    {
        loras.push((RESTORE_LORA.to_string(), restore_strength));
    }
    let batch = common.template.meta.family == "qwen-image-2.1" && runs_native(s, &common.template.meta.id);
    let (backend, _server) = backend_for(s, &common.template.meta.id)?;
    let side = ((tile_mp * 1e6).sqrt() as u32 / 16 * 16).max(256);
    let grid = match p.get("grid") {
        None => None,
        Some(g) => match g.as_array().map(|a| a.iter().map(|v| v.as_u64().filter(|n| (1..=u64::from(MAX_GRID)).contains(n))).collect::<Option<Vec<u64>>>()) {
            Some(Some(v)) if v.len() == 2 => Some((v[0] as u32, v[1] as u32)),
            _ => return Err(EngineError::BadParams { cmd: UPSCALE.into(), msg: format!("grid is [columns, rows], 1 to {MAX_GRID} each") }),
        },
    };
    // Without a grid or a tile size: the automatic grid for the enlarged picture.
    let grid = grid.or_else(|| p.get("tileMegapixels").is_none().then(|| auto_grid(tw, th)));
    // In the result: the tiling it ran with, for a script to pass again as `grid`.
    let recorded_grid = grid.map(|(c, r)| json!([c, r]));
    let enlarge = now_ms();
    if (tw, th) != (w, h) {
        s.execute("image.imageSize", json!({"width": tw, "height": th, "resample": "bicubic"}))?;
    }
    let enlarge_ms = now_ms() - enlarge;
    let label = common.name.clone().unwrap_or_else(|| "Generative Upscale".to_string());
    let template_id = common.template.meta.id.clone();
    let max_pixels = (side as u64) * (side as u64);
    let params: BTreeMap<String, Value> = if loras.is_empty() {
        BTreeMap::new()
    } else {
        BTreeMap::from([("loras".to_string(), Value::Array(loras.iter().map(|(f, s)| json!({"file": f, "strength": s})).collect()))])
    };
    crate::jobs::edit_job(
        s,
        &label.clone(),
        move |doc, active, ctx| {
            let render = now_ms();
            let rect = doc.bounds();
            let c = photocraft_compose::render(doc, rect).to_rgba8();
            let big = Rgba8::new(c.width, c.height, c.pixels).map_err(gen_err)?;
            let (bw, bh) = (big.width, big.height);
            let render_ms = now_ms() - render;
            let grid = match grid {
                Some((c, r)) => grid_tiles(bw, bh, c, r),
                None => tiles(bw, bh, side),
            };
            // A tile of a chosen grid goes at its own size.
            let max_pixels = grid.first().map_or(max_pixels, |t| max_pixels.max(t.width() as u64 * t.height() as u64));
            let n = grid.len();
            // Every tile cut first: each comes from the enlarged picture alone.
            let mut cut = Vec::with_capacity(n);
            for t in &grid {
                let crop = now_ms();
                let src = super::crop_rgba8(&big, *t);
                let (rw, rh) = request_size_within(src.width, src.height, max_pixels);
                let sent = if (rw, rh) == (src.width, src.height) { src.clone() } else { resize_rgba8(&src, rw, rh)? };
                cut.push((src, sent, now_ms() - crop));
            }
            let request_of = |i: usize, image: Rgba8, references: Vec<Rgba8>, params: BTreeMap<String, Value>| Request {
                seed: common.seed.wrapping_add(i as u64) % super::MAX_SEED,
                steps: common.steps,
                guidance: common.guidance,
                negative: common.negative.clone(),
                image: Some(image),
                references,
                models: common.models.clone(),
                params,
                ..Request::new(&template_id, &common.prompt)
            };
            // The native 2.1 edit takes every tile in one request (`batch`: each tile's seed),
            // the pictures encoded together; each tile comes back as its own request would
            // make it. A worker that does not take the batch is asked tile by tile.
            let mut batched: Vec<Option<Rgba8>> = Vec::new();
            let mut batch_ms = None;
            let mut ms = 0u64;
            if batch && n > 1 {
                ctx.check()?;
                ctx.progress(0.0, &format!("{n} tiles"));
                let mut bp = params.clone();
                bp.insert("batch".into(), json!((0..n).map(|i| common.seed.wrapping_add(i as u64) % super::MAX_SEED).collect::<Vec<_>>()));
                let req = request_of(0, cut[0].1.clone(), cut[1..].iter().map(|c| c.1.clone()).collect(), bp);
                // The worker numbers each tile's pictures by its place in the request: each lands
                // on its own tile, all of them at once.
                ctx.preview_places(grid.iter().map(|t| t.translate(rect.x0, rect.y0)).collect());
                let request = now_ms();
                match backend.as_ref().run(&req, &JobProgress::span(ctx, 0.0, 1.0)) {
                    Ok(resp) if resp.images.len() == n => {
                        ms += resp.elapsed_ms;
                        batch_ms = Some(now_ms() - request);
                        batched = resp.images.into_iter().map(Some).collect();
                    }
                    Err(e @ (photocraft_genai::Error::Cancelled | photocraft_genai::Error::Timeout(_))) => return Err(gen_err(e)),
                    Ok(_) | Err(_) => {}
                }
            }
            let mut acc = vec![0.0f32; bw as usize * bh as usize * 3];
            let mut wsum = vec![0.0f32; bw as usize * bh as usize];
            let mut tile_timings = Vec::with_capacity(n);
            for (i, (t, (src, sent, crop_ms))) in grid.iter().zip(cut).enumerate() {
                ctx.check()?;
                let (lo, hi) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
                let (rw, rh) = (sent.width, sent.height);
                let request = now_ms();
                let (out, backend_timings) = match batched.get_mut(i).and_then(Option::take) {
                    Some(out) => (out, None),
                    None => {
                        ctx.progress(lo, &format!("Tile {}/{n}", i + 1));
                        // One tile per request: its pictures land on that tile.
                        ctx.preview_at(t.translate(rect.x0, rect.y0));
                        let resp = backend.as_ref().run(&request_of(i, sent, Vec::new(), params.clone()), &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?;
                        ms += resp.elapsed_ms;
                        let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
                        (out, Some(resp.timings))
                    }
                };
                let request_ms = now_ms() - request;
                let align = now_ms();
                let out = if (out.width, out.height) == (src.width, src.height) { out } else { resize_rgba8(&out, src.width, src.height)? };
                let d = crate::generate_realign::drift(&out, &src, None, crate::generate_realign::max_shift(out.width, out.height));
                let out = crate::generate_realign::shifted(&out, &src, d);
                let align_ms = now_ms() - align;
                let colour = now_ms();
                let out = colour_matched(&out, &src);
                let colour_ms = now_ms() - colour;
                let blend = now_ms();
                let shared = [t.x0 > 0, t.y0 > 0, (t.x1 as u32) < bw, (t.y1 as u32) < bh];
                for y in t.y0..t.y1 {
                    for x in t.x0..t.x1 {
                        let wt = tile_weight(*t, shared, x, y);
                        let (lx, ly) = ((x - t.x0) as usize, (y - t.y0) as usize);
                        let o = (ly * out.width as usize + lx) * 4;
                        let g = y as usize * bw as usize + x as usize;
                        for c in 0..3 {
                            if let (Some(a), Some(v)) = (acc.get_mut(g * 3 + c), out.data.get(o + c)) {
                                *a += wt * f32::from(*v);
                            }
                        }
                        if let Some(s) = wsum.get_mut(g) {
                            *s += wt;
                        }
                    }
                }
                tile_timings.push(json!({"tile": i, "x": t.x0, "y": t.y0, "width": rw, "height": rh,
                    "cropMs": crop_ms, "requestMs": request_ms, "backend": backend_timings,
                    "alignMs": align_ms, "colourMs": colour_ms, "blendMs": now_ms() - blend}));
            }
            let composite = now_ms();
            let mut data = Vec::with_capacity(bw as usize * bh as usize * 4);
            for (g, s) in wsum.iter().enumerate() {
                for c in 0..3 {
                    let v = acc.get(g * 3 + c).copied().unwrap_or(0.0) / s.max(1e-6);
                    data.push(v.round().clamp(0.0, 255.0) as u8);
                }
                data.push(255);
            }
            let redrawn = Rgba8::new(bw, bh, data).map_err(gen_err)?;
            let mut layer = Layer::raster(label.clone(), doc.pixel_format());
            write_rgba8(&mut layer, &redrawn, rect)?;
            set_generative_info(
                &mut layer,
                &GenerativeInfo {
                    command: UPSCALE.to_string(),
                    prompt: common.prompt.clone(),
                    template: template_id.clone(),
                    seed: common.seed,
                    steps: common.steps,
                    guidance: common.guidance,
                    name: label.clone(),
                    width: bw,
                    height: bh,
                    ..GenerativeInfo::default()
                },
            );
            let id = doc.insert_above(*active, layer);
            *active = Some(id);
            let timings = json!({"enlargeMs": enlarge_ms, "renderMs": render_ms, "batchMs": batch_ms, "tiles": tile_timings,
                "compositeMs": now_ms() - composite, "totalMs": now_ms() - started});
            Ok((id, n, bw, bh, ms, template_id, timings))
        },
        move |(id, n, bw, bh, ms, template, timings)| json!({"layer": id.0, "tiles": n, "grid": recorded_grid, "width": bw, "height": bh, "elapsedMs": ms, "template": template, "timings": timings}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_picture_with_shared_strips_and_flush_edges() {
        let t = tiles(2500, 1000, 1024);
        assert_eq!(t.len(), 3, "{t:?}");
        assert_eq!((t[0].x0, t[1].x0, t[2].x0), (0, 896, 1476), "steps of 1024 − 128, the last flush right");
        assert!(t.iter().all(|r| r.width() == 1024 && r.height() == 1000), "one row, the picture's height");
        assert_eq!(tiles(500, 300, 1024), vec![Rect::new(0, 0, 500, 300)], "a small picture is one tile");
        // A chosen grid: 864x1152 enlarged 2x in 2x2 and 1x3, every neighbour sharing >= the strip.
        let g = grid_tiles(1728, 2304, 2, 2);
        assert_eq!(g, vec![Rect::new(0, 0, 928, 1216), Rect::new(800, 0, 1728, 1216), Rect::new(0, 1088, 928, 2304), Rect::new(800, 1088, 1728, 2304)]);
        let g = grid_tiles(1728, 2304, 1, 3);
        assert_eq!(g.iter().map(|r| (r.y0, r.height(), r.width())).collect::<Vec<_>>(), vec![(0, 864, 1728), (720, 864, 1728), (1440, 864, 1728)]);
        assert_eq!(grid_tiles(500, 300, 1, 1), vec![Rect::new(0, 0, 500, 300)]);
        // The automatic grid: 864x1152, 1232x816 and 1152x864 enlarged 2x take 2x2.
        assert_eq!(auto_grid(1728, 2304), (2, 2));
        assert_eq!(auto_grid(2464, 1632), (2, 2));
        assert_eq!(auto_grid(2304, 1728), (2, 2));
        assert_eq!(auto_grid(600, 400), (1, 1), "a picture within a tile is one tile");
        for (w, h) in [(4000, 700), (700, 4000), (8000, 8000), (1728, 2304)] {
            let (c, r) = auto_grid(w, h);
            assert!(!((c == 1 || r == 1) && c * r >= 3), "{w}x{h}: no strip of three or more ({c}x{r})");
            let t = grid_tiles(w, h, c, r)[0];
            assert!(t.width() as u64 * t.height() as u64 <= MAX_TILE_PIXELS, "{w}x{h}: {c}x{r} tiles of {}x{}", t.width(), t.height());
        }
        let w = |x| tile_weight(Rect::new(0, 0, 1024, 1000), [false, false, true, false], x, 10);
        assert_eq!(w(0), 1.0, "no neighbour on the left");
        assert!(w(1023) < 0.01 && w(1023 - 64) > 0.45 && w(1023 - 64) < 0.55, "fades out across the right strip");
    }

    #[test]
    fn an_upscale_enlarges_the_picture_and_redraws_it_as_a_layer_tile_by_tile() {
        let fake = photocraft_genai::fake::FakeNative::start().unwrap();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 300, "height": 200})).unwrap();
        s.edit_prefs(|p| {
            p.integrations.models_path = fake.url.clone();
            p.integrations.runtime_path = "fake".into();
            p.integrations.allow_research_models = false;
        });
        let layers = s.active().unwrap().doc.layers.len();
        let r = s.execute(UPSCALE, json!({"scale": 2, "tileMegapixels": 0.25, "seed": 5})).unwrap();
        // 600×400 in tiles of 496 px: two across (0 and 104), one down.
        assert_eq!((r["tiles"].as_u64(), r["width"].as_u64(), r["height"].as_u64()), (Some(2), Some(600), Some(400)), "{r}");
        let timings = &r["timings"];
        assert_eq!(timings["tiles"].as_array().unwrap().len(), 2);
        assert_eq!(timings["tiles"][0]["width"], 512, "model requests round the 496-pixel source tile to a multiple of 32");
        assert!(timings["totalMs"].as_f64().unwrap() >= timings["enlargeMs"].as_f64().unwrap());
        let d = s.active().unwrap();
        assert_eq!((d.doc.size.width, d.doc.size.height), (600, 400), "enlarged first");
        assert_eq!(d.doc.layers.len(), layers + 1, "the redraw is one layer above");
        let st = fake.state();
        assert_eq!(st.runs.len(), 2, "a run per tile");
        assert!(st.runs[0].request.to_string().contains("Restore the photo"), "the restore wording by default");
        drop(st);
        assert!(s.execute(UPSCALE, json!({"scale": 9})).is_err(), "scale 1 to 4");
    }

    /// The native 2.1 edit takes the tiles as one batch; a worker that answers it with one
    /// picture is asked tile by tile, and both redraw the same pixels.
    #[test]
    fn a_native_21_upscale_sends_its_tiles_as_one_batch() {
        let upscale = |no_batch: bool| {
            let fake = photocraft_genai::fake::FakeNative::start_with(photocraft_genai::fake::Options { echo: true, no_batch, ..Default::default() }).unwrap();
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 300, "height": 200})).unwrap();
            s.execute("filter.addNoise", json!({"amount": 40, "seed": 3})).ok();
            s.edit_prefs(|p| {
                p.integrations.models_path = fake.url.clone();
                p.integrations.runtime_path = "fake".into();
                p.integrations.allow_research_models = true;
            });
            let r = s.execute(UPSCALE, json!({"scale": 2, "tileMegapixels": 0.25, "seed": 5, "template": "qwen-2.1/edit"})).unwrap();
            assert_eq!(r["tiles"].as_u64(), Some(2), "{r}");
            let d = s.active().unwrap();
            let pixels = photocraft_compose::render(&d.doc, d.doc.bounds()).to_rgba8().pixels;
            let st = fake.state();
            let runs: Vec<_> = st.runs.iter().map(|r| (r.run.seed, r.run.references.len(), r.run.params.get("batch").cloned())).collect();
            (runs, pixels)
        };
        let (runs, batched) = upscale(false);
        assert_eq!(runs, vec![(5, 1, Some(json!([5, 6])))], "one request: tile 0 the picture, tile 1 its reference, a seed each");
        let (runs, tiled) = upscale(true);
        assert_eq!(runs, vec![(5, 1, Some(json!([5, 6]))), (5, 0, None), (6, 0, None)], "the batch refused, then a request per tile");
        assert!(batched == tiled, "the same redraw either way");
    }

    /// A batched Upscale reports the worker's progress while it runs, and a cancel stops the
    /// batch at once and leaves the document as a cancelled tile-by-tile Upscale does: no
    /// layer, and no tile asked again.
    #[test]
    fn a_batched_upscale_reports_progress_and_cancels_like_tile_by_tile() {
        use std::time::{Duration, Instant};
        let fake =
            photocraft_genai::fake::FakeNative::start_with(photocraft_genai::fake::Options { delay_ms: 20_000, progress_steps: 200, ..Default::default() })
                .unwrap();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 300, "height": 200})).unwrap();
        s.edit_prefs(|p| {
            p.integrations.models_path = fake.url.clone();
            p.integrations.runtime_path = "fake".into();
            p.integrations.allow_research_models = true;
        });
        let layers = s.active().unwrap().doc.layers.len();
        let id = match s.start(UPSCALE, json!({"scale": 2, "tileMegapixels": 0.25, "seed": 5, "template": "qwen-2.1/edit"})).unwrap() {
            crate::jobs::Started::Job(id) => id,
            crate::jobs::Started::Done(v) => panic!("ran inline: {v}"),
        };
        let t = Instant::now();
        let progress = || s.jobs_with_recent().into_iter().find(|j| j.id == id).map_or(0.0, |j| j.progress);
        while progress() <= 0.0 {
            assert!(t.elapsed() < Duration::from_secs(10), "no progress from the batch");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(progress() < 1.0);
        let t = Instant::now();
        assert!(s.cancel_job(id));
        s.join_cancelled_jobs();
        assert!(t.elapsed() < Duration::from_secs(5), "cancel took {:?}", t.elapsed());
        assert_eq!(s.active().unwrap().doc.layers.len(), layers, "no redraw layer");
        let st = fake.state();
        assert_eq!(st.cancellations, 1, "the worker was told to stop");
        assert_eq!(st.runs.len(), 1, "the batch alone: a cancel is not a refusal to fall back from");
    }

    /// Run in the background, an Upscale records as one step, as inline: its Image Size is not
    /// recorded as well (a replay would enlarge twice).
    #[test]
    fn a_background_upscale_records_one_step() {
        use std::time::{Duration, Instant};
        let fake = photocraft_genai::fake::FakeNative::start_with(photocraft_genai::fake::Options { delay_ms: 50, ..Default::default() }).unwrap();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 300, "height": 200})).unwrap();
        s.edit_prefs(|p| {
            p.integrations.models_path = fake.url.clone();
            p.integrations.runtime_path = "fake".into();
            p.integrations.allow_research_models = true;
        });
        let action = s.execute("actions.record", json!({"name": "A"})).unwrap()["index"].as_u64().unwrap() as usize;
        let id = match s.start(UPSCALE, json!({"scale": 2, "tileMegapixels": 0.25, "seed": 5, "template": "qwen-2.1/edit"})).unwrap() {
            crate::jobs::Started::Job(id) => id,
            crate::jobs::Started::Done(v) => panic!("ran inline: {v}"),
        };
        let t = Instant::now();
        let e = loop {
            if let Some(e) = s.poll_jobs().into_iter().find(|e| e.id == id) {
                break e;
            }
            assert!(t.elapsed() < Duration::from_secs(30), "job did not finish");
            std::thread::sleep(Duration::from_millis(5));
        };
        assert!(matches!(e.outcome, crate::jobs::JobOutcome::Done(_)), "{e:?}");
        s.execute("actions.stop", json!({})).unwrap();
        let ids: Vec<&str> = s.actions.list[action].steps.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, [UPSCALE]);
        assert_eq!(s.journal.iter().filter(|(id, _)| id == "image.imageSize").count(), 0);
    }

    #[test]
    fn colour_lookup_preserves_every_byte_and_alpha() {
        let mut source = Vec::new();
        let mut result = Vec::new();
        for v in 0..=255u8 {
            source.extend([v / 2 + 64, v.saturating_mul(2), v / 3, 255]);
            result.extend([v, 255 - v, 128, v]);
        }
        let src = Rgba8::new(256, 1, source).unwrap();
        let out = Rgba8::new(256, 1, result).unwrap();
        let mut expected = out.data.clone();
        for c in 0..3 {
            let stats = |img: &Rgba8| {
                let mean = img.data.as_chunks::<4>().0.iter().map(|p| f64::from(p[c])).sum::<f64>() / 256.0;
                let spread = (img.data.as_chunks::<4>().0.iter().map(|p| (f64::from(p[c]) - mean).powi(2)).sum::<f64>() / 256.0).sqrt();
                (mean, spread)
            };
            let ((mo, so), (ms, ss)) = (stats(&out), stats(&src));
            let gain = if so > 1.0 { ss / so } else { 1.0 };
            for pixel in expected.as_chunks_mut::<4>().0 {
                pixel[c] = ((f64::from(pixel[c]) - mo) * gain + ms).round().clamp(0.0, 255.0) as u8;
            }
        }
        assert_eq!(colour_matched(&out, &src).data, expected);
    }

    #[test]
    fn a_tiles_colours_are_moved_to_its_sources() {
        let src = Rgba8::new(2, 1, vec![100, 50, 200, 255, 140, 70, 220, 255]).unwrap();
        let out = Rgba8::new(2, 1, vec![130, 40, 190, 255, 210, 80, 230, 255]).unwrap();
        let m = colour_matched(&out, &src);
        assert_eq!(&m.data[..3], &[100, 50, 200], "{:?}", m.data);
        assert_eq!(&m.data[4..7], &[140, 70, 220]);
    }
}
