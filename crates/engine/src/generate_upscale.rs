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
    let (backend, _server) = backend_for(s, &common.template.meta.id)?;
    let side = ((tile_mp * 1e6).sqrt() as u32 / 16 * 16).max(256);
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
            let grid = tiles(bw, bh, side);
            let n = grid.len();
            let mut acc = vec![0.0f32; bw as usize * bh as usize * 3];
            let mut wsum = vec![0.0f32; bw as usize * bh as usize];
            let mut ms = 0u64;
            let mut tile_timings = Vec::with_capacity(n);
            for (i, t) in grid.iter().enumerate() {
                ctx.check()?;
                let (lo, hi) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
                ctx.progress(lo, &format!("Tile {}/{n}", i + 1));
                let crop = now_ms();
                let src = super::crop_rgba8(&big, *t);
                let (rw, rh) = request_size_within(src.width, src.height, max_pixels);
                let sent = if (rw, rh) == (src.width, src.height) { src.clone() } else { resize_rgba8(&src, rw, rh)? };
                let crop_ms = now_ms() - crop;
                let req = Request {
                    seed: common.seed.wrapping_add(i as u64) % super::MAX_SEED,
                    steps: common.steps,
                    guidance: common.guidance,
                    negative: common.negative.clone(),
                    image: Some(sent),
                    models: common.models.clone(),
                    params: params.clone(),
                    ..Request::new(&template_id, &common.prompt)
                };
                let request = now_ms();
                let resp = backend.as_ref().run(&req, &JobProgress::span(ctx, lo, hi)).map_err(gen_err)?;
                let request_ms = now_ms() - request;
                ms += resp.elapsed_ms;
                let backend_timings = resp.timings;
                let align = now_ms();
                let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the backend returned no image".into()))?;
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
            let timings = json!({"enlargeMs": enlarge_ms, "renderMs": render_ms, "tiles": tile_timings,
                "compositeMs": now_ms() - composite, "totalMs": now_ms() - started});
            Ok((id, n, bw, bh, ms, template_id, timings))
        },
        |(id, n, bw, bh, ms, template, timings)| json!({"layer": id.0, "tiles": n, "width": bw, "height": bh, "elapsedMs": ms, "template": template, "timings": timings}),
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
