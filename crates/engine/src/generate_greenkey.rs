//! Key Green Screen (AI) (`layer.matting.keyGreenScreen`): a green- or blue-screen picture to a
//! new layer of the subject with a clean alpha and its straight, un-mixed colour, from the
//! native worker's keyer (`greenkey/corridorkey-v1`: a clean-room engine for CorridorKey's
//! design, Corridor Digital; the user brings CorridorKey's weights, CC BY-NC-SA 4.0).
//!
//! The keyer needs a rough hint of the subject. `hint`: `chroma` keys the screen colour by
//! distance (thresholded, slightly eroded: it only has to be roughly right); `subject` asks SAM
//! 3.1 for the main subject; `auto` takes chroma when most of the frame's border is a screen
//! colour, else the subject. The screen colour (`screen`: auto, green, blue) comes from the
//! border pixels too; it is what despill takes out.

use std::collections::BTreeMap;

use photocraft_geom::Rect;

use super::{
    JobProgress, Layer, Request, Result, Rgba8, Session, Value, backend_for, bad, fit_pixels, gen_err, json, opt_bool, opt_num, opt_str, resize_rgba8,
    resolve_auto, web_unavailable, write_rgba8,
};
use crate::EngineError;
use crate::select_ml_cmds::{picture_rgba, subject_finder};

pub(crate) const KEY_GREEN: &str = "layer.matting.keyGreenScreen";
pub(crate) const GREENKEY_TEMPLATE: &str = "greenkey/corridorkey-v1";

/// The most pixels one key request carries (the model works at 2048², the worker resizes back
/// to what it is sent).
const MAX_KEY_PIXELS: u64 = 4096 * 4096;
/// What the detector is asked for with `hint: subject`.
const SUBJECT_PROMPT: &str = "the main subject";
/// The share of the border that must be screen-coloured for `hint: auto` to key by colour.
const BORDER_SCREEN_SHARE: f32 = 0.6;
/// The border ring sampled for the screen colour, as a fraction of the shorter side.
const BORDER_RING: f32 = 0.03;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Screen {
    Green,
    Blue,
}

impl Screen {
    fn name(self) -> &'static str {
        match self {
            Screen::Green => "green",
            Screen::Blue => "blue",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HintKind {
    Auto,
    Subject,
    Chroma,
}

pub(super) fn enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let d = s.active().ok_or("no document open")?;
    crate::cutout_cmds::check(&d.doc, crate::active_layer_of(s)?)
}

/// Is this pixel a screen colour, and which?
fn screen_of(p: [u8; 4]) -> Option<Screen> {
    let (r, g, b) = (f32::from(p[0]), f32::from(p[1]), f32::from(p[2]));
    if g > 50.0 && g > 1.2 * r && g > 1.2 * b {
        Some(Screen::Green)
    } else if b > 50.0 && b > 1.2 * r && b > 1.1 * g {
        Some(Screen::Blue)
    } else {
        None
    }
}

/// The border ring's screen: which screen most of it is, the share that is, and its mean
/// colour (0-255). Falls back to green when unsure.
// The screen colour is voted from the frame's border, before any hint exists (the algo crate's
// `detect_screen` needs a hint); an explicit `screen` overrides it.
pub(crate) fn border_screen(px: &[[u8; 4]], w: usize, h: usize) -> (Screen, f32, [f32; 3]) {
    let ring = ((w.min(h) as f32 * BORDER_RING).ceil() as usize).max(1);
    let (mut n, mut green, mut blue) = (0usize, 0usize, 0usize);
    let (mut sum_g, mut sum_b) = ([0f64; 3], [0f64; 3]);
    for y in 0..h {
        for x in 0..w {
            if x >= ring && y >= ring && x + ring < w && y + ring < h {
                continue;
            }
            let p = px[y * w + x];
            n += 1;
            match screen_of(p) {
                Some(Screen::Green) => {
                    green += 1;
                    for c in 0..3 {
                        sum_g[c] += f64::from(p[c]);
                    }
                }
                Some(Screen::Blue) => {
                    blue += 1;
                    for c in 0..3 {
                        sum_b[c] += f64::from(p[c]);
                    }
                }
                None => {}
            }
        }
    }
    let n = n.max(1) as f32;
    let (screen, count, sum) = if blue > green { (Screen::Blue, blue, sum_b) } else { (Screen::Green, green, sum_g) };
    let mean = if count > 0 {
        [(sum[0] / count as f64) as f32, (sum[1] / count as f64) as f32, (sum[2] / count as f64) as f32]
    } else {
        match screen {
            Screen::Green => [0.0, 177.0, 64.0],
            Screen::Blue => [0.0, 71.0, 187.0],
        }
    };
    (screen, count as f32 / n, mean)
}

/// A chroma hint: subject (255) where a pixel is far from the screen colour, eroded a little so
/// the solid part stays inside the subject; what is only partly away from the screen colour (a
/// see-through thing, motion blur, the eroded rim) gets a partial value that grows with its
/// distance, so the keyer is told "maybe" there rather than "screen".
pub(crate) fn chroma_hint(px: &[[u8; 4]], w: usize, h: usize, key: [f32; 3]) -> Vec<u8> {
    let norm = (key[0] * key[0] + key[1] * key[1] + key[2] * key[2]).sqrt().max(1.0);
    // Far from the screen: at least 40 % of the screen colour's own length away; partly away
    // from 15 % of it (noise and a little spill stay screen).
    let threshold = (0.4 * norm).max(48.0);
    let low = (0.15 * norm).max(18.0).min(threshold * 0.5);
    let dist: Vec<f32> = px.iter().map(|p| (0..3).map(|c| (f32::from(p[c]) - key[c]).powi(2)).sum::<f32>().sqrt()).collect();
    let off: Vec<bool> = dist.iter().map(|d| *d <= threshold).collect();
    let depth = photocraft_algo::selection::edt(&off, w, h);
    let erode = (w.min(h) as f32 / 400.0).max(2.0);
    dist.iter()
        .zip(depth.iter().chain(std::iter::repeat(&0.0)))
        .map(|(d, e)| {
            if *d > threshold && *e > erode {
                255
            } else if *d > low {
                (((d.min(threshold) - low) / (threshold - low)) * PARTIAL_HINT).round() as u8
            } else {
                0
            }
        })
        .collect()
}

/// The highest hint value a pixel only partly away from the screen colour gets.
const PARTIAL_HINT: f32 = 160.0;

/// Drops the small alpha islands the keyer left (tracking markers, specks) and limits the alpha
/// to a grown, softened garbage matte of what stays (`photocraft_algo::matting::despeckle`, its
/// minimum area scaled with the picture). The subject's own soft edge is untouched.
fn despeckle_hook(alpha: &mut [u8], w: usize, h: usize) -> bool {
    let mut a: Vec<f32> = alpha.iter().map(|v| f32::from(*v) / 255.0).collect();
    let p = photocraft_algo::matting::DespeckleParams::default();
    photocraft_algo::matting::despeckle(&mut a, w, h, &p);
    for (o, v) in alpha.iter_mut().zip(&a) {
        *o = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    true
}

/// Takes the screen's light off the keyed colour: the screen channel limited to the mean of the
/// other two, the removed amount handed back evenly (brightness kept), blended by `strength`.
/// Works on the keyer's straight sRGB colour, as the model gives it.
fn despill_hook(rgba: &mut [u8], screen: Screen, strength: f32) -> bool {
    let screen = match screen {
        Screen::Green => photocraft_algo::matting::Screen::Green,
        Screen::Blue => photocraft_algo::matting::Screen::Blue,
    };
    for p in rgba.as_chunks_mut::<4>().0 {
        let rgb = [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0];
        let out = photocraft_algo::matting::despill_pixel(rgb, screen, photocraft_algo::matting::SpillLimit::Average, strength);
        for c in 0..3 {
            p[c] = (out[c].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    true
}

/// `layer.matting.keyGreenScreen`.
pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    crate::cutout_cmds::check(doc, crate::active_layer_of(s).map_err(EngineError::Other)?).map_err(EngineError::Other)?;
    let screen_arg = match opt_str(KEY_GREEN, p, "screen", 10)? {
        None | Some("auto") => None,
        Some("green") => Some(Screen::Green),
        Some("blue") => Some(Screen::Blue),
        Some(other) => return Err(bad(KEY_GREEN, format!("`screen` must be auto, green or blue (got `{other}`)"))),
    };
    let hint_kind = match opt_str(KEY_GREEN, p, "hint", 10)? {
        None | Some("auto") => HintKind::Auto,
        Some("subject") => HintKind::Subject,
        Some("chroma") => HintKind::Chroma,
        Some(other) => return Err(bad(KEY_GREEN, format!("`hint` must be auto, subject or chroma (got `{other}`)"))),
    };
    let strength = opt_num(KEY_GREEN, p, "refinerStrength", 0.0, 4.0)?.unwrap_or(1.0) as f32;
    let despill = opt_num(KEY_GREEN, p, "despill", 0.0, 1.0)?.unwrap_or(1.0) as f32;
    let despeckle = opt_bool(KEY_GREEN, p, "despeckle", true)?;
    let all_layers = opt_bool(KEY_GREEN, p, "sampleAllLayers", false)?;
    let name = opt_str(KEY_GREEN, p, "name", 200)?.unwrap_or("Keyed").to_string();
    let (backend, _) = backend_for(s, GREENKEY_TEMPLATE)?;
    // The detector, when a subject hint may be wanted and SAM 3.1 is set up.
    let finder = if hint_kind == HintKind::Chroma { None } else { subject_finder(s, Some(SUBJECT_PROMPT)).ok() };
    if hint_kind == HintKind::Subject && finder.is_none() {
        return Err(bad(KEY_GREEN, "`hint: subject` needs SAM 3.1 (see Preferences > Generative)"));
    }
    crate::jobs::edit_job(
        s,
        "Key Green Screen",
        move |doc, active, ctx| {
            let started = std::time::Instant::now();
            if resolve_auto(backend.as_ref(), &[GREENKEY_TEMPLATE], "") != GREENKEY_TEMPLATE {
                return Err(EngineError::Other(
                    "the green-screen keyer needs CorridorKey's weights: download CorridorKey_v1.0.safetensors from the authors' Hugging Face page into models/corridorkey/".into(),
                ));
            }
            let frame: Rect = doc.bounds();
            let (w, h) = (frame.width() as usize, frame.height() as usize);
            ctx.progress(0.0, "Reading the picture");
            let picture = picture_rgba(doc, *active, all_layers);
            let px = picture.as_chunks::<4>().0;
            let (border, share, key) = border_screen(px, w, h);
            let screen = screen_arg.unwrap_or(border);
            let screen_found = share >= BORDER_SCREEN_SHARE;
            let use_chroma = match hint_kind {
                HintKind::Chroma => true,
                HintKind::Subject => false,
                HintKind::Auto => screen_found || finder.is_none(),
            };
            ctx.progress(0.05, "Making the hint");
            // SAM 3.1's main subject: the hint on a picture with no screen, and on a screen in Auto
            // it joins the chroma hint, so a part close to the screen colour (a grey sleeve under
            // green light) is still the subject's and does not come out with holes.
            let subject = |ctx: &crate::jobs::JobCtx| -> Result<Vec<u8>> {
                let Some(f) = finder.as_ref() else { return Err(EngineError::Other("no subject finder".into())) };
                let found = f.find_in(&picture, frame, SUBJECT_PROMPT, 1, ctx)?;
                let mut m = vec![0u8; w * h];
                for (cov, _, _) in &found {
                    for (o, c) in m.iter_mut().zip(cov) {
                        if *c >= 0.5 {
                            *o = 255;
                        }
                    }
                }
                Ok(m)
            };
            let (hint, hint_name): (Vec<u8>, &'static str) = if !use_chroma {
                (subject(ctx)?, "subject")
            } else {
                let mut h = chroma_hint(px, w, h, key);
                if hint_kind == HintKind::Auto && finder.is_some() {
                    for (o, v) in h.iter_mut().zip(subject(ctx)?) {
                        *o = (*o).max(v);
                    }
                    (h, "chroma+subject")
                } else {
                    (h, "chroma")
                }
            };
            if !hint.iter().any(|v| *v > 0) {
                return Err(EngineError::Other("found no subject to key in this picture".into()));
            }
            // At most MAX_KEY_PIXELS go to the worker; the answer comes back up to the frame.
            let (rw, rh) = fit_pixels(w as u32, h as u32, MAX_KEY_PIXELS);
            let image = Rgba8::new(w as u32, h as u32, picture.clone()).map_err(gen_err)?;
            let mask = photocraft_genai::Gray8::new(w as u32, h as u32, hint).map_err(gen_err)?;
            let (image, mask) =
                if (rw, rh) == (w as u32, h as u32) { (image, mask) } else { (resize_rgba8(&image, rw, rh)?, super::resize_gray8(&mask, rw, rh)?) };
            let mut params = BTreeMap::new();
            params.insert("refinerStrength".to_string(), json!(strength));
            let req = Request {
                template: GREENKEY_TEMPLATE.to_string(),
                prompt: String::new(),
                negative: String::new(),
                seed: 0,
                steps: 1,
                guidance: 0.0,
                image: Some(image),
                mask: Some(mask),
                models: Vec::new(),
                size: None,
                params,
                references: Vec::new(),
            };
            let resp = backend.run(&req, &JobProgress::span(ctx, 0.1, 0.9)).map_err(gen_err)?;
            let out = resp.images.into_iter().next().ok_or_else(|| EngineError::Other("the keyer answered with no image".into()))?;
            let out = if (out.width, out.height) == (w as u32, h as u32) { out } else { resize_rgba8(&out, w as u32, h as u32)? };
            let mut keyed = out.data;
            ctx.progress(0.92, "Finishing");
            let mut alpha: Vec<u8> = keyed.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
            let despeckled = despeckle && despeckle_hook(&mut alpha, w, h);
            for (p, a) in keyed.as_chunks_mut::<4>().0.iter_mut().zip(&alpha) {
                p[3] = *a;
            }
            // No screen round the picture (and none named): nothing to despill, and the subject's
            // own greens (clothes, plants) must keep their colour.
            let despilled = despill > 0.0 && (screen_found || screen_arg.is_some()) && despill_hook(&mut keyed, screen, despill);
            // The layer holds straight colour and alpha; the compositor premultiplies.
            let mut layer = Layer::raster(name.clone(), doc.pixel_format());
            write_rgba8(&mut layer, &Rgba8::new(w as u32, h as u32, keyed).map_err(gen_err)?, frame)?;
            let id = doc.insert_above(*active, layer);
            *active = Some(id);
            Ok((id, screen, hint_name, despeckled, despilled, started.elapsed().as_millis() as u64))
        },
        |(id, screen, hint, despeckled, despilled, ms)| json!({"layer": id.0, "screen": screen.name(), "hint": hint, "despeckled": despeckled, "despilled": despilled, "ms": ms}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, bg: [u8; 4], fg: [u8; 4]) -> Vec<[u8; 4]> {
        (0..w * h).map(|i| if (i % w).abs_diff(w / 2) < w / 4 && (i / w).abs_diff(h / 2) < h / 4 { fg } else { bg }).collect()
    }

    #[test]
    fn the_border_tells_green_from_blue() {
        let g = frame(80, 60, [10, 180, 60, 255], [200, 150, 120, 255]);
        let (s, share, key) = border_screen(&g, 80, 60);
        assert_eq!(s, Screen::Green);
        assert!(share > 0.99 && (key[1] - 180.0).abs() < 0.5, "{share} {key:?}");
        let b = frame(80, 60, [20, 60, 200, 255], [200, 150, 120, 255]);
        assert_eq!(border_screen(&b, 80, 60).0, Screen::Blue);
        let plain = frame(80, 60, [120, 120, 120, 255], [200, 150, 120, 255]);
        let (s, share, _) = border_screen(&plain, 80, 60);
        assert_eq!((s, share), (Screen::Green, 0.0));
    }

    #[test]
    fn the_chroma_hint_is_the_subject_slightly_eroded() {
        let (w, h) = (80, 60);
        let px = frame(w, h, [10, 180, 60, 255], [200, 150, 120, 255]);
        let hint = chroma_hint(&px, w, h, [10.0, 180.0, 60.0]);
        // Centre on, background off, the subject's edge (x = 21) eroded to a partial "maybe".
        assert_eq!(hint[30 * w + 40], 255);
        assert_eq!(hint[5 * w + 5], 0);
        assert_eq!(hint[30 * w + 21], PARTIAL_HINT as u8);
        assert_eq!(hint[30 * w + 25], 255);
        // A half see-through grey over the screen is a partial hint, not screen.
        let mix = [((10 + 128) / 2) as u8, ((180 + 128) / 2) as u8, ((60 + 128) / 2) as u8, 255];
        let disc = frame(w, h, [10, 180, 60, 255], mix);
        let v = chroma_hint(&disc, w, h, [10.0, 180.0, 60.0])[30 * w + 40];
        assert!(v > 0 && v < 255, "{v}");
    }

    /// A 64x48 green-screen document (a skin-coloured block in the middle) on the fake worker.
    fn green_session(url: &str) -> crate::Session {
        let mut s = crate::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("green", |doc, active| {
            let r = doc.bounds();
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let n = surf.channels();
            let mut px = Vec::new();
            for y in 0..r.height() {
                for x in 0..r.width() {
                    let inside = (16..48).contains(&x) && (12..36).contains(&y);
                    let c = if inside { [0.8, 0.6, 0.5, 1.0] } else { [0.05, 0.7, 0.25, 1.0] };
                    let mut tmp = [0.0f32; 8];
                    let used = photocraft_raster::from_rgba_into(&surf.format(), c, &mut tmp).min(n);
                    px.extend_from_slice(&tmp[..used]);
                }
            }
            surf.write_region(r, &px);
            Ok(())
        })
        .unwrap();
        s.edit_prefs(|p| {
            p.integrations.models_path = url.to_string();
            p.integrations.runtime_path = "fake".into();
        });
        s
    }

    #[test]
    fn keying_sends_the_picture_and_a_chroma_and_subject_hint_and_adds_a_layer() {
        let fake = photocraft_genai::fake::FakeNative::start().unwrap();
        let mut s = green_session(&fake.url);
        let before = s.active().unwrap().doc.layer_count();
        let r = s.execute(KEY_GREEN, json!({"refinerStrength": 0.5})).unwrap();
        assert_eq!((r["screen"].as_str(), r["hint"].as_str()), (Some("green"), Some("chroma+subject")), "{r}");
        assert_eq!(s.active().unwrap().doc.layer_count(), before + 1);
        assert_eq!(s.active().unwrap().active_layer.map(|l| l.0), r["layer"].as_u64());
        let st = fake.state();
        let run = st.runs.iter().find(|r| r.request["model"] == GREENKEY_TEMPLATE).expect("the keyer asked");
        assert!(run.request["mask"].is_object(), "a hint is sent");
        assert_eq!(run.request["params"]["refinerStrength"], 0.5);
    }

    #[test]
    fn bad_params_are_refused() {
        let fake = photocraft_genai::fake::FakeNative::start().unwrap();
        let mut s = green_session(&fake.url);
        assert!(s.execute(KEY_GREEN, json!({"screen": "red"})).is_err());
        assert!(s.execute(KEY_GREEN, json!({"hint": "trimap"})).is_err());
        assert!(s.execute(KEY_GREEN, json!({"refinerStrength": 9})).is_err());
    }
}
