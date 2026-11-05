//! Clown Pass (`generate.clownPass`, ShotoPhopEX, the owner's ask 2026-11-05): SAM 3.1 picks
//! out everything it can name in the picture and each thing becomes its own layer with the
//! picture's own pixels, in a group over the original; nothing is painted, so it is quick and the
//! layers together show exactly the picture. Each layer gets its own label colour, as a clown
//! pass in a render gives each object its own flat colour.
//!
//! How: the detector is asked for each of a few kinds of thing in turn (people, animals, …, any
//! object), up to [`PER_TERM`] instances each. A mask much like one found already is the same
//! thing (the earlier, more specific kind keeps it), and one reaching across nearly the whole
//! frame is the backdrop. Each pixel goes to the smallest thing that covers it, so a bag held in
//! front of a person is the bag's and the person's layer has a hole there; what no thing covers
//! is the Background layer. Hard edges: every pixel lands on exactly one layer.

use photocraft_doc::LabelColor;
use photocraft_geom::Rect;

use super::{GenerativeInfo, Layer, Result, Rgba8, Session, Value, bad, gen_err, json, opt_str, set_generative_info, web_unavailable, write_rgba8};
use crate::EngineError;
use crate::select_ml_cmds::{Instance, SubjectFinder, subject_finder};

pub(crate) const CLOWN_PASS: &str = "generate.clownPass";

/// What the detector is asked for, in turn, when no prompt is given: the specific kinds first
/// (a person found as "person" stays a person when "object" finds them again).
pub(super) const CLOWN_TERMS: &[&str] = &["person", "animal", "vehicle", "plant", "furniture", "food", "bag", "bottle", "cup", "sign", "object"];

/// Instances the detector may return for one term (each comes back as a full-frame mask).
const PER_TERM: usize = 10;
/// Most layers one pass makes (the background included).
pub(super) const MAX_LAYERS: usize = 40;
/// Two masks overlapping this much (intersection over union) are the same thing.
const SAME_THING: f64 = 0.8;
/// A mask whose bounds reach across this share of the frame both ways and that covers this
/// share of it is the backdrop, not a thing.
const BACKDROP_SPAN: f64 = 0.95;
const BACKDROP_COVER: f64 = 0.6;
/// Layers with fewer pixels than this (every pixel taken by smaller things) are dropped.
const MIN_PIXELS: usize = 16;

/// The label colours the layers take in turn.
const LABELS: [LabelColor; 10] = [
    LabelColor::Red,
    LabelColor::Orange,
    LabelColor::Yellow,
    LabelColor::Green,
    LabelColor::Blue,
    LabelColor::Violet,
    LabelColor::Seafoam,
    LabelColor::Indigo,
    LabelColor::Magenta,
    LabelColor::Fuchsia,
];

pub(super) fn enabled(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// One thing found: its mask over its bounds (row-major, the detector's coverage ≥ 0.5; kept
/// cropped, as a full-frame mask per thing ran to gigabytes on a large photo), its bounds, pixel
/// count and what found it.
struct Thing {
    mask: Vec<bool>,
    bounds: Rect,
    pixels: u64,
    term: String,
}

impl Thing {
    fn new(cov: &[f32], bounds: Rect, pixels: u64, term: &str, frame: Rect) -> Self {
        let w = frame.width() as usize;
        let mut mask = Vec::with_capacity(bounds.width() as usize * bounds.height() as usize);
        for y in bounds.y0..bounds.y1 {
            let row = (y - frame.y0) as usize * w;
            for x in bounds.x0..bounds.x1 {
                mask.push(cov.get(row + (x - frame.x0) as usize).is_some_and(|c| *c >= 0.5));
            }
        }
        Thing { mask, bounds, pixels, term: term.to_string() }
    }

    /// Whether the canvas point (x, y) is on the thing.
    fn at(&self, x: i32, y: i32) -> bool {
        let b = self.bounds;
        x >= b.x0 && y >= b.y0 && x < b.x1 && y < b.y1 && self.mask[(y - b.y0) as usize * b.width() as usize + (x - b.x0) as usize]
    }
}

fn bounds_iou(a: Rect, b: Rect) -> f64 {
    let i = a.intersect(&b);
    if i.is_empty() {
        return 0.0;
    }
    let area = |r: Rect| f64::from(r.width()) * f64::from(r.height());
    let inter = area(i);
    inter / (area(a) + area(b) - inter).max(1.0)
}

/// Intersection over union of two things' masks.
fn mask_iou(a: &Thing, b: &Thing) -> f64 {
    let both = a.bounds.intersect(&b.bounds);
    if both.is_empty() {
        return 0.0;
    }
    let mut inter = 0u64;
    for y in both.y0..both.y1 {
        for x in both.x0..both.x1 {
            if a.at(x, y) && b.at(x, y) {
                inter += 1;
            }
        }
    }
    let union = a.pixels + b.pixels - inter;
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

fn is_backdrop(t: &Thing, frame: Rect) -> bool {
    let (fw, fh) = (f64::from(frame.width()), f64::from(frame.height()));
    f64::from(t.bounds.width()) >= BACKDROP_SPAN * fw && f64::from(t.bounds.height()) >= BACKDROP_SPAN * fh && t.pixels as f64 >= BACKDROP_COVER * fw * fh
}

fn title_case(s: &str) -> String {
    let mut c = s.trim().trim_start_matches("the ").chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

/// `generate.clownPass`.
pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    s.active().ok_or(EngineError::NoDocument)?;
    let terms: Vec<String> = match opt_str(CLOWN_PASS, p, "prompt", 400)?.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => t.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        None => CLOWN_TERMS.iter().map(|t| (*t).to_string()).collect(),
    };
    if terms.len() > 24 {
        return Err(bad(CLOWN_PASS, "`prompt` takes at most 24 comma-separated kinds of thing"));
    }
    let name = opt_str(CLOWN_PASS, p, "name", 200)?.unwrap_or("Clown Pass").to_string();
    // One finder per kind of thing, built on the session (backend and template), run in the job.
    let finders: Vec<(String, SubjectFinder)> =
        terms.iter().map(|t| subject_finder(s, Some(t)).map(|f| (t.clone(), f.at_most(PER_TERM)))).collect::<Result<_>>()?;
    crate::jobs::edit_job(
        s,
        "Clown Pass",
        move |doc, active, ctx| {
            let started = std::time::Instant::now();
            let frame = doc.bounds();
            let (w, h) = (frame.width() as usize, frame.height() as usize);
            let mut things: Vec<Thing> = Vec::new();
            let mut last_err = None;
            let mut asked = 0usize;
            for (i, (term, finder)) in finders.iter().enumerate() {
                ctx.progress(0.85 * i as f32 / finders.len().max(1) as f32, &format!("Finding: {term}"));
                let found: Vec<Instance> = match finder.instances(doc, *active, true, ctx) {
                    Ok((found, _)) => found,
                    Err(e) => {
                        ctx.check()?;
                        last_err = Some(e);
                        continue;
                    }
                };
                asked += 1;
                for (cov, bounds, pixels) in found {
                    let t = Thing::new(&cov, bounds, pixels, term, frame);
                    drop(cov);
                    if is_backdrop(&t, frame) {
                        continue;
                    }
                    let same = things.iter().any(|o| bounds_iou(o.bounds, t.bounds) >= 0.5 && mask_iou(o, &t) >= SAME_THING);
                    if !same {
                        things.push(t);
                    }
                }
            }
            if things.is_empty() {
                return Err(match last_err {
                    Some(e) if asked == 0 => e,
                    _ => EngineError::Other("Clown Pass found nothing to split out of this picture".into()),
                });
            }
            ctx.progress(0.9, "Splitting");
            // Each pixel to the smallest thing covering it.
            things.sort_by_key(|t| t.pixels);
            let mut owner: Vec<u32> = vec![u32::MAX; w * h];
            for (i, t) in things.iter().enumerate() {
                for y in t.bounds.y0..t.bounds.y1 {
                    let row = (y - frame.y0) as usize * w;
                    for x in t.bounds.x0..t.bounds.x1 {
                        let k = row + (x - frame.x0) as usize;
                        if owner[k] == u32::MAX && t.at(x, y) {
                            owner[k] = i as u32;
                        }
                    }
                }
            }
            let mut counts = vec![0usize; things.len()];
            let mut background = 0usize;
            for o in &owner {
                match counts.get_mut(*o as usize) {
                    Some(c) => *c += 1,
                    None => background += 1,
                }
            }
            // Largest lowest, the smallest on top (the order changes nothing that shows: every
            // pixel is on one layer).
            let mut keep: Vec<usize> = (0..things.len()).filter(|i| counts[*i] >= MIN_PIXELS).collect();
            keep.sort_by_key(|i| std::cmp::Reverse(counts[*i]));
            keep.truncate(MAX_LAYERS - 1);
            let kept: std::collections::HashSet<usize> = keep.iter().copied().collect();
            // A thing over the layer budget goes back to the background.
            let in_background = |o: u32| o == u32::MAX || !kept.contains(&(o as usize));
            let picture = photocraft_compose::render(doc, frame).to_rgba8().pixels;
            let cut = |pick: &dyn Fn(u32) -> bool| -> Vec<u8> {
                picture.as_chunks::<4>().0.iter().zip(&owner).flat_map(|(px, o)| if pick(*o) { *px } else { [0, 0, 0, 0] }).collect()
            };
            let mut numbered: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut layers: Vec<(String, Vec<u8>)> = Vec::new();
            if background > 0 || keep.len() < things.len() {
                layers.push(("Background".to_string(), cut(&|o| in_background(o))));
            }
            for &i in &keep {
                let term = title_case(&things[i].term);
                let n = numbered.entry(term.clone()).or_insert(0);
                *n += 1;
                layers.push((format!("{term} {n}"), cut(&|o| o == i as u32)));
            }
            let mut children = Vec::with_capacity(layers.len());
            let mut names = Vec::with_capacity(layers.len());
            for (k, (layer_name, px)) in layers.into_iter().enumerate() {
                let mut layer = Layer::raster(layer_name.clone(), doc.pixel_format());
                write_rgba8(&mut layer, &Rgba8::new(frame.width(), frame.height(), px).map_err(gen_err)?, frame)?;
                layer.label = LABELS[k % LABELS.len()];
                set_generative_info(
                    &mut layer,
                    &GenerativeInfo {
                        command: CLOWN_PASS.to_string(),
                        prompt: terms.join(", "),
                        rect: [frame.x0, frame.y0, frame.width() as i32, frame.height() as i32],
                        name: layer_name.clone(),
                        ..GenerativeInfo::default()
                    },
                );
                names.push(layer_name);
                // Photoshop's layer stack lists a group's children bottom first.
                children.push(layer);
            }
            let ids: Vec<u64> = children.iter().map(|l| l.id.0).collect();
            let group = doc.insert_above(*active, Layer::group(name.clone(), children));
            *active = Some(group);
            Ok((group, ids, names, started.elapsed().as_millis() as u64))
        },
        |(group, ids, names, ms)| json!({"group": group.0, "layers": ids, "names": names, "count": ids.len(), "ms": ms}),
    )
}
