//! `generate.splitLayers` (SAM 3.1 subjects peeled off nearest first over a repainted
//! background) against the fake worker, whose detector returns the rectangles it is given
//! (with `occlusion`, each minus the nearer ones still there) and whose fill paints its colour.

use photocraft_doc::LayerId;
use photocraft_genai::fake::FakeNative;
use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// A 64×48 document with one empty layer, talking to `url`.
fn session(url: &str) -> Session {
    session_of(url, 64, 48)
}

/// A `w`×`h` document with one empty layer, talking to `url`. The removal widens its hole by
/// 16 pixels at least, so a test whose subjects must keep their distance takes a larger canvas.
fn session_of(url: &str, w: u32, h: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    s
}

fn alpha_at(s: &Session, id: LayerId, x: i32, y: i32) -> f32 {
    pixel_at(s, id, x, y)[3]
}

/// RGBA (0..1) of one pixel of a layer.
fn pixel_at(s: &Session, id: LayerId, x: i32, y: i32) -> [f32; 4] {
    let d = s.active().unwrap();
    let px = d.doc.layer(id).unwrap().surface().unwrap().read_region(Rect::new(x, y, x + 1, y + 1));
    [px[0], px[1], px[2], px[3]]
}

fn layer_name(s: &Session, id: LayerId) -> String {
    s.active().unwrap().doc.layer(id).unwrap().name.clone()
}

#[test]
fn split_by_subjects_cuts_each_instance_out_over_a_repainted_background() {
    // The fake detector finds two people; the fill paints its colour.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let (before, steps) = {
        let d = s.active().unwrap();
        (d.doc.layers.len(), d.history.past_len())
    };
    let r = s.execute(SPLIT, json!({"seed": 3})).unwrap();
    assert_eq!(r["count"], 3, "the background and the two subjects: {r}");
    // ViTMatte (the fake's answer is opaque): each subject is its mask grown by the band.
    assert_eq!(r["edges"], "vitmatte", "{r}");
    assert_eq!(r["order"], "auto", "{r}");
    assert_eq!(r["incomplete"], json!([]), "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let d = s.active().unwrap();
    assert_eq!(d.doc.layers.len(), before + 3, "the fill's own layers are folded into the background");
    assert_eq!(d.history.past_len(), steps + 1, "one undo step");
    // Named by what they are, the nearer (its bottom edge lower) on top; the document has a
    // "Background" of its own, so the split's is told apart.
    assert_eq!(layer_name(&s, ids[0]), "Split background");
    assert_eq!(layer_name(&s, ids[1]), "Person 1");
    assert_eq!(layer_name(&s, ids[2]), "Person 2");
    assert_eq!(r["subjects"][0]["term"], "person", "{r}");
    assert_eq!(r["subjects"][1]["completed"], true, "{r}");
    // The background is whole and opaque (the removal's result is tone matched to the picture
    // at the hole's edge, so its colour is not the fake's); each subject is opaque on its
    // instance only.
    assert_eq!(alpha_at(&s, ids[0], 5, 5), 1.0);
    assert_eq!(alpha_at(&s, ids[0], 60, 45), 1.0);
    assert!(alpha_at(&s, ids[1], 5, 5) > 0.5, "the back subject (top-left quarter) is the lower layer");
    assert!(alpha_at(&s, ids[2], 60, 45) > 0.5, "the front subject (bottom-right quarter) is the top layer");
    assert_eq!(alpha_at(&s, ids[1], 60, 5), 0.0, "nothing outside the instances");
    assert_eq!(alpha_at(&s, ids[2], 60, 5), 0.0);
    let st = fake.state();
    assert!(st.runs.iter().any(|r| r.request["task"] == "segment" && r.request["prompt"] == "person"), "SAM asked for the people");
    let fills = st.runs.iter().filter(|r| r.request["task"] != "segment" && r.request["task"] != "text" && r.request["model"] != "vitmatte/refine").count();
    assert_eq!(fills, 2, "one fill behind each subject: {:?}", st.runs.iter().map(|r| r.request["task"].clone()).collect::<Vec<_>>());
    assert!(st.runs.iter().all(|r| r.request["task"] != "layers"), "not the layered model");
    // The removal names what it removes (an edit model told only "the object" changed nothing).
    assert!(
        st.runs.iter().any(|r| r.request["task"] != "segment" && r.request["prompt"].as_str().is_some_and(|p| p.contains("the person"))),
        "{:?}",
        st.runs.iter().map(|r| r.request["prompt"].clone()).collect::<Vec<_>>()
    );
    drop(st);
    assert!(s.undo());
    assert_eq!(s.active().unwrap().doc.layers.len(), before);
    fake.restore_scene();
    // `layers` caps the subjects: the largest kept.
    let r = s.execute(SPLIT, json!({"layers": 2})).unwrap();
    assert_eq!(r["count"], 2, "{r}");
    assert_eq!(layer_name(&s, LayerId(r["layers"][1].as_u64().unwrap())), "Person", "one of a kind takes no number");
}

#[test]
fn split_completes_what_a_nearer_subject_hid() {
    // Two people on a 256×192 canvas, the front one (nearest first in the fake) over the back
    // one's middle: front x 102..154, y 38..173; back x 13..243, y 19..96.
    let front = [0.4, 0.2, 0.6, 0.9];
    let back = [0.05, 0.1, 0.95, 0.5];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    // A blue strip on the back one's visible part, inside the band the front one's removal
    // repaints (x 82..102): the back one's own pixels, which its layer must keep.
    s.execute("select.rect", json!({"x": 82, "y": 19, "width": 20, "height": 77})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#2040c8"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1})).unwrap();
    assert_eq!(r["count"], 3, "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    // The front subject (bottom edge lower) is the top layer.
    assert_eq!(layer_name(&s, ids[2]), "Person 2");
    assert!(alpha_at(&s, ids[2], 128, 160) > 0.5, "the front one holds its lower body");
    assert!(alpha_at(&s, ids[1], 40, 50) > 0.5, "the back one holds its visible part");
    // Where the front one stood over the back one (x 102..154, y 38..96): the front layer has
    // the picture's pixels; the back layer is now whole there, cut from what the peel painted.
    let (x, y) = (128, 64);
    let front_px = pixel_at(&s, ids[2], x, y);
    assert!(front_px[3] > 0.5 && front_px[0] > front_px[1], "the front layer keeps the picture's pixels: {front_px:?}");
    let back_px = pixel_at(&s, ids[1], x, y);
    assert!(back_px[3] > 0.5, "the back subject is completed where the front one hid it: {back_px:?}");
    assert_eq!(alpha_at(&s, ids[1], 240, 180), 0.0, "and nothing more");
    // Where it was visible beside the front one, the back one keeps the picture's own pixels
    // (the blue strip), not what the front one's fill painted over its band.
    let kept = pixel_at(&s, ids[1], 90, 50);
    assert!(kept[3] > 0.5 && kept[2] > 0.7 && kept[0] < 0.3, "the back subject's visible pixels are its own: {kept:?}");
    assert_eq!(r["subjects"][0]["completed"], true, "{r}");
    assert_eq!(r["subjects"][1]["completed"], true, "{r}");
    assert_eq!(r["subjects"][0]["bounds"][0], 13, "the back one's bounds take in the completed part: {r}");
    // The detector was asked once to find the people (the two filled the layer budget), once
    // for anyone else in the picture before any fill, then again after each fill, for the
    // subject painted back and the people still behind, and once more on the finished
    // background (nobody left on it). A picture this small is not looked at tile by tile.
    assert_eq!(r["leftBehind"], json!([]), "{r}");
    let st = fake.state();
    let segments = st.runs.iter().filter(|r| r.request["task"] == "segment").count();
    assert_eq!(segments, 5, "found, who was there, checked after each fill, the background looked over: {segments}");
    assert_eq!((r["detector"]["requests"].as_u64(), r["detector"]["tiles"].as_u64(), r["detector"]["fog"].as_u64()), (Some(5), Some(0), Some(0)), "{r}");
    assert!(st.runs.iter().filter(|r| r.request["model"] == "vitmatte/refine").count() >= 3, "the completed subject was matted again");
}

#[test]
fn split_sweeps_nearest_first_and_finds_the_rest_again_after_each_removal() {
    // The same two people as above, as a sweep: the front one is found, cut and removed; the
    // back one is found again on the filled picture, whole (the fake returns each rectangle
    // minus the nearer ones still there); a third look finds nobody.
    let front = [0.4, 0.2, 0.6, 0.9];
    let back = [0.05, 0.1, 0.95, 0.5];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    s.execute("select.rect", json!({"x": 82, "y": 19, "width": 20, "height": 77})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#2040c8"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "sweep": true, "layers": 6})).unwrap();
    assert_eq!(r["count"], 3, "two people, then nobody: {r}");
    assert_eq!(r["incomplete"], json!([]), "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    assert_eq!((layer_name(&s, ids[1]).as_str(), layer_name(&s, ids[2]).as_str()), ("Person 1", "Person 2"), "the front one on top");
    assert!(alpha_at(&s, ids[2], 128, 160) > 0.5, "the front one holds its lower body");
    let back_px = pixel_at(&s, ids[1], 128, 64);
    assert!(back_px[3] > 0.5 && back_px[1] > 0.6, "the back one is whole where the front one stood, from the fill: {back_px:?}");
    let kept = pixel_at(&s, ids[1], 90, 50);
    assert!(kept[3] > 0.5 && kept[2] > 0.7 && kept[0] < 0.3, "its visible pixels are its own through the first removal: {kept:?}");
    assert_eq!(alpha_at(&s, ids[1], 240, 180), 0.0, "and nothing more");
    assert_eq!(r["subjects"][0]["bounds"][0], 13, "found again whole: {r}");
    let bg = pixel_at(&s, ids[0], 128, 64);
    assert!(bg[1] > 0.5 && bg[0] < 0.3, "the background is the fill: {bg:?}");
    let tries = r["tries"].as_array().unwrap();
    assert_eq!(tries.len(), 2, "{tries:?}");
    assert_eq!((tries[0]["behind"].as_str(), tries[1]["behind"].as_str()), (Some("Person 2"), Some("Person 1")), "{tries:?}");
    let st = fake.state();
    let segments: Vec<&str> = st.runs.iter().filter(|r| r.request["task"] == "segment").map(|r| r.request["prompt"].as_str().unwrap_or("")).collect();
    assert!(segments.len() >= 5, "found, checked, found again, checked, found nobody: {segments:?}");
    assert_eq!(st.runs.iter().filter(|r| r.request["model"] == "vitmatte/refine").count(), 2, "each subject matted once, at its turn");
}

#[test]
fn split_without_complete_runs_one_fill_and_keeps_the_visible_parts() {
    let front = [0.3, 0.2, 0.7, 0.9];
    let back = [0.1, 0.1, 0.9, 0.5];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "complete": false, "order": "size"})).unwrap();
    assert_eq!(r["count"], 3, "{r}");
    assert_eq!(r["order"], "size", "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    // Largest lowest: with only its visible part, the back (wide) one is the smaller of the two.
    assert!(alpha_at(&s, ids[2], 8, 12) > 0.5, "the smaller (back) one is the top layer");
    assert_eq!(alpha_at(&s, ids[2], 32, 16), 0.0, "and holds only its visible part");
    assert!(alpha_at(&s, ids[1], 32, 40) > 0.5, "the larger (front) one is the lower layer");
    assert_eq!(r["subjects"][0]["completed"], false, "{r}");
    let st = fake.state();
    let fills = st.runs.iter().filter(|r| r.request["task"] != "segment" && r.request["task"] != "text" && r.request["model"] != "vitmatte/refine").count();
    assert_eq!(fills, 2, "one fill for both, judged over two variations: {fills}");
}

#[test]
fn split_finds_the_things_the_vision_model_names_and_takes_a_list_of_terms() {
    // No prompt: the people, then the animals (none), then the things the model lists.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: Vec::new(),
        segments_for: vec![
            ("person".into(), vec![[0.0, 0.0, 0.5, 0.5]]),
            // 9 pixels to the person's 768: a speck against the person, a layer of its own term.
            ("cup".into(), vec![[0.6, 0.6, 0.65, 0.65]]),
            ("the lighthouse".into(), vec![[0.5, 0.0, 1.0, 0.5]]),
        ],
        enhanced_text: Some("a cup, the wooden table".into()),
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"layers": 4, "seed": 1})).unwrap();
    assert_eq!(r["count"], 3, "the person and the cup; no table was found: {r}");
    assert_eq!(r["foundBy"], "person, cup", "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let names: Vec<String> = ids.iter().map(|id| layer_name(&s, *id)).collect();
    assert_eq!(names, ["Split background", "Person", "Cup"], "the cup (its bottom edge lower) on top");
    let st = fake.state();
    let asked: Vec<String> =
        st.runs.iter().filter(|r| r.request["task"] == "segment").map(|r| r.request["prompt"].as_str().unwrap_or("").to_string()).collect();
    assert!(asked.iter().any(|p| p == "wooden table"), "the model's phrases, articles dropped: {asked:?}");
    assert!(asked.iter().any(|p| p == "animal"), "{asked:?}");
    let so_far = st.runs.len();
    drop(st);
    assert!(s.undo());
    fake.restore_scene();
    // A list of terms, each asked for by name, in the caller's words.
    let r = s.execute(SPLIT, json!({"layers": 4, "prompt": "the lighthouse, cup:1", "seed": 1})).unwrap();
    assert_eq!(r["foundBy"], "the lighthouse, cup", "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let names: Vec<String> = ids.iter().map(|id| layer_name(&s, *id)).collect();
    assert_eq!(names, ["Split background", "Lighthouse", "Cup"]);
    let st = fake.state();
    let runs = &st.runs[so_far..];
    assert!(runs.iter().any(|r| r.request["task"] == "segment" && r.request["prompt"] == "the lighthouse"));
    assert!(runs.iter().all(|r| r.request["task"] != "text"), "no vision-language listing when the caller named the terms");
}

#[test]
fn split_validates_its_parameters_and_has_no_similar() {
    let fake = FakeNative::start_with(photocraft_genai::fake::Options { occlusion: true, ..Default::default() }).unwrap();
    let mut s = session(&fake.url);
    // The picture in the fake's fill colour: a fill in another brightness over a white canvas
    // is rightly refused as a shape painted over the hole.
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    for p in [json!({"layers": 1}), json!({"layers": 21}), json!({"layers": 2.5}), json!({"seed": "x"}), json!({"order": "depth"}), json!({"group": "all"})] {
        assert!(matches!(s.execute(SPLIT, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
    }
    let r = s.execute(SPLIT, json!({"layers": 2, "prompt": "the lighthouse", "name": "Scene"})).unwrap();
    assert_eq!(r["foundBy"], "the lighthouse", "the prompt is what to cut out: {r}");
    let id = LayerId(r["layers"][0].as_u64().unwrap());
    assert_eq!(s.active().unwrap().doc.layer(id).unwrap().name, "Scene 1/2");
    assert_eq!(layer_name(&s, LayerId(r["layers"][1].as_u64().unwrap())), "Scene 2/2");
    let st = fake.state();
    assert!(st.runs.iter().any(|r| r.request["task"] == "segment" && r.request["prompt"] == "the lighthouse"));
    assert!(
        st.runs.iter().any(|r| r.request["prompt"].as_str().is_some_and(|p| p.contains("the lighthouse") && r.request["task"] != "segment")),
        "the fill removes the lighthouse by name"
    );
    drop(st);
    let e = s.execute(SIMILAR, json!({"layer": id.0})).unwrap_err().to_string();
    assert!(e.contains("split"), "{e}");
    let mut empty = Session::new();
    empty.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    assert!(!empty.is_enabled(SPLIT));
}

#[test]
fn split_takes_hair_from_the_matte_around_each_subject() {
    // The detector finds the top-left quarter; the matte keeps a little more (x < 38, y < 28):
    // the "hair" just past the mask is kept, nothing far from it.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.0, 0.0, 0.5, 0.5]],
        matte: Some([0.0, 0.0, 0.6, 0.6]),
        // Not the picture's own red: a fill in the subject's colours is the subject kept.
        color: [10, 200, 10, 255],
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"layers": 2, "seed": 1, "edges": "matte"})).unwrap();
    assert_eq!(r["edges"], "matte", "{r}");
    let subject = LayerId(r["layers"][1].as_u64().unwrap());
    assert_eq!(alpha_at(&s, subject, 5, 5), 1.0, "inside the mask");
    assert!(alpha_at(&s, subject, 34, 5) > 0.9, "the matte's fringe past the mask");
    assert_eq!(alpha_at(&s, subject, 60, 40), 0.0, "nothing far away");
    let st = fake.state();
    assert!(st.runs.iter().any(|r| r.request["model"] == DEFAULT_MATTE_TEMPLATE || r.request["template"] == DEFAULT_MATTE_TEMPLATE), "the matte ran");
    drop(st);
    // The other edge sources, by name, each from the plain picture again (a split over the last
    // one's layers has their seams in the ring, and a flat fill is then no fill).
    assert!(s.undo());
    let r = s.execute(SPLIT, json!({"layers": 2, "edges": "refine"})).unwrap();
    assert_eq!(r["edges"], "refine", "{r}");
    assert!(s.undo());
    let r = s.execute(SPLIT, json!({"layers": 2, "edges": "vitmatte"})).unwrap();
    assert_eq!(r["edges"], "vitmatte", "{r}");
    let subject = LayerId(r["layers"][1].as_u64().unwrap());
    assert_eq!(alpha_at(&s, subject, 5, 5), 1.0, "solid inside the mask");
    assert_eq!(alpha_at(&s, subject, 60, 40), 0.0, "nothing past the band");
    let st = fake.state();
    let run = st.runs.iter().rev().find(|r| r.request["model"] == "vitmatte/refine").expect("ViTMatte asked");
    assert!(run.request["mask"].is_object(), "with a trimap: {}", run.request);
    drop(st);
    for bad in ["soft", "hair", "bogus"] {
        assert!(matches!(s.execute(SPLIT, json!({"edges": bad})), Err(EngineError::BadParams { .. })), "{bad}");
    }
}

#[test]
fn split_fails_with_nothing_changed_when_every_fill_paints_the_subject_back() {
    // The fake's fill hands the picture back (echo), so the detector still finds the subject
    // in the hole after each try.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options { echo: true, occlusion: true, ..Default::default() }).unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#1e5aa0"})).unwrap();
    let (before, steps) = {
        let d = s.active().unwrap();
        (d.doc.layers.len(), d.history.past_len())
    };
    let e = s.execute(SPLIT, json!({"seed": 3})).unwrap_err().to_string();
    assert!(e.contains("could not clear"), "{e}");
    let d = s.active().unwrap();
    assert_eq!((d.doc.layers.len(), d.history.past_len()), (before, steps), "nothing changed");
    let fills = fake.state().runs.iter().filter(|r| r.request["task"] != "segment" && r.request["task"] != "matte" && r.request["task"] != "text").count();
    assert!(fills >= 5, "five fills tried: {fills}");
    // After the removal (twice, on a plain backdrop), the backdrop fill: the hole's box
    // flattened, the subject never named.
    let st = fake.state();
    let backdrop: Vec<_> = st.runs.iter().filter(|r| r.request["prompt"].as_str().is_some_and(|p| p.contains("with nobody there"))).collect();
    assert_eq!(backdrop.len(), 3, "three backdrop fills after the two removals");
    assert!(backdrop.iter().all(|r| !r.request["prompt"].as_str().unwrap_or("").contains("person")), "the subject is not named");
}

#[test]
fn the_prompt_is_a_list_of_terms() {
    use super::split_subjects::{Term, parse_terms};
    let t = |n: &str, m: Option<usize>| Term { name: n.to_string(), max: m };
    assert_eq!(parse_terms("the dog"), vec![t("the dog", None)]);
    assert_eq!(parse_terms("person, cup:2; table and the chair"), vec![t("person", None), t("cup", Some(2)), t("table", None), t("the chair", None)]);
    assert_eq!(parse_terms("Person, the person, , cup:0"), vec![t("Person", None), t("cup", None)], "duplicates and empties go");
}

#[test]
fn a_detection_grows_a_subject_only_where_it_joins_it() {
    use super::split_subjects::joined_to;
    // An 8×2 grid: the subject is the first pixel of row 0; the detection adds row 0 up to x 3
    // and a patch of its own at x 6..7.
    let seed: Vec<f32> = (0..16).map(|k| if k == 0 { 1.0 } else { 0.0 }).collect();
    let wanted: Vec<bool> = (0..16).map(|k| k < 8 && ((1..=3).contains(&k) || k >= 6)).collect();
    let joined = joined_to(&wanted, &seed, 8, 2);
    let got: Vec<usize> = joined.iter().enumerate().filter(|(_, j)| **j).map(|(k, _)| k).collect();
    assert_eq!(got, vec![1, 2, 3], "the run next to the subject joins; the patch does not");
}

#[test]
fn the_split_background_takes_the_plain_name_when_the_document_has_no_background_layer() {
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.0, 0.0, 0.5, 0.5]],
        occlusion: true,
        color: [10, 200, 10, 255],
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    let base = s.active().unwrap().doc.layers.iter().find(|l| l.name == "Background").map(|l| l.id).expect("the document's own Background");
    s.execute("layer.renameLayer", json!({"layer": base.0, "name": "Photo"})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"layers": 2, "seed": 1})).unwrap();
    assert_eq!(layer_name(&s, LayerId(r["layers"][0].as_u64().unwrap())), "Background");
}

#[test]
fn the_layer_budget_counts_the_subjects_kept_not_the_detections() {
    // "man" finds the person again: a duplicate, cleaned as the terms are searched. Counted raw
    // it would have taken the cup's place among two subjects (the café photo's seven people came
    // back as eleven detections and its table was never asked for; in the app 2026-11-03).
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: Vec::new(),
        segments_for: vec![
            ("person".into(), vec![[0.0, 0.0, 0.5, 0.5]]),
            ("man".into(), vec![[0.0, 0.0, 0.5, 0.5]]),
            ("cup".into(), vec![[0.6, 0.6, 0.8, 0.8]]),
        ],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"layers": 3, "prompt": "person, man, cup", "seed": 1})).unwrap();
    assert_eq!(r["count"], 3, "the person once, and the cup: {r}");
    assert_eq!(r["foundBy"], "person, cup", "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let names: Vec<String> = ids.iter().map(|id| layer_name(&s, *id)).collect();
    assert_eq!(names, ["Split background", "Person", "Cup"]);
    let st = fake.state();
    assert!(st.runs.iter().any(|r| r.request["task"] == "segment" && r.request["prompt"] == "cup"), "the cup was asked for");
}

#[test]
fn split_rejects_a_fill_that_paints_someone_new_into_the_hole() {
    // Two people apart on a 256×192 canvas, the right one nearer (its bottom edge lower). Every
    // fill paints a new person into the picture (`appears`), inside the right one's hole: it
    // continues nobody, so all five tries behind the right one are rejected and it stays in the
    // background; the left one's hole is clear of it and is filled at the first try.
    let front = [0.55, 0.1, 0.95, 0.95];
    let back = [0.05, 0.1, 0.45, 0.85];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        appears: vec![[0.65, 0.3, 0.85, 0.7]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1})).unwrap();
    assert_eq!(r["count"], 3, "{r}");
    assert_eq!(r["incomplete"], json!(["Person 2"]), "the right one could not be cleared: {r}");
    assert_eq!(r["refusedLayers"], json!([]), "no refused fill is kept unless asked for: {r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    assert_eq!(layer_name(&s, ids[2]), "Person 2");
    let kept = pixel_at(&s, ids[0], 192, 96);
    assert!(kept[0] > 0.7 && kept[1] < 0.3, "the right one stays in the background, untouched: {kept:?}");
    assert!(alpha_at(&s, ids[2], 192, 96) > 0.5, "and still gets its layer");
    let st = fake.state();
    let fills: Vec<String> = st
        .runs
        .iter()
        .filter(|r| r.request["task"] != "segment" && r.request["task"] != "text" && r.request["model"] != "vitmatte/refine")
        .map(|r| format!("{}:{}", r.request["task"], r.request["model"]))
        .collect();
    assert_eq!(fills.len(), 6, "five rejected tries behind the right one, one fill behind the left: {fills:?} {:?}", r["tries"]);
    // The record of every try: the removal, then the grey-box fills, each with its verdict,
    // then the classical patch fill (which copies the canvas around the hole, the very colour
    // the fake's "person" is, so the pixels refuse it as kept).
    let tries = r["tries"].as_array().unwrap();
    assert_eq!((tries.len(), r["fills"].as_u64()), (7, Some(7)), "{tries:?}");
    assert!(tries.iter().take(5).all(|t| t["behind"] == "Person 2" && t["verdict"].as_str().unwrap_or("").contains("was not there")), "{tries:?}");
    assert_eq!(
        tries.iter().map(|t| t["fill"].as_str().unwrap_or("")).collect::<Vec<_>>(),
        ["removal", "removal", "fill", "fill", "fill", "patch", "removal"],
        "the removal twice on a plain backdrop, then the grey box: {tries:?}"
    );
    assert!(tries.iter().all(|t| t["generations"].is_u64()), "{tries:?}");
    assert_eq!(r["generations"], 6, "{r}");
    assert!(tries[5]["verdict"].as_str().unwrap_or("").starts_with("kept what stood there"), "{tries:?}");
    assert_eq!(
        (tries[6]["behind"].as_str(), tries[6]["fill"].as_str(), tries[6]["verdict"].as_str()),
        (Some("Person 1"), Some("removal"), Some("accepted")),
        "{tries:?}"
    );
}

#[test]
fn split_refuses_a_fill_that_leaves_the_subject_in_place_though_the_detector_misses_it() {
    // The fake's fill hands the picture back (echo) and its detector forgets the subject all
    // the same (overlooked): the removal's try is refused by the pixels, the subject's
    // silhouette as it was. The grey-box try after it gets the hole flattened to grey in its
    // request and hands that back, which is not the subject, so it passes.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options { echo: true, occlusion: true, overlooked: true, ..Default::default() }).unwrap();
    let mut s = session(&fake.url);
    s.execute("edit.fill", json!({"contents": "color", "color": "#1e5aa0"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 3})).unwrap();
    let tries = r["tries"].as_array().unwrap();
    assert_eq!(tries.len(), 3, "{tries:?}");
    for t in &tries[..2] {
        assert_eq!(t["fill"], "removal", "the removal, and once more on a plain backdrop: {tries:?}");
        assert!(t["verdict"].as_str().unwrap_or("").starts_with("kept what stood there (100 %"), "{tries:?}");
    }
    assert_eq!((tries[2]["fill"].as_str(), tries[2]["verdict"].as_str()), (Some("fill"), Some("accepted")), "{tries:?}");
    // The detector was not asked after the refused tries: the pixels refused them.
    let st = fake.state();
    let after_first: Vec<&str> = st
        .runs
        .iter()
        .skip_while(|r| r.request["task"] != "fill" && r.request["task"] != "edit")
        .skip(1)
        .map(|r| r.request["task"].as_str().unwrap_or(""))
        .collect();
    assert!(matches!(after_first.first().copied(), Some("fill" | "edit")), "no detector call between the refused try and the next: {after_first:?}");
    assert!(matches!(after_first.get(1).copied(), Some("fill" | "edit")), "{after_first:?}");
}

#[test]
fn a_fill_kept_what_it_left_the_colour_it_was() {
    use super::split_subjects::kept_share;
    // A 16×16 picture, red; the region is its left half (two 8-px blocks).
    let (w, h) = (16usize, 16usize);
    let before: Vec<u8> = (0..w * h).flat_map(|_| [200u8, 40, 40, 255]).collect();
    let region: Vec<f32> = (0..w * h).map(|k| if k % w < 8 { 1.0 } else { 0.0 }).collect();
    assert_eq!(kept_share(&before, &before, &region, w, h), 1.0, "unchanged: all kept");
    let shifted: Vec<u8> = (0..w * h).flat_map(|_| [190u8, 50, 30, 255]).collect();
    assert_eq!(kept_share(&before, &shifted, &region, w, h), 1.0, "a few levels off (re-encoded): kept");
    let blue: Vec<u8> = (0..w * h).flat_map(|_| [30u8, 60, 200, 255]).collect();
    assert_eq!(kept_share(&before, &blue, &region, w, h), 0.0, "repainted: nothing kept");
    let half: Vec<u8> = (0..w * h).flat_map(|k| if k / w < 8 { [200u8, 40, 40, 255] } else { [30, 60, 200, 255] }).collect();
    assert_eq!(kept_share(&before, &half, &region, w, h), 0.5, "the top block kept, the bottom repainted");
    let nowhere = vec![0.0f32; w * h];
    assert_eq!(kept_share(&before, &before, &nowhere, w, h), 0.0, "no region: nothing to keep");
}

#[test]
fn one_of_each_listed_thing_comes_before_a_second_of_any() {
    // No people or animals; the model lists two things; three cups and one table are there. The
    // budget of three subjects takes a cup and the table before the second cup.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: Vec::new(),
        segments_for: vec![
            ("cup".into(), vec![[0.05, 0.7, 0.15, 0.85], [0.25, 0.7, 0.35, 0.85], [0.45, 0.7, 0.55, 0.85]]),
            ("wooden table".into(), vec![[0.0, 0.4, 1.0, 0.6]]),
        ],
        enhanced_text: Some("a cup, the wooden table".into()),
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 128, 96);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"layers": 4, "seed": 1})).unwrap();
    assert_eq!(r["count"], 4, "{r}");
    assert_eq!(r["foundBy"], "cup, wooden table", "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let mut names: Vec<String> = ids.iter().skip(1).map(|id| layer_name(&s, *id)).collect();
    names.sort();
    assert_eq!(names, ["Cup 1", "Cup 2", "Wooden table"], "two cups and the table, not three cups");
    let st = fake.state();
    let asked: Vec<usize> = st
        .runs
        .iter()
        .filter(|r| r.request["task"] == "segment" && r.request["prompt"] == "cup")
        .map(|r| r.request["params"]["maxDetections"].as_u64().unwrap_or(0) as usize)
        .collect();
    assert!(asked.len() >= 2 && asked[0] == 1, "one cup first, then more: {asked:?}");
}

#[test]
fn a_detection_is_the_fills_own_work_when_it_adds_more_than_a_completion() {
    use super::split_subjects::fills_own_work;
    // A row of 16 pixels: the cut subject held pixels 4–11 (the hole), the fill repainted
    // 0–13 (the hole widened); the subject behind holds pixels 0–1.
    let hole: Vec<f32> = (0..16).map(|k| if (4..12).contains(&k) { 1.0 } else { 0.0 }).collect();
    let grown: Vec<f32> = (0..16).map(|k| if k < 14 { 1.0 } else { 0.0 }).collect();
    let subject: Vec<f32> = (0..16).map(|k| if k < 2 { 1.0 } else { 0.0 }).collect();
    let behind = vec![("person".to_string(), subject, 2u64)];
    let det = |from: usize, to: usize| -> Vec<f32> { (0..16).map(|k| if k >= from && k < to { 1.0 } else { 0.0 }).collect() };
    let cut = ("person", &hole[..], 8u64);
    let own = |term: &str, d: &[f32], kept: f32| fills_own_work(term, d, &hole, &grown, cut, &behind, &|_| kept, &|_| false, 16, &|_| false);
    assert_eq!(own("person", &det(0, 6), 0.0), None, "the subject with twice its pixels added: completed");
    assert!(own("person", &det(0, 6), 0.5).unwrap().contains("painted back over"), "but looking as the cut one did: painted back");
    let table = vec![("table".to_string(), det(0, 2), 2u64)];
    let under = fills_own_work("table", &det(0, 8), &hole, &grown, cut, &table, &|_| 0.5, &|_| false, 16, &|_| false).unwrap();
    assert!(
        under.starts_with("the person still there under the table behind"),
        "the cut one's pixels under a thing's completion, named for what each is: {under}"
    );
    assert!(own("person", &det(0, 7), 0.0).unwrap().contains("more than a completion"), "more than twice: someone new standing over it");
    assert!(own("person", &det(12, 14), 0.0).unwrap().contains("was not there"), "in the repainted band, continuing nobody: painted in");
    assert!(own("person", &det(4, 12), 0.0).unwrap().contains("the person painted back"), "the cut one's own shape");
    assert!(own("cup", &det(4, 10), 0.0).unwrap().contains("a cup in the hole"), "another kind on the spot: painted in");
    // A thing seen apart from the one behind it that it joins (within two pixels) is that
    // thing's completion; two pixels further it is a new one.
    let cups = vec![("cup".to_string(), det(13, 16), 3u64)];
    let cup = |d: &[f32]| fills_own_work("cup", d, &hole, &grown, cut, &cups, &|_| 0.0, &|_| false, 16, &|_| false);
    assert_eq!(cup(&det(4, 12)), None, "the table's stretch across the hole, found on its own");
    assert!(cup(&det(4, 9)).unwrap().contains("a cup in the hole"), "apart from it: painted in");
    assert_eq!(own("person", &det(14, 16), 0.0), None, "outside the repainted area: was there before");
    // A thing is judged on its own spot: a cup in the widened band only was there before (a
    // saucer beside the cup); a person there is painted in.
    assert_eq!(own("cup", &det(12, 14), 0.0), None, "a cup beside the hole");
    assert_eq!(own("cup", &det(0, 6), 0.0), None, "a cup mostly beside the hole");
    assert!(own("person", &det(12, 14), 0.0).is_some(), "a person in the widened band");
    // A part of the one behind, seen on its own (less than the 60 % that continues them), is
    // that person; so is their completed part joined to them (within two pixels), by at most
    // twice their pixels, unless it looks like what was cut; apart from them, or more than that,
    // someone is painted in.
    assert_eq!(own("person", &det(0, 1), 0.0), None, "a part of the one behind");
    assert_eq!(own("person", &det(1, 3), 0.0), None, "half on the one behind");
    assert_eq!(own("person", &det(1, 4), 0.0), None, "their completed part, joined to them");
    assert_eq!(own("person", &det(2, 6), 0.0), None, "joined to them, twice their pixels");
    assert!(own("person", &det(2, 6), 0.5).unwrap().contains("painted back over"), "but looking as the cut one did");
    assert!(own("person", &det(2, 7), 0.0).unwrap().contains("was not there"), "more than twice their pixels: painted in");
    assert!(own("person", &det(5, 8), 0.0).unwrap().contains("was not there"), "apart from them: painted in");
    // A figure with no more detail than fog is no figure; a thing is judged as before.
    let foggy = |term: &str, d: &[f32]| fills_own_work(term, d, &hole, &grown, cut, &behind, &|_| 0.0, &|_| false, 16, &|_| true);
    assert_eq!(foggy("person", &det(12, 14)), None, "fog where someone new would be");
    assert!(foggy("person", &det(4, 12)).unwrap().contains("the person painted back"), "the cut one's own shape, foggy or not");
    assert!(foggy("cup", &det(4, 10)).unwrap().contains("a cup in the hole"), "a thing is not fog");
}

#[test]
fn a_blank_fill_and_a_smear_are_no_fill() {
    use super::split_subjects::blank_or_ghost;
    // A 64×64 picture of 4-px stripes (texture); the hole is the middle 24×24, the ring the
    // 12 px around it.
    let (w, h) = (64usize, 64usize);
    let stripes = |k: usize| -> [u8; 4] { if (k / w / 4).is_multiple_of(2) { [60u8, 60, 60, 255] } else { [180, 180, 180, 255] } };
    let before: Vec<u8> = (0..w * h).flat_map(stripes).collect();
    let in_hole = |k: usize| (20..44).contains(&(k % w)) && (20..44).contains(&(k / w));
    let in_ring = |k: usize| !in_hole(k) && (8..56).contains(&(k % w)) && (8..56).contains(&(k / w));
    let hole: Vec<f32> = (0..w * h).map(|k| if in_hole(k) { 1.0 } else { 0.0 }).collect();
    let ring: Vec<f32> = (0..w * h).map(|k| if in_ring(k) { 1.0 } else { 0.0 }).collect();
    let with = |paint: &dyn Fn(usize) -> [u8; 4]| -> Vec<u8> { (0..w * h).flat_map(|k| if in_hole(k) { paint(k) } else { stripes(k) }).collect() };
    let blank = with(&|_| [128u8, 128, 128, 255]);
    assert!(
        blank_or_ghost(&before, &blank, &hole, &ring, w, h).is_some_and(|v| v.starts_with("left the hole blank")),
        "a flat grey where the picture has stripes"
    );
    let continued = with(&stripes);
    assert_eq!(blank_or_ghost(&before, &continued, &hole, &ring, w, h), None, "the stripes continued: a fill");
    let dark = with(&|k| if (k / w).is_multiple_of(2) { [10u8, 10, 10, 255] } else { [40, 40, 40, 255] });
    assert!(
        blank_or_ghost(&before, &dark, &hole, &ring, w, h).is_some_and(|v| v.starts_with("painted a dark shape")),
        "a dark smear: textured, but not the picture's brightness"
    );
    // A plain backdrop: its fill is rightly plain, but not another brightness altogether.
    let plain: Vec<u8> = (0..w * h).flat_map(|_| [120u8, 120, 120, 255]).collect();
    assert_eq!(blank_or_ghost(&plain, &blank, &hole, &ring, w, h), None, "plain around the hole: a flat fill is right");
    let white = with(&|_| [250u8, 250, 250, 255]);
    assert!(
        blank_or_ghost(&plain, &white, &hole, &ring, w, h).is_some_and(|v| v.starts_with("painted a light shape over the hole")),
        "white on a grey backdrop"
    );
    // A bigger hole (64×64 in a 128×128 picture of the same stripes) with its top 24 rows
    // continued and the rest flat: the hole's averages pass, the cells do not.
    let (w, h) = (128usize, 128usize);
    let stripes = |k: usize| -> [u8; 4] { if (k / w / 4).is_multiple_of(2) { [60u8, 60, 60, 255] } else { [180, 180, 180, 255] } };
    let before: Vec<u8> = (0..w * h).flat_map(stripes).collect();
    let in_hole = |k: usize| (32..96).contains(&(k % w)) && (32..96).contains(&(k / w));
    let in_ring = |k: usize| !in_hole(k) && (20..108).contains(&(k % w)) && (20..108).contains(&(k / w));
    let hole: Vec<f32> = (0..w * h).map(|k| if in_hole(k) { 1.0 } else { 0.0 }).collect();
    let ring: Vec<f32> = (0..w * h).map(|k| if in_ring(k) { 1.0 } else { 0.0 }).collect();
    let part: Vec<u8> = (0..w * h).flat_map(|k| if in_hole(k) && k / w >= 56 { [128u8, 128, 128, 255] } else { stripes(k) }).collect();
    let why = blank_or_ghost(&before, &part, &hole, &ring, w, h).expect("the lower part of the hole is blank");
    assert!(why.starts_with("left 6") && why.contains("% of the hole blank"), "{why}");
    let whole: Vec<u8> = (0..w * h).flat_map(stripes).collect();
    assert_eq!(blank_or_ghost(&before, &whole, &hole, &ring, w, h), None, "continued throughout");
}

#[test]
fn a_big_hole_gets_the_removal_then_strips_then_the_box_and_the_patch() {
    // One person over 540 px tall on a 900×600 canvas of 24-px stripes (texture): the fake's
    // fills paint a flat colour, which is no fill here, so every try is refused and shows the
    // plan: the removal first, the grey box in two strips twice, the box whole, then the
    // classical patch fill, which continues the stripes and is accepted.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.3, 0.04, 0.7, 0.96]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 900, 600);
    for row in 0..25u32 {
        let color = if row.is_multiple_of(2) { "#404040" } else { "#b4b4b4" };
        s.execute("select.rect", json!({"x": 0, "y": row * 24, "width": 900, "height": 24})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": color})).unwrap();
    }
    s.execute("select.rect", json!({"x": 270, "y": 24, "width": 360, "height": 552})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#2040c8"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "layers": 2})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(2), Some(0)), "{r}");
    let tries = r["tries"].as_array().unwrap();
    assert_eq!(tries.iter().map(|t| t["fill"].as_str().unwrap_or("")).collect::<Vec<_>>(), ["removal", "strips", "strips", "fill", "patch"], "{tries:?}");
    assert_eq!(tries.iter().map(|t| t["generations"].as_u64().unwrap_or(99)).collect::<Vec<_>>(), [1, 2, 2, 1, 0], "two strips a try: {tries:?}");
    assert!(tries[..4].iter().all(|t| t["verdict"].as_str().unwrap_or("").starts_with("left the hole blank")), "{tries:?}");
    assert_eq!(tries[4]["verdict"], "accepted", "{tries:?}");
    assert_eq!((r["fills"].as_u64(), r["generations"].as_u64()), (Some(5), Some(6)), "{r}");
    let st = fake.state();
    let tasks: Vec<&str> = st.runs.iter().map(|r| r.request["task"].as_str().unwrap_or("")).filter(|t| *t == "fill" || *t == "edit").collect();
    assert_eq!(tasks.len(), 6, "the removal, two strip pairs and the box: {tasks:?}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let bg = pixel_at(&s, ids[0], 450, 300);
    assert!(bg[2] < 0.6 && (bg[1] - bg[0]).abs() < 0.1, "the background in the hole is the patch, neither the blue nor the fake's green: {bg:?}");
    assert!(alpha_at(&s, ids[1], 450, 300) > 0.5 && pixel_at(&s, ids[1], 450, 300)[2] > 0.7, "the person's layer holds their own pixels");
}

#[test]
fn a_small_person_a_fill_paints_in_is_refused_however_small() {
    // Two people apart on a 256×192 canvas; every fill paints a 15×19 px figure into the right
    // one's hole (`appears`): under 1 % of the hole, which the frame-wide check lets through,
    // but a person all the same.
    let front = [0.55, 0.1, 0.95, 0.95];
    let back = [0.05, 0.1, 0.45, 0.85];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        appears: vec![[0.7, 0.4, 0.76, 0.5]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1})).unwrap();
    assert_eq!(r["incomplete"], json!(["Person 2"]), "the right one could not be cleared: {r}");
    let tries = r["tries"].as_array().unwrap();
    assert!(
        tries.iter().take(5).all(|t| t["behind"] == "Person 2" && t["verdict"].as_str().unwrap_or("").starts_with("a person painted into the hole (")),
        "{tries:?}"
    );
    assert_eq!(tries.last().map(|t| (t["behind"].as_str(), t["verdict"].as_str())), Some((Some("Person 1"), Some("accepted"))), "{tries:?}");
    // The detector asked for people down to a floor, not the two or three largest.
    let st = fake.state();
    let asked: Vec<u64> = st
        .runs
        .iter()
        .filter(|r| r.request["task"] == "segment" && r.request["prompt"] == "person")
        .map(|r| r.request["params"]["maxDetections"].as_u64().unwrap_or(0))
        .collect();
    assert!(asked.iter().skip(1).all(|n| *n >= 16), "{asked:?}");
}

#[test]
fn the_classical_patch_fill_is_the_last_try_when_every_fill_leaves_the_hole_blank() {
    // The backdrop is striped; the fake's fills paint a flat colour over the hole, which is no
    // fill where the picture has texture: five refused, then the patch fill continues the
    // stripes from around the hole and the detector finds nobody.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options { color: [10, 200, 10, 255], occlusion: true, ..Default::default() }).unwrap();
    let mut s = session(&fake.url);
    for row in 0..12u32 {
        let color = if row.is_multiple_of(2) { "#404040" } else { "#b4b4b4" };
        s.execute("select.rect", json!({"x": 0, "y": row * 4, "width": 64, "height": 4})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": color})).unwrap();
    }
    // The person: a blue block where the fake's detector finds them (16, 12)–(48, 36).
    s.execute("select.rect", json!({"x": 16, "y": 12, "width": 32, "height": 24})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#2040c8"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 2, "layers": 2})).unwrap();
    assert_eq!(r["incomplete"], json!([]), "{r}");
    let tries = r["tries"].as_array().unwrap();
    assert_eq!(tries.len(), 6, "five fills and the patch: {tries:?}");
    assert!(tries[..5].iter().all(|t| t["verdict"].as_str().unwrap_or("").starts_with("left the hole blank")), "{tries:?}");
    assert_eq!((tries[5]["fill"].as_str(), tries[5]["verdict"].as_str()), (Some("patch"), Some("accepted")), "{tries:?}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let bg = pixel_at(&s, ids[0], 32, 24);
    assert!(bg[2] < 0.6 && (bg[1] - bg[0]).abs() < 0.1, "the background in the hole is neither the blue nor the fake's green: {bg:?}");
    assert_eq!(fake.state().runs.iter().filter(|r| r.request["task"] == "fill" || r.request["task"] == "edit").count(), 5, "the patch fill asks no model");
}

#[test]
fn a_sweep_cuts_everyone_the_first_pass_saw_even_when_the_detector_loses_them() {
    // Two people on a 256×192 canvas: a thin one (15 px wide) right beside a big one, inside
    // the band the big one's removal repaints. The fake's detector loses whoever that mask
    // covers, so after the first removal it finds nobody; the thin one is on the sweep's list
    // from the first pass and is cut all the same, by their first mask, with their own pixels
    // (kept through the neighbour's fill). Before, a later pass found nobody and the sweep
    // ended with the thin one in the background (the striped man, the waiter and the man at
    // the frame's edge, 3 of 3 café seeds, measured in the app 2026-11-04).
    let front = [0.3, 0.1, 0.6, 0.9];
    let back = [0.6, 0.3, 0.66, 0.8];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    // The thin one (x 154..168, y 58..153), blue: their own pixels.
    s.execute("select.rect", json!({"x": 154, "y": 58, "width": 15, "height": 96})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#2040c8"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "sweep": true, "layers": 6})).unwrap();
    assert_eq!(r["count"], 3, "both people: {r}");
    assert_eq!(r["incomplete"], json!([]), "{r}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    assert_eq!((layer_name(&s, ids[1]).as_str(), layer_name(&s, ids[2]).as_str()), ("Person 1", "Person 2"), "the big one on top");
    assert!(alpha_at(&s, ids[2], 115, 100) > 0.5, "the big one holds their body");
    let thin = pixel_at(&s, ids[1], 160, 100);
    assert!(thin[3] > 0.5 && thin[2] > 0.7 && thin[0] < 0.3, "the thin one's layer holds their own pixels: {thin:?}");
    assert_eq!(alpha_at(&s, ids[1], 115, 100), 0.0, "and nothing of the big one");
    let bg = pixel_at(&s, ids[0], 160, 100);
    assert!(bg[1] > 0.5 && bg[2] < 0.3, "the background is the fill where the thin one stood: {bg:?}");
    assert_eq!(r["tries"].as_array().map(|t| t.iter().filter(|x| x["verdict"] == "accepted").count()), Some(2), "{r}");
    // The detector was asked for people on the filled picture and found nobody, twice.
    let st = fake.state();
    let finds = st.runs.iter().filter(|r| r.request["task"] == "segment" && r.request["params"]["maxDetections"].as_u64().is_some_and(|n| n < 16)).count();
    assert!(finds >= 3, "the first pass, then a look for the rest after each removal: {finds}");
}

#[test]
fn a_find_is_a_listed_subject_seen_again_when_either_holds_half_of_the_other() {
    use super::split_subjects::same_subject;
    let (w, h) = (16usize, 16usize);
    let rect = |x0: usize, y0: usize, x1: usize, y1: usize| -> Vec<f32> {
        (0..w * h).map(|k| if (x0..x1).contains(&(k % w)) && (y0..y1).contains(&(k / w)) { 1.0 } else { 0.0 }).collect()
    };
    let strip = rect(4, 0, 12, 2);
    let whole = rect(4, 0, 12, 16);
    assert!(same_subject(&whole, 128, &strip, 16), "the whole of someone against their visible part");
    assert!(same_subject(&rect(4, 0, 12, 1), 8, &whole, 128), "a part of them against the whole");
    assert!(!same_subject(&rect(12, 0, 16, 16), 64, &whole, 128), "a neighbour touching them");
    assert!(!same_subject(&rect(0, 0, 16, 16), 256, &rect(0, 0, 4, 1), 0), "nothing against anything");
}

#[test]
fn someone_a_fill_painted_stands_mostly_where_it_painted() {
    use super::split_subjects::someone_new;
    // A row of 16 pixels: the fill was laid over 2–13 (the hole 4–11 widened and feathered);
    // the subject behind holds pixels 0–1.
    let det = |from: usize, to: usize| -> Vec<f32> { (0..16).map(|k| if k >= from && k < to { 1.0 } else { 0.0 }).collect() };
    let touched = det(2, 14);
    let hole = det(4, 12);
    let behind = vec![("person".to_string(), det(0, 2), 2u64)];
    let cut = ("person", &hole[..], 8u64);
    let bounds = |from: i32, to: i32| Rect::new(from, 0, to, 1);
    let new = |d: &[f32], b: Rect| someone_new("person", d, b, &touched, cut, &behind, &[], &|_| false, 16);
    assert_eq!(new(&det(5, 8), bounds(5, 8)), Some("a person painted into the hole (3×1 px)".to_string()), "a small figure in the hole");
    assert_eq!(new(&det(4, 12), bounds(4, 12)), Some("the person painted back".to_string()), "the cut one's own shape");
    assert_eq!(new(&det(0, 6), bounds(0, 6)), None, "the one behind completed by twice their pixels");
    assert!(new(&det(0, 7), bounds(0, 7)).is_some(), "more than that: someone painted over them");
    assert_eq!(new(&det(12, 16), bounds(12, 16)), None, "half in the feathered edge: there before");
    assert_eq!(new(&det(14, 16), bounds(14, 16)), None, "outside: there before");
    // Their completed part seen on its own, joined to them (within two pixels) and at most twice
    // their pixels, is theirs; apart from them, or more than that, it is someone painted in.
    assert_eq!(new(&det(2, 5), bounds(2, 5)), None, "joined to the one behind: their completed part");
    assert_eq!(new(&det(2, 7), bounds(2, 7)).as_deref(), Some("a person painted into the hole (5×1 px)"), "more than twice their pixels");
    assert_eq!(new(&det(4, 7), bounds(4, 7)).as_deref(), Some("a person painted into the hole (3×1 px)"), "apart from them");
    // Someone the picture had before any fill (apart from the subjects) is nobody's work.
    let already = vec![(det(5, 8), 3u64)];
    assert_eq!(someone_new("person", &det(5, 8), bounds(5, 8), &touched, cut, &behind, &already, &|_| false, 16), None, "there before the fill");
    assert_eq!(
        someone_new("person", &det(7, 10), bounds(7, 10), &touched, cut, &behind, &already, &|_| false, 16).as_deref(),
        Some("a person painted into the hole (3×1 px)"),
        "mostly not"
    );
    // Someone partly there before (one pixel of them in the feathered edge) and completed by at
    // most twice that is nobody's work either; more than that is someone painted over them,
    // unless they stood against the cut subject (within two pixels), whose removal shows the
    // rest of them, however much.
    let partly = vec![(det(13, 14), 1u64)];
    assert_eq!(someone_new("person", &det(11, 14), bounds(11, 14), &touched, cut, &behind, &partly, &|_| false, 16), None, "completed by twice their pixels");
    assert_eq!(
        someone_new("person", &det(10, 14), bounds(10, 14), &touched, cut, &behind, &partly, &|_| false, 16),
        None,
        "more than that, against the cut one: what it hid of them"
    );
    let apart = ("person", &det(4, 10)[..], 6u64);
    assert_eq!(someone_new("person", &det(11, 14), bounds(11, 14), &touched, apart, &behind, &partly, &|_| false, 16), None, "completed by twice their pixels");
    assert_eq!(
        someone_new("person", &det(10, 14), bounds(10, 14), &touched, apart, &behind, &partly, &|_| false, 16).as_deref(),
        Some("a person painted into the hole (4×1 px)"),
        "more than that, apart from the cut one"
    );
}

#[test]
fn a_figure_with_no_more_detail_than_fog_is_no_figure() {
    use super::split_subjects::{no_more_than_fog, picture_texture};
    // A 256×256 picture of 4-px stripes; a fill left a flat 64×64 patch at (96, 96).
    let (w, h) = (256usize, 256usize);
    let stripes = |k: usize| -> [u8; 4] { if (k / w / 4).is_multiple_of(2) { [60u8, 60, 60, 255] } else { [180, 180, 180, 255] } };
    let picture: Vec<u8> = (0..w * h).flat_map(stripes).collect();
    let flat = |k: usize| (96..160).contains(&(k % w)) && (96..160).contains(&(k / w));
    let filled: Vec<u8> = (0..w * h).flat_map(|k| if flat(k) { [120u8, 120, 120, 255] } else { stripes(k) }).collect();
    let tex = picture_texture(&picture, w, h);
    assert!(tex > 0.01, "stripes have texture: {tex}");
    let region = |from: usize, to: usize| -> Vec<f32> {
        (0..w * h).map(|k| if (from..to).contains(&(k % w)) && (from..to).contains(&(k / w)) { 1.0 } else { 0.0 }).collect()
    };
    assert!(no_more_than_fog(&filled, &region(100, 156), w, h, tex), "a detection on the flat patch is fog");
    assert!(!no_more_than_fog(&filled, &region(16, 80), w, h, tex), "one on the stripes has the picture's detail");
    assert!(!no_more_than_fog(&filled, &region(100, 156), w, h, 0.0), "a plain picture says nothing");
    // A plain picture's texture is under the plain threshold, so nothing on it is fog.
    let plain: Vec<u8> = vec![200u8; w * h * 4];
    assert!(!no_more_than_fog(&plain, &region(100, 156), w, h, picture_texture(&plain, w, h)));
}

#[test]
fn a_mask_over_the_whole_tile_is_no_figure() {
    // The scene of the tile test, but the detector shown a tile hands back a mask over the whole
    // tile (`appears_in_crops` over all of it): no figure, the fill is accepted, and the result
    // counts the detector's requests and tiles.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.3, 0.04, 0.7, 0.96]],
        appears_in_crops: vec![[0.0, 0.0, 1.0, 1.0]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 900, 600);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "layers": 2})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(2), Some(0)), "{r}");
    assert!(r["tries"].as_array().unwrap().iter().all(|t| t["verdict"] == "accepted"), "{r}");
    // On a plain backdrop the look before any fill and the one at the finished background are
    // not tiled; the fill's check is.
    let d = &r["detector"];
    assert!(d["requests"].as_u64().unwrap() > d["tiles"].as_u64().unwrap() && d["tiles"].as_u64().unwrap() >= 2, "whole-frame and tile requests: {d}");
    assert_eq!(r["already"], json!([]), "nobody apart from the subject: {r}");
    assert_eq!(d["fog"], 0, "a plain picture has no fog: {d}");
    assert!(d["ms"].is_u64(), "{d}");
}

#[test]
fn a_figure_the_detector_sees_only_in_a_tile_of_the_picture_refuses_the_fill() {
    // One person 360×552 px on a plain 900×600 canvas. Every fill paints a small figure into the
    // hole that the detector finds only when shown the part of the picture it is in
    // (`appears_in_crops`: at the detector's size, a figure a few pixels tall in the whole frame
    // is a third of that), so every try is refused and the split fails with the verdicts.
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![[0.3, 0.04, 0.7, 0.96]],
        appears_in_crops: vec![[0.4, 0.4, 0.45, 0.47]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 900, 600);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let err = s.execute(SPLIT, json!({"seed": 1, "layers": 2})).unwrap_err().to_string();
    assert!(err.contains("a person painted into the hole (32×42 px)") && err.contains("nothing was changed"), "{err}");
    // Every model fill; the classical patch copies the canvas around the hole, the colour the
    // fake's person is, and is refused as kept before the detector is asked.
    assert_eq!(err.matches("painted into the hole").count(), 5, "{err}");
    assert!(err.contains("patch kept what stood there"), "{err}");
    // The whole frame never showed the figure: the detector was shown a part of the picture.
    let st = fake.state();
    let sizes: Vec<(u32, u32)> =
        st.runs.iter().filter(|r| r.request["task"] == "segment").filter_map(|r| r.run.image.as_ref().map(|b| (b.width, b.height))).collect();
    assert!(sizes.contains(&(900, 600)) && sizes.contains(&(640, 600)), "the whole frame and a tile: {sizes:?}");
}

#[test]
fn someone_in_the_picture_before_the_fill_is_neither_the_fills_work_nor_left_behind() {
    // Two people apart on a 256×192 canvas and, beside the right one, a figure 8×14 px: a speck,
    // so no layer of its own, but a person the detector finds within the fill's feathered edge
    // after the right one is taken out. It was there before any fill: the fill is accepted, and
    // the finished background is not said to have anyone left on it.
    let front = [0.55, 0.1, 0.95, 0.95];
    let back = [0.05, 0.1, 0.45, 0.85];
    let speck = [0.5, 0.45, 0.53, 0.52];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back, speck],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(3), Some(0)), "{r}");
    assert_eq!(r["leftBehind"], json!([]), "{r}");
    assert_eq!(r["already"].as_array().map(Vec::len), Some(1), "the figure, found before any fill: {r}");
    let tries = r["tries"].as_array().unwrap();
    assert!(tries.iter().all(|t| t["verdict"] == "accepted"), "{tries:?}");
}

#[test]
fn an_invented_thing_is_taken_as_the_last_resort_when_every_try_is_refused() {
    // A person and a table apart on a plain 256×192 canvas. Every fill behind the person paints
    // a second table into the hole, away from the real one (`appears_for`): refused every time,
    // but a table is the one invention a hole may fall back on, so the first try is taken after
    // all and the person leaves the background; the table's own hole is clear.
    let person = [0.55, 0.1, 0.95, 0.95];
    let table = [0.05, 0.6, 0.4, 0.9];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments_for: vec![("person".to_string(), vec![person]), ("table".to_string(), vec![table])],
        appears_for: vec![("table".to_string(), vec![[0.62, 0.3, 0.8, 0.5]])],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "prompt": "person, table"})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(3), Some(0)), "{r}");
    let tries = r["tries"].as_array().unwrap();
    assert_eq!(
        (tries[0]["behind"].as_str(), tries[0]["fill"].as_str(), tries[0]["verdict"].as_str()),
        (Some("Person"), Some("removal"), Some("accepted as the last resort (a table in the hole that was not there)")),
        "{tries:?}"
    );
    assert!(tries[1..5].iter().all(|t| t["behind"] == "Person" && t["verdict"] == "a table in the hole that was not there"), "{tries:?}");
    // The classical patch copies the canvas around the hole, the colour the fake's person is:
    // refused as kept, before the detector is asked.
    assert!(tries[5]["fill"] == "patch" && tries[5]["verdict"].as_str().unwrap_or("").starts_with("kept what stood there"), "{tries:?}");
    assert_eq!((tries.len(), tries[6]["behind"].as_str(), tries[6]["verdict"].as_str()), (7, Some("Table"), Some("accepted")), "{tries:?}");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let bg = pixel_at(&s, ids[0], 192, 96);
    assert!(bg[1] > 0.6 && bg[0] < 0.2, "the first try's fill stands in the background where the person was: {bg:?}");
}

#[test]
fn a_shape_over_part_of_the_hole_is_a_ghost() {
    use super::split_subjects::blank_or_ghost;
    // A 256×256 picture of 4-px stripes; the hole is the middle 192×192, the ring 12 px around
    // it. The fill continues the stripes, but for a dark textured blob 100×68 px in the hole: a
    // sixth of the hole, too little for the share rule, but a cluster of six cells all the same
    // (the cells are 32 px, laid from 12 px outside the hole).
    let (w, h) = (256usize, 256usize);
    let stripes = |k: usize| -> [u8; 4] { if (k / w / 4).is_multiple_of(2) { [60u8, 60, 60, 255] } else { [180, 180, 180, 255] } };
    let before: Vec<u8> = (0..w * h).flat_map(stripes).collect();
    let in_hole = |k: usize| (32..224).contains(&(k % w)) && (32..224).contains(&(k / w));
    let in_ring = |k: usize| !in_hole(k) && (20..236).contains(&(k % w)) && (20..236).contains(&(k / w));
    let hole: Vec<f32> = (0..w * h).map(|k| if in_hole(k) { 1.0 } else { 0.0 }).collect();
    let ring: Vec<f32> = (0..w * h).map(|k| if in_ring(k) { 1.0 } else { 0.0 }).collect();
    let dark = |k: usize| -> [u8; 4] { if (k / w).is_multiple_of(2) { [10u8, 10, 10, 255] } else { [40, 40, 40, 255] } };
    let blob = |k: usize, (x0, y0): (usize, usize), (bw, bh): (usize, usize)| (x0..x0 + bw).contains(&(k % w)) && (y0..y0 + bh).contains(&(k / w));
    let with_blob: Vec<u8> = (0..w * h).flat_map(|k| if blob(k, (50, 50), (100, 68)) { dark(k) } else { stripes(k) }).collect();
    let why = blank_or_ghost(&before, &with_blob, &hole, &ring, w, h).expect("a dark shape over part of the hole");
    assert!(why.starts_with("painted a dark shape over part of the hole (about 96×64 px, -") && why.ends_with("% in brightness)"), "{why}");
    // Two small blobs apart (a cell or two each) are no cluster, nor is one on its own.
    let two: Vec<u8> = (0..w * h).flat_map(|k| if blob(k, (60, 60), (40, 40)) || blob(k, (150, 150), (40, 40)) { dark(k) } else { stripes(k) }).collect();
    assert_eq!(blank_or_ghost(&before, &two, &hole, &ring, w, h), None);
    let one: Vec<u8> = (0..w * h).flat_map(|k| if blob(k, (100, 100), (40, 40)) { dark(k) } else { stripes(k) }).collect();
    assert_eq!(blank_or_ghost(&before, &one, &hole, &ring, w, h), None);
}

#[test]
fn refused_fills_are_kept_as_hidden_layers_on_request() {
    // The scene of `split_rejects_a_fill_that_paints_someone_new_into_the_hole`, with
    // `keepTries`: the six refused fills behind the right one come back as hidden layers under
    // the background, each named for its subject, try and verdict; the layer count is unchanged.
    let front = [0.55, 0.1, 0.95, 0.95];
    let back = [0.05, 0.1, 0.45, 0.85];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        appears: vec![[0.65, 0.3, 0.85, 0.7]],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "keepTries": true})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(3), Some(1)), "{r}");
    let refused: Vec<LayerId> = r["refusedLayers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    assert_eq!(refused.len(), 6, "{r}");
    let d = s.active().unwrap();
    for (n, id) in refused.iter().enumerate() {
        let layer = d.doc.layer(*id).unwrap();
        assert!(!layer.visible, "{}", layer.name);
        assert!(layer.name.starts_with("Person 2: ") && layer.name.contains(" refused ("), "{}", layer.name);
        if n < 5 {
            assert!(layer.name.ends_with("(a person in the hole that was not there)"), "{}", layer.name);
        }
    }
    // Under the background: the first layer the split made is a refused fill, the background above them.
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    let order: Vec<LayerId> = d.doc.layers.iter().map(|l| l.id).collect();
    let at = |id: LayerId| order.iter().position(|o| *o == id).unwrap();
    assert!(refused.iter().all(|r| at(*r) < at(ids[0])), "refused fills below the background: {order:?}");
    let green = pixel_at(&s, refused[0], 192, 96);
    assert!(green[1] > 0.6 && green[0] < 0.2, "the refused fill's pixels: {green:?}");
}

#[test]
fn what_a_fill_left_as_it_was_is_nobodys_work() {
    use super::split_subjects::{fills_own_work, says_someone_new, someone_new};
    // The row of the two tests above: the hole 4–11, the fill laid over 2–13, the hole widened to
    // 0–13, the subject behind at 0–1. A figure the fill left as it was (`as_was`) was there
    // before, unless it lies mostly where the cut subject stood: that is the cut subject painted
    // back as it was.
    let det = |from: usize, to: usize| -> Vec<f32> { (0..16).map(|k| if k >= from && k < to { 1.0 } else { 0.0 }).collect() };
    let touched = det(2, 14);
    let hole = det(4, 12);
    let grown = det(0, 14);
    let behind = vec![("person".to_string(), det(0, 2), 2u64)];
    let cut = ("person", &hole[..], 8u64);
    let bounds = |from: i32, to: i32| Rect::new(from, 0, to, 1);
    let kept = |d: &[f32], b: Rect| someone_new("person", d, b, &touched, cut, &behind, &[], &|_| true, 16);
    assert_eq!(kept(&det(11, 14), bounds(11, 14)), None, "mostly beside the cut one, left as it was: there before");
    assert_eq!(kept(&det(5, 8), bounds(5, 8)).as_deref(), Some("a person painted into the hole (3×1 px)"), "where the cut one stood: painted back as it was");
    let own = |d: &[f32], as_was: bool| fills_own_work("person", d, &hole, &grown, cut, &behind, &|_| 0.0, &|_| as_was, 16, &|_| false);
    assert_eq!(own(&det(11, 14), true), None, "mostly beside the cut one, left as it was");
    assert!(own(&det(11, 14), false).unwrap().contains("was not there"), "changed: painted in");
    assert!(own(&det(5, 8), true).unwrap().contains("was not there"), "where the cut one stood, as it was: painted back");
    // A thing is judged on its own spot, kept or not.
    let cup = fills_own_work("cup", &det(4, 10), &hole, &grown, cut, &behind, &|_| 0.0, &|_| true, 16, &|_| false);
    assert!(cup.unwrap().contains("a cup in the hole"));
    // The verdicts a picture of a person lifts.
    assert!(says_someone_new("a person painted into the hole (3×1 px)"));
    assert!(says_someone_new("a dog in the hole that was not there"));
    assert!(!says_someone_new("a table in the hole that was not there"), "a thing is no picture of a person");
    assert!(!says_someone_new("the person painted back"));
    assert!(!says_someone_new("a person over the one behind, more than a completion"));
}

#[test]
fn a_figure_in_a_picture_is_in_the_picture() {
    use super::split_subjects::in_a_picture;
    // A 16×16 area: a figure 2×4 px at (6, 4) inside a framed picture 6×8 px at (4, 2), one
    // half out of it, and the picture handed back as the figure.
    let area = Rect::new(0, 0, 16, 16);
    let figure = |x0: i32, y0: i32, x1: i32, y1: i32| -> Vec<f32> {
        (0..256)
            .map(|k| {
                let (x, y) = (k % 16, k / 16);
                if x >= x0 && x < x1 && y >= y0 && y < y1 { 1.0 } else { 0.0 }
            })
            .collect()
    };
    let frame = Rect::new(4, 2, 10, 10);
    assert!(in_a_picture(&figure(6, 4, 8, 8), Rect::new(6, 4, 8, 8), &[frame], area));
    assert!(!in_a_picture(&figure(8, 4, 12, 8), Rect::new(8, 4, 12, 8), &[frame], area), "half out of it");
    assert!(!in_a_picture(&figure(4, 2, 10, 10), Rect::new(4, 2, 10, 10), &[frame], area), "the figure handed back as the picture");
    assert!(!in_a_picture(&figure(6, 4, 8, 8), Rect::new(6, 4, 8, 8), &[], area), "no picture");
}

#[test]
fn someone_in_a_framed_picture_is_a_picture_of_a_person() {
    // The scene of `split_rejects_a_fill_that_paints_someone_new_into_the_hole`, but what every
    // fill paints into the right one's hole is a framed picture with a person in it
    // (`appears_for`: the person for "person", the frame for "framed picture"). A picture of a
    // person is nobody's work: the fill is accepted at the first try, and the finished background
    // is not said to have anyone left on it.
    let front = [0.55, 0.1, 0.95, 0.95];
    let back = [0.05, 0.1, 0.45, 0.85];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        appears_for: vec![("person".to_string(), vec![[0.65, 0.30, 0.72, 0.42]]), ("framed picture".to_string(), vec![[0.63, 0.28, 0.74, 0.44]])],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1})).unwrap();
    assert_eq!((r["count"].as_u64(), r["incomplete"].as_array().map(Vec::len)), (Some(3), Some(0)), "{r}");
    assert_eq!(r["leftBehind"], json!([]), "{r}");
    let tries = r["tries"].as_array().unwrap();
    assert!(tries.len() == 2 && tries.iter().all(|t| t["verdict"] == "accepted"), "{tries:?}");
    let d = &r["detector"];
    assert!(d["pictures"].as_u64().unwrap() >= 2, "the figure taken for a picture at the fill's check and at the finished background: {d}");
    assert_eq!(d["kept"], 0, "{d}");
    let st = fake.state();
    let frames = st.runs.iter().filter(|r| r.request["task"] == "segment" && r.request["prompt"] == "framed picture").count();
    assert!(frames >= 2, "the detector was asked for the framed pictures: {frames}");
}

#[test]
fn a_sweep_takes_a_subjects_part_apart_from_their_mask_for_theirs() {
    use super::split_subjects::{Subject, part_below};
    // A wide person across the lower half of a 256×192 canvas and one above them, apart. Once
    // the front one's fill has run, the detector also returns a small person just below the
    // back one, within their width (`uncovered`): their lower body, apart from their mask (a
    // table edge cut it off), in the feathered edge of that fill, which leaves it the colour it
    // was (green, the fake fill's colour). The fill behind the front one is accepted (what it
    // left as it was is nobody's work); the sweep's second pass takes the part for the back
    // one's: it goes into their mask and their layer, no layer of its own is cut, and the fill
    // behind them takes it out with them.
    let front = [0.05, 0.62, 0.95, 0.95];
    let back = [0.20, 0.05, 0.60, 0.32];
    let part = [0.28, 0.37, 0.52, 0.43];
    let fake = FakeNative::start_with(photocraft_genai::fake::Options {
        segments: vec![front, back],
        uncovered: vec![part],
        color: [10, 200, 10, 255],
        occlusion: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = session_of(&fake.url, 256, 192);
    s.execute("edit.fill", json!({"contents": "color", "color": "#c82828"})).unwrap();
    s.execute("select.rect", json!({"x": 71, "y": 71, "width": 63, "height": 12})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#0ac80a"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute(SPLIT, json!({"seed": 1, "sweep": true, "layers": 6})).unwrap();
    assert_eq!(r["count"], 3, "two people and the background, no layer for the part: {r}");
    assert_eq!(r["incomplete"], json!([]), "{r}");
    assert_eq!(r["leftBehind"], json!([]), "{r}");
    let tries = r["tries"].as_array().unwrap();
    assert!(tries.len() == 2 && tries.iter().all(|t| t["verdict"] == "accepted"), "{tries:?}");
    assert!(r["detector"]["kept"].as_u64().unwrap() >= 1, "the part, left as it was by the front one's fill: {}", r["detector"]);
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    assert_eq!(layer_name(&s, ids[1]), "Person 1", "the back one");
    assert!(alpha_at(&s, ids[1], 102, 40) > 0.5, "holds their body");
    assert!(alpha_at(&s, ids[1], 102, 77) > 0.5, "and the part below their mask");
    assert_eq!(alpha_at(&s, ids[2], 102, 77), 0.0, "which the front one's layer does not");
    // The rule itself: smaller, its top below their middle, at most a quarter of their height
    // under their bottom, within their width widened by a quarter each side.
    let subject = |x0: i32, y0: i32, x1: i32, y1: i32| Subject {
        cov: Vec::new(),
        bounds: Rect::new(x0, y0, x1, y1),
        pixels: ((x1 - x0) * (y1 - y0)) as u64,
        term: "person".to_string(),
        asked: false,
        alpha: Vec::new(),
        completed: false,
        grown: false,
    };
    let of = subject(100, 0, 200, 100);
    assert!(part_below(&subject(120, 110, 180, 140), &of));
    assert!(!part_below(&subject(120, 130, 180, 160), &of), "too far under them");
    assert!(!part_below(&subject(120, 40, 180, 70), &of), "over their middle: their own body, matched as such");
    assert!(!part_below(&subject(60, 110, 120, 140), &of), "out of their width");
    assert!(!part_below(&subject(100, 110, 240, 220), &of), "larger than them");
    assert!(part_below(&subject(80, 90, 220, 120), &of), "a little wider, starting above their bottom");
}
