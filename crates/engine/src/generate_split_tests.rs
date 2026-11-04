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
    // The detector was asked once to find the people (the two filled the layer budget), then
    // again after each fill, for the subject painted back and the people still behind.
    let st = fake.state();
    let segments = st.runs.iter().filter(|r| r.request["task"] == "segment").count();
    assert_eq!(segments, 3, "found, then checked after each fill: {segments}");
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
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
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
    // The other edge sources, by name.
    let r = s.execute(SPLIT, json!({"layers": 2, "edges": "refine"})).unwrap();
    assert_eq!(r["edges"], "refine", "{r}");
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
    // After the removal, the backdrop fill: the hole's box flattened, the subject never named.
    let st = fake.state();
    let backdrop: Vec<_> = st.runs.iter().filter(|r| r.request["prompt"].as_str().is_some_and(|p| p.contains("with nobody there"))).collect();
    assert_eq!(backdrop.len(), 4, "four backdrop fills after the removal");
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
    // The record of every try: the removal, then the grey-box fills, each with its verdict.
    let tries = r["tries"].as_array().unwrap();
    assert_eq!((tries.len(), r["fills"].as_u64()), (6, Some(6)), "{tries:?}");
    assert!(tries.iter().take(5).all(|t| t["behind"] == "Person 2" && t["verdict"].as_str().unwrap_or("").contains("was not there")), "{tries:?}");
    assert_eq!((tries[0]["fill"].as_str(), tries[1]["fill"].as_str()), (Some("removal"), Some("fill")), "{tries:?}");
    assert_eq!(
        (tries[5]["behind"].as_str(), tries[5]["fill"].as_str(), tries[5]["verdict"].as_str()),
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
    assert_eq!(tries.len(), 2, "{tries:?}");
    assert_eq!(tries[0]["fill"], "removal", "{tries:?}");
    assert!(tries[0]["verdict"].as_str().unwrap_or("").starts_with("kept what stood there (100 %"), "{tries:?}");
    assert_eq!((tries[1]["fill"].as_str(), tries[1]["verdict"].as_str()), (Some("fill"), Some("accepted")), "{tries:?}");
    // The detector was asked after the first try and saw nothing: the pixels refused it.
    let st = fake.state();
    let after_first: Vec<&str> = st
        .runs
        .iter()
        .skip_while(|r| r.request["task"] != "fill" && r.request["task"] != "edit")
        .skip(1)
        .map(|r| r.request["task"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(after_first.first().copied(), Some("fill"), "no detector call between the refused try and the next: {after_first:?}");
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
    let own = |term: &str, d: &[f32], kept: f32| fills_own_work(term, d, &hole, &grown, cut, &behind, &|_| kept);
    assert_eq!(own("person", &det(0, 6), 0.0), None, "the subject with twice its pixels added: completed");
    assert!(own("person", &det(0, 6), 0.5).unwrap().contains("painted back over"), "but looking as the cut one did: painted back");
    assert!(own("person", &det(0, 7), 0.0).unwrap().contains("more than a completion"), "more than twice: someone new standing over it");
    assert!(own("person", &det(12, 14), 0.0).unwrap().contains("was not there"), "in the repainted band, continuing nobody: painted in");
    assert!(own("person", &det(4, 12), 0.0).unwrap().contains("the person painted back"), "the cut one's own shape");
    assert!(own("cup", &det(4, 10), 0.0).unwrap().contains("a cup in the hole"), "another kind on the spot: painted in");
    assert_eq!(own("person", &det(14, 16), 0.0), None, "outside the repainted area: was there before");
    // A thing is judged on its own spot: a cup in the widened band only was there before (a
    // saucer beside the cup); a person there is painted in.
    assert_eq!(own("cup", &det(12, 14), 0.0), None, "a cup beside the hole");
    assert_eq!(own("cup", &det(0, 6), 0.0), None, "a cup mostly beside the hole");
    assert!(own("person", &det(12, 14), 0.0).is_some(), "a person in the widened band");
}
