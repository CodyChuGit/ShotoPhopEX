//! `generate.clownPass`, `edit.generativeRemove` and `select.byPoint {detail: "part"}` against the
//! fake worker, whose detector returns the rectangles it is given.

use photocraft_doc::{LabelColor, LayerId};
use photocraft_genai::fake::{FakeNative, Options};
use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// A 64×48 document with one red layer, talking to `url`.
fn session(url: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"use": "color", "color": [200, 40, 40]})).ok();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    s
}

fn alpha_at(s: &Session, id: LayerId, x: i32, y: i32) -> f32 {
    let d = s.active().unwrap();
    d.doc.layer(id).unwrap().surface().unwrap().read_region(Rect::new(x, y, x + 1, y + 1))[3]
}

#[test]
fn clown_pass_puts_every_thing_on_its_own_layer_with_nothing_painted() {
    // "person" finds two people; "object" finds the first person again (the same thing) and a
    // small cup inside the second person's box; every other kind finds the first person.
    let fake = FakeNative::start_with(Options {
        segments: vec![[0.0, 0.0, 0.5, 0.5]],
        segments_for: vec![
            ("person".into(), vec![[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
            ("object".into(), vec![[0.0, 0.0, 0.5, 0.5], [0.75, 0.75, 0.875, 0.875]]),
        ],
        ..Options::default()
    })
    .unwrap();
    let mut s = session(&fake.url);
    let (before, steps) = {
        let d = s.active().unwrap();
        (d.doc.layers.len(), d.history.past_len())
    };
    let r = s.execute(CLOWN_PASS, json!({})).unwrap();
    assert_eq!(r["count"], 4, "background, two people and the cup: {r}");
    assert_eq!(r["names"], json!(["Background", "Person 1", "Person 2", "Object 1"]), "{r}");
    let d = s.active().unwrap();
    assert_eq!(d.doc.layers.len(), before + 1, "one group");
    assert_eq!(d.history.past_len(), steps + 1, "one undo step");
    let group = d.doc.layer(LayerId(r["group"].as_u64().unwrap())).unwrap();
    assert_eq!(group.name, "Clown Pass");
    let ids: Vec<LayerId> = r["layers"].as_array().unwrap().iter().map(|v| LayerId(v.as_u64().unwrap())).collect();
    // Each pixel on exactly one layer: the cup's pixels are the cup's, not the person's.
    for (x, y) in [(5, 5), (60, 5), (40, 30), (51, 39)] {
        let on: Vec<usize> = (0..ids.len()).filter(|i| alpha_at(&s, ids[*i], x, y) > 0.5).collect();
        assert_eq!(on.len(), 1, "({x}, {y}) on {on:?}");
    }
    assert_eq!(alpha_at(&s, ids[1], 5, 5), 1.0, "person 1");
    assert_eq!(alpha_at(&s, ids[2], 40, 30), 1.0, "person 2");
    assert_eq!(alpha_at(&s, ids[3], 51, 39), 1.0, "the cup");
    assert_eq!(alpha_at(&s, ids[2], 51, 39), 0.0, "a hole in person 2 where the cup is");
    assert_eq!(alpha_at(&s, ids[0], 60, 5), 1.0, "the background");
    // Each layer its own label colour.
    let labels: Vec<LabelColor> = ids.iter().map(|id| s.active().unwrap().doc.layer(*id).unwrap().label).collect();
    for (i, a) in labels.iter().enumerate() {
        assert!(labels.iter().skip(i + 1).all(|b| b != a), "{labels:?}");
    }
    // Nothing but the detector ran.
    assert!(fake.state().runs.iter().all(|r| r.request["task"] == "segment"));
    assert!(s.undo());
    assert_eq!(s.active().unwrap().doc.layers.len(), before);
    // A prompt names the kinds of thing to look for.
    let r = s.execute(CLOWN_PASS, json!({"prompt": "person", "name": "IDs"})).unwrap();
    assert_eq!(r["names"], json!(["Background", "Person 1", "Person 2"]), "{r}");
    assert!(matches!(s.execute(CLOWN_PASS, json!({"prompt": vec!["a"; 30].join(",")})), Err(EngineError::BadParams { .. })));
}

#[test]
fn clown_pass_says_so_when_nothing_is_found() {
    let fake = FakeNative::start_with(Options { segments: vec![], ..Options::default() }).unwrap();
    let mut s = session(&fake.url);
    assert!(s.execute(CLOWN_PASS, json!({})).is_err());
}

#[test]
fn generative_remove_takes_the_selection_out_through_the_fill_model() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    assert!(s.execute("edit.generativeRemove", json!({})).is_err(), "needs a selection");
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 12})).unwrap();
    let layers = s.active().unwrap().doc.layers.len();
    let r = s.execute("edit.generativeRemove", json!({})).unwrap();
    assert!(r["layer"].is_number(), "a generative layer: {r}");
    // The shown fill and its hidden variation (two seeds, as the Remove tool's Pick Best).
    assert_eq!(s.active().unwrap().doc.layers.len(), layers + 2);
    assert!(fake.state().runs.iter().any(|r| r.request["task"] != "segment" && r.request["task"] != "text"), "the fill ran");
    // Without the generative models set up it is greyed out.
    s.edit_prefs(|p| {
        p.integrations.models_path = String::new();
        p.integrations.runtime_path = String::new();
    });
    assert!(crate::commands::find("edit.generativeRemove").is_some_and(|c| (c.enabled)(&s).is_err()));
}

#[test]
fn a_part_click_asks_for_all_three_answers_and_keeps_the_middle_one() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url);
    s.execute("select.byPoint", json!({"x": 30, "y": 20, "detail": "part"})).unwrap();
    let asked: Vec<serde_json::Value> = fake.state().runs.iter().map(|r| r.request["params"]["candidate"].clone()).collect();
    assert_eq!(asked, vec![json!(0), json!(1), json!(2)], "SAM's three answers");
    assert!(s.active().unwrap().doc.selection.is_some());
    // A whole click asks once; a bad detail is refused.
    let n = fake.state().runs.len();
    s.execute("select.byPoint", json!({"x": 30, "y": 20})).unwrap();
    assert_eq!(fake.state().runs.len(), n + 1);
    assert!(matches!(s.execute("select.byPoint", json!({"x": 30, "y": 20, "detail": "tiny"})), Err(EngineError::BadParams { .. })));
}
