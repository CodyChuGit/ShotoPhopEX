//! ShotoPhopEX: the Remove tool and Spot Healing through the generative fill (`generative`),
//! against the fake worker. Moved from photocraft-engine-core's `generate_refs` tests when 0.24
//! moved these commands here.

use photocraft_doc::LayerId;
use photocraft_engine_core::Session;
use photocraft_engine_core::generate_cmds::generative_info;
use photocraft_genai::fake::FakeNative;
use serde_json::{Value, json};

/// A 64×48 document with a textured layer, a second small layer (the reference) and a
/// selection, talking to `url`.
fn session(url: &str) -> (Session, LayerId, LayerId) {
    let mut s = crate::session();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let paint = |s: &mut Session, rect: photocraft_geom::Rect, rgba: [f32; 4]| {
        s.edit("paint", |doc, active| {
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let n = surf.channels();
            let mut tmp = [0.0f32; 8];
            let used = photocraft_raster::from_rgba_into(&surf.format(), rgba, &mut tmp).min(n);
            let px: Vec<f32> = (0..rect.width() * rect.height()).flat_map(|_| tmp[..used].to_vec()).collect();
            surf.write_region(rect, &px);
            Ok(())
        })
        .unwrap();
    };
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s, photocraft_geom::Rect::new(0, 0, 64, 48), [0.2, 0.6, 0.9, 1.0]);
    let base = s.active().unwrap().active_layer.unwrap();
    s.execute("layer.new.layer", json!({"name": "Hat"})).unwrap();
    paint(&mut s, photocraft_geom::Rect::new(10, 10, 30, 22), [0.9, 0.1, 0.1, 1.0]);
    let hat = s.active().unwrap().active_layer.unwrap();
    s.execute("layer.select", json!({"layer": base.0})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    (s, base, hat)
}

/// The Remove tool and Spot Healing go through the fill model when asked (`generative`), a
/// layer each: each asks the detector for the object under the stroke first (the fake finds
/// nothing: the stroke alone), then the rewriter, then the fill. Without a backend configured
/// they fall back to their classical fills, in place.
#[test]
fn the_remove_tool_and_spot_healing_go_through_the_model_when_asked_and_fall_back_without_a_server() {
    let fake = FakeNative::start().unwrap();
    let (mut s, _, _) = session(&fake.url);
    let layers = s.active().unwrap().doc.layers.len();
    s.execute("paint.remove", json!({"points": [[20, 16], [26, 16]], "size": 6, "generative": true, "variations": 1})).unwrap();
    s.execute("paint.spotHealing", json!({"points": [[22, 18]], "size": 6, "type": "generative", "variations": 1})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), layers + 2, "a layer each for the two tools");
    assert_eq!(fake.state().runs.len(), 6, "two detector asks, two rewrites, two fills");
    s.edit_prefs(|p| {
        p.integrations.models_path.clear();
        p.integrations.runtime_path.clear();
    });
    let n = s.active().unwrap().doc.layers.len();
    s.execute("paint.remove", json!({"points": [[20, 16], [26, 16]], "size": 6, "generative": true, "variations": 1})).unwrap();
    s.execute("paint.spotHealing", json!({"points": [[22, 18]], "size": 6, "type": "generative", "variations": 1})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), n, "the classical fills, in place");
    assert_eq!(fake.state().runs.len(), 6, "nothing more was asked of the server");
}

#[test]
fn name_shadow_asks_a_whole_picture_removal_for_the_objects_shadow_too() {
    use photocraft_genai::fake::Options;
    let fake = FakeNative::start_with(Options {
        segments_for: vec![("red hat".into(), vec![[0.25, 1.0 / 3.0, 0.75, 170.0 / 300.0]])],
        enhanced_text: Some("Remove the red hat.".into()),
        ..Options::default()
    })
    .unwrap();
    let (mut s, _, _) = session(&fake.url);
    s.edit_prefs(|p| {
        p.integrations.default_fill_template = "qwen-2.1/edit".into();
        p.integrations.allow_research_models = true;
    });
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    let stroke = |extra: serde_json::Value| {
        let mut p = json!({"points": [[110, 135], [210, 135]], "size": 30, "hardness": 1, "type": "generative", "variations": 1, "seed": 3});
        if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        p
    };
    let mut said = Vec::new();
    for extra in [json!({}), json!({"nameShadow": true})] {
        let r = s.execute("paint.spotHealing", stroke(extra)).unwrap();
        let info = generative_info(s.active().unwrap().doc.layer(LayerId(r["layer"].as_u64().unwrap())).unwrap()).unwrap();
        said.push(info.enhanced);
    }
    assert!(said[0].starts_with("Remove the red hat and fill its place"), "off by default: {said:?}");
    assert!(said[1].starts_with("Remove the red hat and its shadow and fill its place"), "{said:?}");
}

#[test]
fn a_generative_stroke_takes_the_object_under_it_whole_when_it_covers_enough_of_it() {
    use photocraft_genai::fake::Options;
    // A 400×300 picture; the detector "finds" an object at (100, 100)–(220, 170) under the
    // stroke's spots (8400 pixels), and, asked by the name the rewriter gives it, the whole
    // "red hat" at (100, 100)–(300, 170) (14000 pixels: a thin object's point prompt returns
    // only the brushed part).
    let fake = FakeNative::start_with(Options {
        segments: vec![[0.25, 1.0 / 3.0, 0.55, 170.0 / 300.0]],
        segments_for: vec![("red hat".into(), vec![[0.25, 1.0 / 3.0, 0.75, 170.0 / 300.0]])],
        enhanced_text: Some("Remove the red hat.".into()),
        ..Options::default()
    })
    .unwrap();
    let (mut s, _, _) = session(&fake.url);
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    // A stroke across the object (its footprint 150×51, from x 85 to 235): the rewriter names
    // "the red hat", the detector finds the whole of it by that name and the stroke covers
    // over a tenth of it, so it goes whole, the hole grown by 15 % of its larger side (the
    // stroke and the whole hat: x 85 to 300 → 215 → 32). The point prompt is not asked.
    let r = s
        .execute("paint.spotHealing", json!({"points": [[110, 135], [210, 135]], "size": 30, "hardness": 1, "type": "generative", "variations": 1, "seed": 3}))
        .unwrap();
    let made = LayerId(r["layer"].as_u64().unwrap());
    let info = generative_info(s.active().unwrap().doc.layer(made).unwrap()).unwrap();
    assert_eq!(info.grow, 32, "{info:?}");
    // The rewriter gives the name; the fill keeps the generic wording around it.
    assert_eq!(
        info.enhanced,
        "Remove the red hat and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there",
        "{info:?}"
    );
    let rect = photocraft_geom::Rect::new(info.rect[0], info.rect[1], info.rect[0] + info.rect[2], info.rect[1] + info.rect[3]);
    assert!(rect.contains(70, 70) && rect.contains(330, 200), "the request takes in the whole hat, its growth and margin: {rect:?}");
    {
        let st = fake.state();
        assert_eq!(st.runs.len(), 3, "the rewriter, the detector by name, the fill");
        assert!(st.runs[1].run.prompt == "red hat", "asked by the rewriter's name: {}", st.runs[1].request);
    }
    // A dab on one spot of it (a blemish, not the object): the named hat is 14000 pixels, the
    // dab covers under a tenth of it, and so of the "hat" the shorter name finds (the fake's
    // default object, 8400 pixels); the point prompt's object (the same) is far bigger than
    // the dab's box. The hat stays, the dab is healed with the least growth.
    let r =
        s.execute("paint.spotHealing", json!({"points": [[120, 130]], "size": 8, "hardness": 1, "type": "generative", "variations": 1, "seed": 3})).unwrap();
    let made = LayerId(r["layer"].as_u64().unwrap());
    let info = generative_info(s.active().unwrap().doc.layer(made).unwrap()).unwrap();
    assert_eq!(info.grow, 16, "{info:?}");
    let rect = photocraft_geom::Rect::new(info.rect[0], info.rect[1], info.rect[0] + info.rect[2], info.rect[1] + info.rect[3]);
    // The request is widened to at least 384 px a side (the canvas is 300 tall) so the model
    // sees context around the dab, but no further.
    assert!(rect.width() == 384 && rect.height() == 300 && !rect.contains(399, 10), "a dab's request: {rect:?}");
    assert_eq!(fake.state().runs.len(), 8, "the rewriter, the detector by \"red hat\" then \"hat\", the detector by point, the fill");
    // `wholeObject: false` never asks the detector; `enhance: false` never asks the rewriter.
    s.execute(
        "paint.remove",
        json!({"points": [[110, 135], [210, 135]], "size": 30, "generative": true, "wholeObject": false, "enhance": false, "variations": 1}),
    )
    .unwrap();
    assert_eq!(fake.state().runs.len(), 9, "the fill alone for the Remove stroke");
    // With no rewriter (`enhance: false`) the point prompt's object is taken when it fits
    // the stroke's box: the stroke's box is 150×51, the object 120×70 (under 1.5× each way).
    let r = s
        .execute(
            "paint.spotHealing",
            json!({"points": [[110, 135], [210, 135]], "size": 30, "hardness": 1, "type": "generative", "enhance": false, "variations": 1, "seed": 3}),
        )
        .unwrap();
    let made = LayerId(r["layer"].as_u64().unwrap());
    let info = generative_info(s.active().unwrap().doc.layer(made).unwrap()).unwrap();
    // The stroke stops short of its last point since upstream 0.23 turned Catch-up on Stroke
    // End off by default (#11887), so the hole is x 84 to 223 (139 px, grown 21).
    assert_eq!((info.grow, info.enhanced.as_str()), (21, ""), "the stroke and the point object, x 84 to 223: {info:?}");
    assert_eq!(fake.state().runs.len(), 11, "the detector by point, the fill");
}

#[test]
fn a_removal_matches_the_tone_of_its_surroundings_at_the_hole() {
    // The fake paints its result a flat colour; the document is another. A removal's result
    // comes back matched to the document at the hole's edge, so the layer's pixels inside the
    // hole are the document's tone, not the fake's; a plain fill keeps the fake's colour.
    use photocraft_genai::fake::Options;
    let fake = FakeNative::start_with(Options { color: [200, 120, 40, 255], ..Options::default() }).unwrap();
    let (mut s, _, _) = session(&fake.url);
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    s.execute("edit.fill", json!({"color": "#2060a0"})).unwrap();
    let sample = |s: &Session, layer: LayerId, x: i32, y: i32| -> [u8; 4] {
        let l = s.active().unwrap().doc.layer(layer).unwrap();
        let mut px = [[0u8; 4]; 1];
        l.surface().unwrap().read_rgba8_into(photocraft_geom::Rect::new(x, y, x + 1, y + 1), &mut px);
        px[0]
    };
    let before = s.active().unwrap().doc.layers.len();
    let r = s
        .execute(
            "paint.spotHealing",
            json!({"points": [[200, 150]], "size": 40, "hardness": 1, "type": "generative", "enhance": false, "wholeObject": false, "seed": 3}),
        )
        .unwrap();
    let removed = sample(&s, LayerId(r["layer"].as_u64().unwrap()), 200, 150);
    assert!(removed[0] < 80 && removed[2] > 120, "the document's blue, not the fake's orange: {removed:?}");
    // Two seeds by default: two layers, the shown one first (the fake paints both alike).
    let ids: Vec<u64> = r["layers"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    assert_eq!((ids.len(), s.active().unwrap().doc.layers.len()), (2, before + 2), "{r}");
    let d = s.active().unwrap();
    assert!(d.doc.layer(LayerId(ids[0])).unwrap().visible && !d.doc.layer(LayerId(ids[1])).unwrap().visible);
    s.execute("select.rect", json!({"x": 180, "y": 130, "width": 40, "height": 40})).unwrap();
    let r = s.execute(photocraft_engine_core::generate_cmds::FILL, json!({"prompt": "a mark", "enhance": false, "seed": 3})).unwrap();
    let filled = sample(&s, LayerId(r["layer"].as_u64().unwrap()), 200, 150);
    assert!(filled[0] > 150 && filled[2] < 80, "a fill keeps what the model painted: {filled:?}");
}
