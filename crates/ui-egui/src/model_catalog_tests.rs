use super::*;

fn item(id: &str, family: &str) -> Item {
    Item { id: id.into(), name: id.into(), family: family.into(), allowed: true, ..Item::default() }
}

fn page(ids: &[(&str, &str)]) -> Vec<Item> {
    ids.iter().map(|(id, f)| item(id, f)).collect()
}

fn image_page() -> Vec<Item> {
    page(&[
        ("flux3-klein/image", "flux3-klein"),
        ("krea2-turbo/image-style", "krea2"),
        ("krea2-turbo/image", "krea2"),
        ("minimax-h3-image/image", "minimax-h3-image"),
        ("moss-image/image-base", "moss-image"),
        ("moss-image/image", "moss-image"),
        ("moss-image/turbo", "moss-image"),
        ("p-h3-image/image", "p-h3-image"),
        ("qwen-2.1/image-ref", "qwen-image-2.1"),
        ("qwen-2.1/image", "qwen-image-2.1"),
        ("qwen-edit-2511/image-ref", "qwen-image-edit-2511"),
    ])
}

fn fill_page() -> Vec<Item> {
    page(&[
        ("flux3-klein/fill", "flux3-klein"),
        ("krea2-turbo/fill", "krea2"),
        ("qwen-2.1/fill", "qwen-image-2.1"),
        ("qwen-edit-2511/fill-guided", "qwen-image-edit-2511"),
        ("qwen-edit-2511/fill-lightning-4", "qwen-image-edit-2511"),
        ("qwen-edit-2511/fill-lightning-8", "qwen-image-edit-2511"),
        ("qwen-edit-2511/fill-ref-lightning-8", "qwen-image-edit-2511"),
        ("qwen-edit-2511/fill-ref", "qwen-image-edit-2511"),
        ("qwen-edit-2511/fill", "qwen-image-edit-2511"),
        ("qwen-edit-2511/remove-lightning-8", "qwen-image-edit-2511"),
    ])
}

fn keys(rows: &[Row]) -> Vec<&str> {
    rows.iter().map(|r| r.model.key).collect()
}

#[test]
fn the_image_page_has_one_row_per_model_best_picks_first() {
    let items = image_page();
    let r = rows(&items, "image", false);
    assert_eq!(keys(&r), ["krea2", "moss", "flux3-klein", "qwen-2.1", "p-h3-image", "minimax-h3-image"], "11 recipes, 6 rows");
    // Qwen Image Edit only builds from references: its row shows once one is attached.
    assert!(keys(&rows(&items, "image", true)).contains(&"qwen-edit-2511"));
    // MOSS's row holds MOSS and MOSS Turbo, not the base.
    let moss = r.iter().find(|r| r.model.key == "moss").unwrap();
    let ids: Vec<&str> = moss.templates.iter().map(|&i| items[i].id.as_str()).collect();
    assert_eq!(ids, ["moss-image/image", "moss-image/turbo"]);
}

#[test]
fn speed_picks_the_version_and_a_missing_speed_takes_the_nearest() {
    let items = image_page();
    let r = rows(&items, "image", false);
    let get = |key: &str, s: Speed, refs: bool| {
        let row = r.iter().find(|r| r.model.key == key).unwrap();
        resolve(&items, row, "image", s, refs).map(|t| t.id.clone()).unwrap()
    };
    assert_eq!(get("moss", Speed::Fast, false), "moss-image/turbo");
    assert_eq!(get("moss", Speed::Balanced, false), "moss-image/turbo");
    assert_eq!(get("moss", Speed::Full, false), "moss-image/image", "Full = no turbo LoRA");
    assert_eq!(get("krea2", Speed::Full, false), "krea2-turbo/image", "no Krea 2 Raw yet: Balanced");
    assert_eq!(get("krea2", Speed::Balanced, true), "krea2-turbo/image-style", "references take the style recipe");
    assert_eq!(get("qwen-2.1", Speed::Fast, true), "qwen-2.1/image-ref");
    assert_eq!(get("flux3-klein", Speed::Full, false), "flux3-klein/image");
    let moss = r.iter().find(|r| r.model.key == "moss").unwrap();
    assert_eq!(row_speeds(&items, moss, "image"), [Speed::Fast, Speed::Balanced, Speed::Full]);
}

#[test]
fn fill_maps_lightning_4_8_and_40_steps_to_the_three_speeds() {
    let items = fill_page();
    let r = rows(&items, "fill", false);
    assert_eq!(keys(&r), ["qwen-2.1", "qwen-edit-2511", "krea2", "flux3-klein"]);
    let qe = r.iter().find(|r| r.model.key == "qwen-edit-2511").unwrap();
    let get = |s, refs| resolve(&items, qe, "fill", s, refs).unwrap().id.clone();
    assert_eq!(get(Speed::Fast, false), "qwen-edit-2511/fill-lightning-4");
    assert_eq!(get(Speed::Balanced, false), "qwen-edit-2511/fill-lightning-8");
    assert_eq!(get(Speed::Full, false), "qwen-edit-2511/fill");
    assert_eq!(get(Speed::Fast, true), "qwen-edit-2511/fill-ref-lightning-8");
    assert_eq!(get(Speed::Full, true), "qwen-edit-2511/fill-ref");
    let ids: Vec<&str> = qe.templates.iter().map(|&i| items[i].id.as_str()).collect();
    assert!(!ids.contains(&"qwen-edit-2511/remove-lightning-8") && !ids.contains(&"qwen-edit-2511/fill-guided"), "other tools' recipes");
}

#[test]
fn a_stored_choice_follows_speed_and_references() {
    let items = fill_page();
    assert_eq!(remap(&items, "fill", "qwen-edit-2511/fill-lightning-8", Speed::Full, false).as_deref(), Some("qwen-edit-2511/fill"));
    assert_eq!(remap(&items, "fill", "qwen-edit-2511/fill", Speed::Balanced, true).as_deref(), Some("qwen-edit-2511/fill-ref-lightning-8"));
    // A recipe that is not a row (a script named it) runs as named.
    assert_eq!(remap(&items, "fill", "qwen-edit-2511/remove-lightning-8", Speed::Full, false).as_deref(), Some("qwen-edit-2511/remove-lightning-8"));
    assert_eq!(remap(&items, "fill", "nope/x", Speed::Full, false), None);
}

#[test]
fn the_outpaint_turbo_is_expands_fast_and_not_an_edit_row() {
    let items = page(&[("qwen-2.1/edit", "qwen-image-2.1"), ("qwen-2.1/outpaint-turbo", "qwen-image-2.1"), ("qwen-2.1/expand", "qwen-image-2.1")]);
    assert_eq!(role("qwen-2.1/outpaint-turbo", "edit"), Role::Hidden);
    let r = rows(&items, "expand", false);
    assert_eq!(resolve(&items, &r[0], "expand", Speed::Fast, false).unwrap().id, "qwen-2.1/outpaint-turbo");
    assert_eq!(Speed::from_tier("best"), Speed::Full);
    assert_eq!(Speed::Full.pref(), "full");
}

#[test]
fn every_shipped_template_has_a_known_model() {
    let mut s = photocraft_engine::Session::new();
    let v = s.execute("generate.models", serde_json::json!({"probe": false})).unwrap();
    for t in v["templates"].as_array().unwrap() {
        let (id, family) = (t["id"].as_str().unwrap(), t["family"].as_str().unwrap());
        assert_ne!(model(model_key(id, family)).key, "other", "{id} ({family}) has no model row");
    }
}
