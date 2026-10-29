//! The Generations library (`generate.jobs` and friends) against the fake ComfyUI server.

use photocraft_doc::LayerId;
use photocraft_genai::fake::{FakeNative, Options};
use serde_json::json;

use super::*;
use crate::generate_cmds::{EDIT, EXPAND, FILL, IMAGE, SIMILAR};
use crate::jobs::Started;

/// A fresh, empty library folder for one test.
fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("photocraft-generations-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A 64×48 document with a textured layer and a 20×16 selection at (10, 8), talking to `url`,
/// its library in `dir` (None = in memory).
fn session(url: &str, dir: Option<&Path>) -> Session {
    let mut s = Session::new();
    if let Some(d) = dir {
        s.set_generations_dir(d.to_path_buf());
    }
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("texture", |doc, active| {
        let r = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let n = surf.channels();
        let mut px = Vec::new();
        for y in 0..r.height() {
            for x in 0..r.width() {
                let v = ((x * 7 + y * 13) % 31) as f32 / 30.0;
                let mut tmp = [0.0f32; 8];
                let used = photocraft_raster::from_rgba_into(&surf.format(), [v, 1.0 - v, 0.5, 1.0], &mut tmp).min(n);
                px.extend_from_slice(&tmp[..used]);
            }
        }
        surf.write_region(r, &px);
        Ok(())
    })
    .unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 8, "width": 20, "height": 16})).unwrap();
    s.edit_prefs(|p| {
        p.integrations.models_path = url.to_string();
        p.integrations.runtime_path = "fake".into();
        p.integrations.allow_research_models = false;
    });
    s
}

fn jobs(s: &mut Session) -> Vec<Value> {
    s.execute(JOBS, json!({})).unwrap()["jobs"].as_array().cloned().unwrap_or_default()
}

#[test]
fn an_inline_fill_is_recorded_with_its_result_and_thumbnail_and_survives_a_restart() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("inline");
    let mut s = session(&fake.url, Some(&dir));
    assert!(jobs(&mut s).is_empty());
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 5})).unwrap();
    let layer = r["layer"].as_u64().unwrap();
    let list = jobs(&mut s);
    assert_eq!(list.len(), 1);
    let j = &list[0];
    assert_eq!(j["command"], FILL);
    assert_eq!(j["status"], "done");
    assert_eq!(j["prompt"], "a kite");
    assert_eq!(j["seed"], 5);
    assert_eq!(j["template"], r["template"], "the resolved template, not auto");
    assert_eq!(j["document"], s.active().unwrap().doc.name.as_str());
    assert_eq!(j["label"], "Generative Fill: a kite");
    assert_eq!(j["result"]["layers"], json!([layer]));
    assert_eq!(j["result"]["saved"], true);
    assert!(j["result"]["bytes"].as_u64().unwrap() > 0);
    assert_eq!(j["info"]["prompt"], "a kite");
    let id = j["id"].as_u64().unwrap();
    // The stored result is the layer over its request rectangle, mask as alpha; the thumbnail
    // fits in 128 px.
    let png = std::fs::read(s.generations.result_path(id, 0).unwrap()).unwrap();
    let img = photocraft_genai::png::decode_rgba8(&png).unwrap();
    assert_eq!((u64::from(img.width), u64::from(img.height)), (r["width"].as_u64().unwrap(), r["height"].as_u64().unwrap()));
    assert!(img.data.chunks(4).any(|p| p[3] == 255), "inside the selection: opaque");
    assert!(img.data.chunks(4).any(|p| p[3] < 128), "the soft-edged mask fades it out towards the request's border");
    let thumb = photocraft_genai::png::decode_rgba8(&std::fs::read(s.generations.thumbnail_path(id).unwrap()).unwrap()).unwrap();
    assert!(thumb.width <= THUMB_SIDE && thumb.height <= THUMB_SIDE);
    // generate.info points back at the row.
    assert_eq!(s.execute("generate.info", json!({"layer": layer})).unwrap()["job"], id);

    // A restart: the row is still there, without its process-local document link.
    s.flush_generations();
    drop(s);
    let mut s2 = Session::new();
    s2.set_generations_dir(dir.clone());
    let list = jobs(&mut s2);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["id"], id);
    assert_eq!(list[0]["status"], "done");
    assert!(s2.generations.get(id).unwrap().doc.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_background_job_shows_as_running_then_done_with_its_files_written_off_thread() {
    let fake = FakeNative::start_with(Options { delay_ms: 150, ..Options::default() }).unwrap();
    let dir = temp_dir("background");
    let mut s = session(&fake.url, Some(&dir));
    let Started::Job(job) = s.start(FILL, json!({"prompt": "a boat", "seed": 1, "variations": 2})).unwrap() else { panic!("a background job") };
    let list = jobs(&mut s);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["status"], "running");
    assert_eq!(list[0]["count"], 2);
    let id = list[0]["id"].as_u64().unwrap();
    assert_eq!(s.generations.record_for_job(job), Some(id));
    assert_eq!(s.execute(JOBS, json!({})).unwrap()["running"], 1);
    s.wait_job(job).unwrap();
    s.flush_generations();
    let rec = s.generations.get(id).unwrap().clone();
    assert_eq!(rec.status, GenStatus::Done);
    let res = rec.result.unwrap();
    assert_eq!(res.layers.len(), 2, "both variations");
    assert_eq!(res.files.len(), 2);
    assert!(res.saved && res.bytes > 0);
    for i in 0..2 {
        assert!(s.generations.result_path(id, i).unwrap().is_file());
    }
    assert_eq!(s.generations.record_for_job(job), Some(id), "an ended job still finds its row (the UI's JobEvent)");
    // The index is on disk after the flush.
    let index: Value = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    assert_eq!(index["version"], 1);
    assert_eq!(index["records"][0]["id"], id);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failures_are_recorded_with_a_kind_the_user_can_act_on_and_refused_requests_are_not() {
    // The server is down.
    let mut s = session("http://127.0.0.1:9", None);
    assert!(s.execute(FILL, json!({"prompt": "a kite"})).is_err());
    let list = jobs(&mut s);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["status"], "failed");
    assert_eq!(list[0]["failure"]["kind"], "serverUnreachable", "{}", list[0]["failure"]["message"]);
    // A parameter mistake or a disabled command never ran: no row.
    assert!(s.execute(FILL, json!({"prompt": "a kite", "seed": -1})).is_err());
    s.execute("select.deselect", json!({})).unwrap();
    assert!(s.execute(FILL, json!({"prompt": "a kite"})).is_err());
    assert_eq!(jobs(&mut s).len(), 1);

    // Out of memory on the server.
    let fake = FakeNative::start_with(Options { fail: Some("torch.OutOfMemoryError: CUDA out of memory".into()), ..Options::default() }).unwrap();
    let mut s = session(&fake.url, None);
    assert!(s.execute(FILL, json!({"prompt": "a kite"})).is_err());
    assert_eq!(jobs(&mut s)[0]["failure"]["kind"], "outOfMemory");

    // A model file the graph names is not installed (ComfyUI's validation error).
    let fake = FakeNative::start_with(Options {
        fail: Some("Prompt outputs failed validation: UNETLoader: Value not in list: unet_name: 'x.safetensors' not in []".into()),
        ..Options::default()
    })
    .unwrap();
    let mut s = session(&fake.url, None);
    assert!(s.execute(IMAGE, json!({"prompt": "a kite"})).is_err());
    assert_eq!(jobs(&mut s)[0]["failure"]["kind"], "modelFilesMissing");
}

#[test]
fn failure_messages_are_classified() {
    use GenFailureKind::*;
    for (msg, kind) in [
        ("the generative backend is not available: cannot reach the ComfyUI server at http://127.0.0.1:8188 (connection refused)", ServerUnreachable),
        ("the generative server reported an error: CUDA error: out of memory", OutOfMemory),
        ("the generative server reported an error: Value not in list: ckpt_name", ModelFilesMissing),
        ("the generation did not finish within 600 s", Timeout),
        ("`Generative Fill: a kite`: the document changed while it ran; the result was discarded", Discarded),
        ("the generative server reported an error: KSampler: bad input", ServerError),
        ("the generative backend is not available: ComfyUI 0.3 is older than 0.26 needed by `x`; update ComfyUI", ServerError),
        ("invalid generative request: the prompt is empty", BadInput),
        ("something else", Other),
    ] {
        assert_eq!(classify_failure(msg), kind, "{msg}");
    }
    assert_eq!(GenFailureKind::ServerUnreachable.label(), "Server unreachable");
}

#[test]
fn a_selection_command_is_recorded_as_a_selection() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url, None);
    s.execute("select.byText", json!({"prompt": "the kite"})).unwrap();
    let j = &jobs(&mut s)[0];
    assert_eq!(j["command"], "select.byText");
    assert_eq!(j["status"], "done");
    assert_eq!(j["result"]["selection"], true);
    assert_eq!(j["result"]["files"], json!([]));
}

#[test]
fn rows_left_running_by_a_closed_process_become_interrupted() {
    let dir = temp_dir("interrupted");
    std::fs::create_dir_all(&dir).unwrap();
    let rec = GenerationRecord { id: 7, created_ms: unix_ms(), command: FILL.into(), status: GenStatus::Running, ..GenerationRecord::default() };
    std::fs::write(dir.join("index.json"), serde_json::to_vec(&json!({"version": 1, "records": [rec]})).unwrap()).unwrap();
    let mut s = Session::new();
    s.set_generations_dir(dir.clone());
    let r = s.generations.get(7).unwrap();
    assert_eq!(r.status, GenStatus::Failed);
    assert_eq!(r.failure.as_ref().map(|f| f.kind), Some(GenFailureKind::Interrupted));
    // A corrupt index is ignored, never an error.
    std::fs::write(dir.join("index.json"), b"{not json").unwrap();
    let mut s = Session::new();
    s.set_generations_dir(dir.clone());
    assert!(jobs(&mut s).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dismiss_hides_pin_protects_and_the_caps_drop_the_oldest_unpinned_results() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("caps");
    let mut s = session(&fake.url, Some(&dir));
    s.generations.set_caps(Caps { max_results: 2, max_bytes: MAX_BYTES });
    let mut ids = Vec::new();
    for seed in 1..=2 {
        s.execute(FILL, json!({"prompt": "a kite", "seed": seed})).unwrap();
        ids.push(jobs(&mut s)[0]["id"].as_u64().unwrap());
    }
    s.execute(PIN, json!({"id": ids[0]})).unwrap();
    s.execute(FILL, json!({"prompt": "a kite", "seed": 3})).unwrap();
    let list = jobs(&mut s);
    assert_eq!(list.len(), 2);
    assert!(s.generations.get(ids[0]).is_some(), "pinned: kept");
    assert!(s.generations.get(ids[1]).is_none(), "the oldest unpinned: dropped");
    assert!(!dir.join(format!("results/{}-0.png", ids[1])).exists(), "its files too");
    assert!(!dir.join(format!("thumbs/{}.png", ids[1])).exists());
    // Dismiss hides without deleting.
    assert_eq!(s.execute(DISMISS, json!({"id": ids[0]})).unwrap()["dismissed"], 1);
    assert_eq!(jobs(&mut s).len(), 1);
    assert_eq!(s.execute(JOBS, json!({"includeDismissed": true})).unwrap()["jobs"].as_array().unwrap().len(), 2);
    assert!(s.generations.result_path(ids[0], 0).unwrap().is_file());
    assert_eq!(s.execute(DISMISS, json!({"which": "completed"})).unwrap()["dismissed"], 1);
    assert!(jobs(&mut s).is_empty());
    // Clear deletes rows and files.
    let kept = s.generations.result_path(ids[0], 0).unwrap();
    assert_eq!(s.execute(CLEAR, json!({"which": "completed"})).unwrap()["removed"], 2);
    assert!(!kept.exists());
    assert_eq!(s.execute(JOBS, json!({"includeDismissed": true})).unwrap()["total"], 0);
    // Bad params.
    assert!(matches!(s.execute(DISMISS, json!({})), Err(EngineError::BadParams { .. })));
    assert!(matches!(s.execute(CLEAR, json!({"which": "everything"})), Err(EngineError::BadParams { .. })));
    assert!(matches!(s.execute(JOBS, json!({"status": "pending"})), Err(EngineError::BadParams { .. })));
    assert!(s.execute(PIN, json!({"id": 1})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn place_brings_a_result_back_after_its_document_closed() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("place");
    let mut s = session(&fake.url, Some(&dir));
    let r = s.execute(FILL, json!({"prompt": "a kite", "seed": 5})).unwrap();
    let (w, h) = (r["width"].as_u64().unwrap(), r["height"].as_u64().unwrap());
    let id = jobs(&mut s)[0]["id"].as_u64().unwrap();
    // Into the same document: a new layer at the original place, with its generative info.
    let p = s.execute(PLACE, json!({"id": id})).unwrap();
    let nid = LayerId(p["layer"].as_u64().unwrap());
    let d = s.active().unwrap();
    let info = generative_info(d.doc.layer(nid).unwrap()).unwrap();
    assert_eq!(info.prompt, "a kite");
    let orig = generative_info(d.doc.layer(LayerId(r["layer"].as_u64().unwrap())).unwrap()).unwrap();
    assert_eq!(info.rect, orig.rect, "where it was made");
    assert!(s.journal.iter().any(|(c, _)| c == PLACE), "journaled");
    // Close everything: Place makes a document of the result's size.
    while let Some(i) = s.active_index() {
        s.close(i);
    }
    let p = s.execute(PLACE, json!({"id": id})).unwrap();
    assert_eq!(p["document"], 0);
    let d = s.active().unwrap();
    assert_eq!((u64::from(d.doc.size.width), u64::from(d.doc.size.height)), (w, h));
    assert_eq!(generative_info(&d.doc.layers[0]).unwrap().rect, [0, 0, w as i32, h as i32]);
    // Errors.
    assert!(s.execute(PLACE, json!({"id": id, "index": 3})).is_err());
    assert!(s.execute(PLACE, json!({"id": 12345})).is_err());
    assert!(matches!(s.execute(PLACE, json!({"id": id, "target": "canvas"})), Err(EngineError::BadParams { .. })));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn recreate_varies_in_place_while_the_layer_exists_and_falls_back_to_an_image_when_the_document_closed() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("recreate");
    let mut s = session(&fake.url, Some(&dir));
    s.execute(FILL, json!({"prompt": "a kite", "seed": 5})).unwrap();
    let id = jobs(&mut s)[0]["id"].as_u64().unwrap();
    // The result layer is open: Generate Similar, recorded as that.
    let r = s.execute(RECREATE, json!({"id": id, "seed": 6})).unwrap();
    assert_eq!(r["seed"], 6);
    let list = jobs(&mut s);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["command"], SIMILAR, "the row names the command that ran");
    assert_eq!(list[0]["label"].as_str().unwrap(), "Generate Similar");
    assert!(s.journal.iter().any(|(c, _)| c == RECREATE));
    // Closed: a fill becomes a Generate Image of its prompt in a new document.
    while let Some(i) = s.active_index() {
        s.close(i);
    }
    let r = s.execute(RECREATE, json!({"id": id, "seed": 7})).unwrap();
    assert_eq!(r["fallback"], "image");
    assert!(r.get("document").is_some());
    let list = jobs(&mut s);
    assert_eq!(list[0]["command"], IMAGE);
    assert_eq!(list[0]["prompt"], "a kite");
    assert_eq!(list[0]["seed"], 7);
    let st = fake.state();
    assert_eq!(st.runs.len(), 3);
    drop(st);
    // An image whose layer is open varies in place too; once closed it runs as an image again.
    let img_id = list[0]["id"].as_u64().unwrap();
    s.execute(RECREATE, json!({"id": img_id, "seed": 8})).unwrap();
    assert_eq!(jobs(&mut s)[0]["command"], SIMILAR);
    while let Some(i) = s.active_index() {
        s.close(i);
    }
    let r = s.execute(RECREATE, json!({"id": img_id, "seed": 9})).unwrap();
    assert!(r.get("document").is_some(), "a new document: {r}");
    assert_eq!(jobs(&mut s)[0]["command"], IMAGE);
    assert!(s.execute(RECREATE, json!({"id": 999})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn recreate_of_an_expand_needs_its_document_and_of_an_edit_falls_back() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url, None);
    s.execute("select.deselect", json!({})).unwrap();
    s.execute(EXPAND, json!({"right": 16, "seed": 1})).unwrap();
    let expand = jobs(&mut s)[0]["id"].as_u64().unwrap();
    s.execute(EDIT, json!({"prompt": "make it night", "seed": 2})).unwrap();
    let edit = jobs(&mut s)[0]["id"].as_u64().unwrap();
    while let Some(i) = s.active_index() {
        s.close(i);
    }
    let e = s.execute(RECREATE, json!({"id": expand})).unwrap_err().to_string();
    assert!(e.contains("closed"), "{e}");
    let r = s.execute(RECREATE, json!({"id": edit})).unwrap();
    assert_eq!(r["fallback"], "image");
}

#[test]
fn export_names_files_from_the_pattern_and_never_replaces_one() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("export");
    let out = temp_dir("export-out");
    let mut s = session(&fake.url, Some(&dir));
    s.execute(FILL, json!({"prompt": "a red kite!", "seed": 42, "variations": 2})).unwrap();
    let id = jobs(&mut s)[0]["id"].as_u64().unwrap();
    let template = jobs(&mut s)[0]["template"].as_str().unwrap().replace('/', "-");
    let r = s.execute(EXPORT, json!({"id": id, "dir": out.to_string_lossy(), "name": "{prompt}_{seed}"})).unwrap();
    let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(files.len(), 2);
    assert!(files[0].ends_with("a-red-kite_42_1.png"), "{}", files[0]);
    assert!(files[1].ends_with("a-red-kite_42_2.png"), "{}", files[1]);
    // Again: new names, not replacements.
    let r = s.execute(EXPORT, json!({"id": id, "index": 0, "dir": out.to_string_lossy(), "name": "{prompt}_{seed}_{n}"})).unwrap();
    assert!(r["files"][0].as_str().unwrap().ends_with("a-red-kite_42_1-1.png"), "{}", r["files"][0]);
    // The default pattern.
    let r = s.execute(EXPORT, json!({"id": id, "index": 1, "dir": out.to_string_lossy()})).unwrap();
    let name = Path::new(r["files"][0].as_str().unwrap()).file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.starts_with(&format!("{template}_")) && name.ends_with("_42.png"), "{name}");
    assert!(matches!(s.execute(EXPORT, json!({"id": id})), Err(EngineError::BadParams { .. })), "no folder anywhere");
    assert!(matches!(s.execute(EXPORT, json!({"id": id, "dir": out.to_string_lossy(), "name": "../x"})), Err(EngineError::BadParams { .. })));
    // The preferences: the folder and the naming when the call leaves them out.
    let pref_out = temp_dir("export-pref");
    s.execute("prefs.set", json!({"path": "integrations.saveResultsDir", "value": pref_out.to_string_lossy()})).unwrap();
    s.execute("prefs.set", json!({"path": "integrations.saveResultsNaming", "value": "kite-{seed}-{n}"})).unwrap();
    let r = s.execute(EXPORT, json!({"id": id})).unwrap();
    let names: Vec<String> =
        r["files"].as_array().unwrap().iter().map(|v| Path::new(v.as_str().unwrap()).file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["kite-42-1.png", "kite-42-2.png"]);
    assert!(Path::new(r["files"][0].as_str().unwrap()).starts_with(&pref_out));
    for bad in ["no-token", "a/{seed}", "..{seed}", ""] {
        assert!(s.execute("prefs.set", json!({"path": "integrations.saveResultsNaming", "value": bad})).is_err(), "{bad}");
    }
    assert!(s.execute("prefs.set", json!({"path": "integrations.saveResultsDir", "value": "a\u{0}b"})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&pref_out);
}

#[test]
fn every_library_command_refuses_bad_params_without_panicking() {
    let mut s = Session::new();
    for (cmd, p) in [
        (JOBS, json!({"limit": "x"})),
        (JOBS, json!({"limit": 0})),
        (JOBS, json!({"includeDismissed": 3})),
        (DISMISS, json!({"which": 5})),
        (DISMISS, json!({"id": 1})),
        (PIN, json!({})),
        (PIN, json!({"id": "x"})),
        (PIN, json!({"id": 1, "pinned": "yes"})),
        (CLEAR, json!({})),
        (CLEAR, json!({"which": null})),
        (RECREATE, json!({})),
        (RECREATE, json!({"id": 1, "seed": -3})),
        (RECREATE, json!({"id": 1, "variations": 99})),
        (PLACE, json!({"id": -1})),
        (PLACE, json!({"id": 1, "index": 1e9})),
        (EXPORT, json!({"id": 1})),
        (EXPORT, json!({"id": 1, "dir": ""})),
        (EXPORT, json!({"id": u64::MAX, "dir": "out", "name": "a/b"})),
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} {p}");
    }
    assert!(s.execute(CLEAR, json!({"which": "all"})).is_ok(), "nothing to clear is fine");
}

#[test]
fn date_stamps_are_utc() {
    assert_eq!(date_stamp(0), "1970-01-01_000000");
    assert_eq!(date_stamp(1_700_000_000_000), "2023-11-14_221320");
    assert_eq!(date_stamp(951_782_400_000), "2000-02-29_000000");
    assert_eq!(thumb_size(512, 256), (128, 64));
    assert_eq!(thumb_size(40, 30), (40, 30));
}

#[test]
fn two_processes_share_the_index_and_keep_each_others_rows() {
    let fake = FakeNative::start().unwrap();
    let dir = temp_dir("merge");
    let mut a = session(&fake.url, Some(&dir));
    a.execute(FILL, json!({"prompt": "a kite"})).unwrap();
    a.flush_generations();
    // "Another process" (the CLI) opens the same library, adds a row and writes.
    let mut b = session(&fake.url, Some(&dir));
    assert_eq!(jobs(&mut b).len(), 1);
    b.execute(FILL, json!({"prompt": "a boat"})).unwrap();
    b.flush_generations();
    // The first sees it on its next listing and keeps it when it writes.
    let list = jobs(&mut a);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["prompt"], "a boat");
    assert_ne!(list[0]["id"], list[1]["id"]);
    a.execute(DISMISS, json!({"id": list[1]["id"]})).unwrap();
    a.flush_generations();
    let mut c = Session::new();
    c.set_generations_dir(dir.clone());
    assert_eq!(c.execute(JOBS, json!({"includeDismissed": true})).unwrap()["total"], 2);
    // A row one process cleared is not brought back by merging an older index.
    a.execute(CLEAR, json!({"which": "all"})).unwrap();
    b.flush_generations();
    assert_eq!(jobs(&mut a).len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_a_folder_the_library_lives_in_memory() {
    let fake = FakeNative::start().unwrap();
    let mut s = session(&fake.url, None);
    s.execute(FILL, json!({"prompt": "a kite"})).unwrap();
    let j = &jobs(&mut s)[0];
    assert_eq!(j["status"], "done");
    assert_eq!(j["result"]["files"], json!([]));
    assert_eq!(j["result"]["thumbnail"], "");
    let id = j["id"].as_u64().unwrap();
    let e = s.execute(PLACE, json!({"id": id})).unwrap_err().to_string();
    assert!(e.contains("no stored result"), "{e}");
    s.flush_generations();
}
