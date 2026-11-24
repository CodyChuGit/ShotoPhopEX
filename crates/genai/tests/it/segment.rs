//! Segmentation templates through the client: one result image per instance, template-specific
//! parameters bound from `Request::params`.
#![cfg(all(feature = "fake-server", not(target_arch = "wasm32")))]

use photocraft_genai::fake::{FakeNative, Options};
use photocraft_genai::native::test_backend;
use photocraft_genai::{Error, GenerativeBackend, NoProgress, Request, Rgba8};
use serde_json::json;

fn request() -> Request {
    let mut r = Request::new("sam3.1/segment", "the lighthouse");
    r.image = Some(Rgba8::solid(40, 24, [10, 20, 30, 255]).unwrap_or_else(|e| panic!("{e}")));
    r.params.insert("threshold".into(), json!(0.4));
    r
}

#[test]
fn a_detector_run_returns_one_mask_per_instance() {
    let fake =
        FakeNative::start_with(Options { segments: vec![[0.0, 0.0, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]], ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let b = test_backend(&fake.url).unwrap().unwrap();
    let resp = b.run(&request(), &NoProgress).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(resp.images.len(), 2);
    let first = &resp.images[0];
    assert_eq!((first.width, first.height), (40, 24));
    assert_eq!(first.get(5, 5).map(|p| p[0]), Some(255), "inside the first instance");
    assert_eq!(first.get(30, 5).map(|p| p[0]), Some(0), "outside it");
    assert_eq!(resp.images[1].get(30, 20).map(|p| p[0]), Some(255));
    let st = fake.state();
    let g = &st.runs[0].request;
    assert_eq!(g["task"], "segment");
    assert_eq!(g["params"]["threshold"], 0.4, "bound from Request::params");
    assert_eq!(g["prompt"], "the lighthouse");
    assert_eq!(st.runs[0].blobs.len(), 1);
}

#[test]
fn nothing_found_is_no_output() {
    let fake = FakeNative::start_with(Options { segments: vec![], ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let b = test_backend(&fake.url).unwrap().unwrap();
    assert!(matches!(b.run(&request(), &NoProgress), Err(Error::NoOutput(_))));
}

#[test]
fn standard_bindings_win_over_params() {
    let fake = FakeNative::start().unwrap_or_else(|e| panic!("{e}"));
    let b = test_backend(&fake.url).unwrap().unwrap();
    let mut r = request();
    r.params.insert("prompt".into(), json!("not this"));
    b.run(&r, &NoProgress).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fake.state().runs[0].request["prompt"], "the lighthouse");
}
