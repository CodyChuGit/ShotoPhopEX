//! Native-only routing and worker request regression tests.
use crate::Session;
use photocraft_genai::fake::FakeNative;
use serde_json::json;
#[test]
fn every_route_uses_the_native_worker() {
    let fake = FakeNative::start().unwrap();
    let mut s = Session::new();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
    });
    for id in ["qwen-2.1/image", "krea2-turbo/image", "qwen-edit-2511/fill", "sam3.1/segment", "enhance/edit"] {
        assert_eq!(s.execute("generate.health", json!({"template":id})).unwrap()["backend"], "native/fake");
    }
    s.execute("generate.image", json!({"prompt":"a fox", "width":64, "height":64, "seed":9})).unwrap();
    s.execute("generate.edit", json!({"prompt":"make it winter", "template":"qwen-2.1/edit", "enhance":true})).unwrap();
    let st = fake.state();
    assert_eq!(st.runs.len(), 3);
    assert_eq!(st.runs[0].run.model, "krea2-turbo/image");
    assert_eq!(st.runs[0].run.seed, 9);
    assert_eq!(st.runs[1].run.task, "text");
    assert_eq!(st.runs[2].run.model, "qwen-2.1/edit");
    assert!(st.runs[2].run.prompt.contains("Enhanced: make it winter"));
    assert_eq!(st.frees, 0, "worker manages residency; no purge before requests");
}

#[test]
fn lifecycle_status_purge_and_stop_address_only_the_native_worker() {
    let fake = FakeNative::start().unwrap();
    let mut s = Session::new();
    s.edit_prefs(|p| {
        p.integrations.models_path = fake.url.clone();
        p.integrations.runtime_path = "fake".into();
    });
    let st = s.execute("generate.engine.status", json!({})).unwrap();
    assert_eq!(st["native"]["state"], "ready");
    assert!(st.get("state").is_none());
    s.execute("generate.engine.purge", json!({})).unwrap();
    assert_eq!(fake.state().frees, 1);
    assert_eq!(s.execute("generate.engine.status", json!({})).unwrap()["native"]["state"], "ready");
    s.execute("generate.engine.stop", json!({})).unwrap();
    assert_eq!(s.execute("generate.engine.status", json!({})).unwrap()["native"]["state"], "stopped");
}

#[test]
fn an_invalid_external_endpoint_does_not_bypass_required_model_folders() {
    assert_eq!(super::native_endpoint("1234:token").unwrap(), (1234, "token".into()));
    for spec in ["", "missing", "0:token", "65536:token", "42:", "x:token"] {
        assert!(super::native_endpoint(spec).is_err(), "{spec}");
    }
}
