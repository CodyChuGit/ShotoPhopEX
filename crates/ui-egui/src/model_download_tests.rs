use serde_json::json;

use super::*;

#[test]
fn a_model_finds_its_pack_or_the_version_that_runs_here() {
    let info = json!({
        "pack": {"id": "standard", "missingBytes": 0},
        "extras": [{"id": "low-vram", "missingBytes": 9}, {"id": "flux3-klein-q4", "missingBytes": 5}],
    });
    assert_eq!(pack_entry(&info, "standard").unwrap()["id"], "standard");
    assert_eq!(pack_entry(&info, "low-vram").unwrap()["id"], "low-vram");
    assert_eq!(pack_entry(&info, "flux3-klein").unwrap()["id"], "flux3-klein-q4", "only the 4-bit version runs on this system");
    assert!(pack_entry(&info, "qwen-edit-2511").is_none());
}

#[test]
fn only_the_missing_files_licences_are_shown() {
    let info = json!({"licenses": [{"id": "a"}, {"id": "b"}]});
    let entry = json!({"files": [{"license": "a", "installed": false, "gated": false}, {"license": "b", "installed": true, "gated": true}]});
    let ls = licences(&info, &entry);
    assert_eq!(ls.len(), 1);
    assert_eq!(ls[0]["id"], "a");
    assert!(!gated(&entry), "a gated file already here asks for no sign-in");
}
