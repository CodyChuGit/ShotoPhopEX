//! Native worker lifecycle commands. Status and purge never start a worker.
#[cfg(target_arch = "wasm32")]
use crate::EngineError;
use crate::commands::CommandSpec;
use crate::generate_cmds::web_unavailable;
use crate::{Result, Session};
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use serde_json::json;
pub const STATUS: &str = "generate.engine.status";
pub const START: &str = "generate.engine.start";
pub const STOP: &str = "generate.engine.stop";
pub const RESTART: &str = "generate.engine.restart";
pub const PURGE: &str = "generate.engine.purge";
pub const WARM_UP: &str = "generate.engine.warmUp";
/// Native worker stderr, reset when a worker starts.
pub fn native_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("photocraft-native-engine.log")
}
fn run_status(s: &mut Session, _p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(json!({"native": crate::generate_cmds::native_status(s)}))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        Err(EngineError::Other("not available in the web build".into()))
    }
}
fn run_start(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    crate::generate_cmds::native_start(s)?;
    run_status(s, p)
}
fn run_stop(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    crate::generate_cmds::native_stop(s);
    run_status(s, p)
}
fn run_restart(s: &mut Session, p: &Value) -> Result<Value> {
    run_stop(s, p)?;
    run_start(s, p)
}
fn run_purge(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    crate::generate_cmds::native_purge(s)?;
    run_status(s, p)
}
fn run_warm_up(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::generate_cmds::native_warm_up(s, p.get("wait").and_then(Value::as_bool).unwrap_or(false))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
}
/// At launch: the warm-up when `integrations.warmUpAtLaunch` asks for it and the folders are
/// set; `None` otherwise. Never fails the launch.
pub fn warm_up_at_launch(s: &mut Session) -> Option<Value> {
    if !s.prefs().integrations.warm_up_at_launch || when_configured(s).is_err() {
        return None;
    }
    s.execute(WARM_UP, serde_json::json!({})).ok()
}
fn always(_s: &Session) -> std::result::Result<(), String> {
    web_unavailable()
}
fn when_configured(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    if !s.prefs().integrations.native_configured() {
        return Err("set the runtime and Models folders in Preferences › AI Integrations first".into());
    }
    Ok(())
}
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: STATUS,
            label: "Engine Status",
            menu: &[],
            shortcut: None,
            params: r#"{} → {native:{state,pid,engine,device,vramTotal,vramFree,resident,detail,log}}; does not start the worker"#,
            enabled: always,
            run: run_status,
            journal: false,
        },
        CommandSpec {
            id: START,
            label: "Start Engine",
            menu: &[],
            shortcut: None,
            params: "{} → native worker status; starts asynchronously",
            enabled: when_configured,
            run: run_start,
            journal: false,
        },
        CommandSpec {
            id: STOP,
            label: "Stop Engine",
            menu: &[],
            shortcut: None,
            params: "{} → native worker status; next generation starts a fresh worker",
            enabled: always,
            run: run_stop,
            journal: false,
        },
        CommandSpec {
            id: RESTART,
            label: "Restart Engine",
            menu: &[],
            shortcut: None,
            params: "{} → native worker status; stops then starts the worker",
            enabled: when_configured,
            run: run_restart,
            journal: false,
        },
        CommandSpec {
            id: PURGE,
            label: "Purge Engine Models",
            menu: &[],
            shortcut: None,
            params: "{} → native worker status; unloads resident models without starting a worker",
            enabled: always,
            run: run_purge,
            journal: false,
        },
        CommandSpec {
            id: WARM_UP,
            label: "Warm Up AI Models",
            menu: &[],
            shortcut: None,
            params: r#"{"wait"?:bool} → {state:started|warm|skipped|cancelled|failed,template,detail?,ms?}; starts the worker and loads the default image template's models (in the background unless wait), skipped when the card's free memory is short, cancelled by any generation that starts meanwhile"#,
            enabled: when_configured,
            run: run_warm_up,
            journal: false,
        },
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn status_is_native_only_and_start_requires_folders() {
        let mut s = Session::new();
        let st = s.execute(STATUS, json!({})).unwrap();
        assert_eq!(st["native"]["state"], "unconfigured");
        assert!(st.get("state").is_none());
        assert!(s.execute(START, json!({})).is_err());
        assert!(s.execute(PURGE, json!({})).is_ok());
        assert!(s.execute(STOP, json!({})).is_ok());
    }

    fn fake_session(options: photocraft_genai::fake::Options) -> (photocraft_genai::fake::FakeNative, Session) {
        let fake = photocraft_genai::fake::FakeNative::start_with(options).unwrap();
        let mut s = Session::new();
        s.edit_prefs(|p| {
            p.integrations.models_path = fake.url.clone();
            p.integrations.runtime_path = "fake".into();
        });
        (fake, s)
    }

    /// A warm-up loads the default image template's models and generates nothing; with too little
    /// free memory on the card it loads nothing.
    #[test]
    fn a_warm_up_loads_the_default_image_models_when_the_card_has_room() {
        let (fake, mut s) = fake_session(photocraft_genai::fake::Options { vram_free: Some(40_000_000_000), ..Default::default() });
        let r = s.execute(WARM_UP, json!({"wait": true})).unwrap();
        assert_eq!((r["state"].as_str(), r["template"].as_str()), (Some("warm"), Some("krea2-turbo/image")), "{r}");
        {
            let st = fake.state();
            assert_eq!(st.runs.len(), 1);
            assert_eq!(st.runs[0].run.params.get("warmup"), Some(&json!(true)));
        }
        let (fake, mut s) = fake_session(photocraft_genai::fake::Options { vram_free: Some(4_000_000_000), ..Default::default() });
        let r = s.execute(WARM_UP, json!({"wait": true})).unwrap();
        assert_eq!(r["state"], "skipped", "{r}");
        assert!(r["detail"].as_str().unwrap().contains("4.0 GB free"), "{r}");
        assert_eq!(fake.state().runs[0].run.params.get("need_free"), Some(&json!(20_000_000_000u64)), "the worker checks the card against the family's size");
        // Off in the preferences: nothing at launch.
        s.edit_prefs(|p| p.integrations.warm_up_at_launch = false);
        assert!(warm_up_at_launch(&mut s).is_none());
    }

    /// A generation that starts while the warm-up loads cancels it and goes next, without
    /// waiting for the rest of the warm-up.
    #[test]
    fn a_generation_cancels_a_warm_up_in_flight() {
        use std::time::{Duration, Instant};
        let (fake, mut s) =
            fake_session(photocraft_genai::fake::Options { vram_free: Some(40_000_000_000), delay_ms: 3_000, progress_steps: 300, ..Default::default() });
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        assert_eq!(warm_up_at_launch(&mut s).unwrap()["state"], "started");
        let t = Instant::now();
        while fake.state().runs.is_empty() {
            assert!(t.elapsed() < Duration::from_secs(10), "the warm-up never reached the worker");
            std::thread::sleep(Duration::from_millis(5));
        }
        let t = Instant::now();
        s.execute(
            "generate.image",
            json!({"prompt": "a lighthouse", "template": "krea2-turbo/image", "target": "layer", "width": 64, "height": 64, "enhance": false}),
        )
        .unwrap();
        // The generation's own 3 s, plus at most the warm-up's cancel round trip.
        assert!(t.elapsed() < Duration::from_millis(5_000), "waited {:?} for the warm-up", t.elapsed());
        let st = fake.state();
        assert_eq!(st.cancellations, 1, "the warm-up was told to stop");
        assert_eq!(st.runs.len(), 2);
    }
}
