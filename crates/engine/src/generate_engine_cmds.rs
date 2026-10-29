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
}
