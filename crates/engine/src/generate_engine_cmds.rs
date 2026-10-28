//! `generate.engine.*`: the app-managed ComfyUI process (the sidecar, `photocraft_genai::sidecar`;
//! plan-native-engine §5.4). With Preferences › AI Integrations › ComfyUI folder set, the app
//! starts that install itself, hidden, when a generative command needs it or at launch
//! (`integrations.engineStartOnLaunch`), stops it on quit (`integrations.engineStopOnQuit`), and
//! shows its state in the status bar. A server started by hand at the same URL is used as it is.
//!
//! Preferences read here: `integrations.comfyPath`, `integrations.comfyArgs`,
//! `integrations.comfyServer` (the host and port the child binds), `integrations.engineStartOnLaunch`,
//! `integrations.engineStopOnQuit`.

use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::generate_cmds::web_unavailable;
use crate::{EngineError, Result, Session};

pub const STATUS: &str = "generate.engine.status";
pub const START: &str = "generate.engine.start";
pub const STOP: &str = "generate.engine.stop";
pub const RESTART: &str = "generate.engine.restart";
pub const PURGE: &str = "generate.engine.purge";

/// The file the engine's output goes to (under the system temp folder, like the model-set
/// memory `generate_cmds` keeps there).
pub fn log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("photocraft-engine.log")
}

/// The native worker's log (its stderr: the phase timings, a refused card, a fault), beside
/// the ComfyUI one; each start of a worker begins it afresh.
pub fn native_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("photocraft-native-engine.log")
}

/// The engine a start / stop / restart / purge names: `comfy` (the default, as before the
/// native worker had commands) or `native`; `all` for purge.
fn engine_param(p: &Value, default: &str) -> Result<String> {
    let e = p.get("engine").and_then(Value::as_str).unwrap_or(default).trim().to_string();
    match e.as_str() {
        "comfy" | "native" => Ok(e),
        "all" if default == "all" => Ok(e),
        _ => Err(EngineError::Other(format!("unknown engine `{e}` (comfy or native)"))),
    }
}

/// What the preferences say to launch; `None` when no folder is set.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn launch_from_prefs(s: &Session) -> Option<photocraft_genai::sidecar::Launch> {
    let i = &s.prefs().integrations;
    let root = i.comfy_path.trim();
    if root.is_empty() {
        return None;
    }
    Some(photocraft_genai::sidecar::Launch {
        root: std::path::PathBuf::from(root),
        url: i.comfy_server.trim().to_string(),
        args: i.comfy_args.split_whitespace().map(str::to_string).collect(),
        log: log_path(),
    })
}

/// Preferences › AI Integrations › Start engine at launch.
pub fn start_on_launch(s: &Session) -> bool {
    s.prefs().integrations.engine_start_on_launch && !s.prefs().integrations.comfy_path.trim().is_empty()
}

/// Preferences › AI Integrations › Stop engine on quit.
pub fn stop_on_quit(s: &Session) -> bool {
    s.prefs().integrations.engine_stop_on_quit
}

/// The status as JSON (the ComfyUI sidecar's, or the "unconfigured" shape when no folder is
/// set), with the native worker's under `native`.
#[cfg(not(target_arch = "wasm32"))]
fn status_json(s: &Session) -> Result<Value> {
    let mut v = match launch_from_prefs(s) {
        Some(l) => serde_json::to_value(photocraft_genai::sidecar::status(&l)).map_err(|e| EngineError::Other(e.to_string()))?,
        None => json!({"state": "unconfigured", "url": s.prefs().integrations.comfy_server.trim(), "detail": "", "log": log_path().to_string_lossy()}),
    };
    v["native"] = crate::generate_cmds::native_status(s);
    Ok(v)
}

#[cfg(not(target_arch = "wasm32"))]
fn configured(s: &Session) -> Result<photocraft_genai::sidecar::Launch> {
    launch_from_prefs(s).ok_or_else(|| EngineError::Other("no ComfyUI folder is set (Preferences › AI Integrations › ComfyUI folder)".into()))
}

fn run_status(s: &mut Session, _p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        status_json(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        Err(EngineError::Other("not available in the web build".into()))
    }
}

fn run_start(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if engine_param(p, "comfy")? == "native" {
            crate::generate_cmds::native_start(s)?;
        } else {
            let l = configured(s)?;
            photocraft_genai::sidecar::start(&l).map_err(EngineError::Other)?;
        }
        status_json(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
}

fn run_stop(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if engine_param(p, "comfy")? == "native" {
            crate::generate_cmds::native_stop();
        } else {
            photocraft_genai::sidecar::stop_if_owned();
        }
        status_json(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
}

fn run_restart(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if engine_param(p, "comfy")? == "native" {
            crate::generate_cmds::native_stop();
            crate::generate_cmds::native_start(s)?;
        } else {
            let l = configured(s)?;
            photocraft_genai::sidecar::stop_if_owned();
            photocraft_genai::sidecar::start(&l).map_err(EngineError::Other)?;
        }
        status_json(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
}

/// Purge: the native worker's models leave the card (the worker stays, ready); the ComfyUI
/// server is asked to unload its models when it answers. Nothing is started.
fn run_purge(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let which = engine_param(p, "all")?;
        if which != "comfy" {
            crate::generate_cmds::native_purge()?;
        }
        if which != "native" {
            let st = status_json(s)?;
            if matches!(st["state"].as_str(), Some("ready" | "external")) {
                crate::generate_cmds::backend(s)?.free().map_err(crate::generate_cmds::gen_err)?;
            }
        }
        status_json(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
}

fn always(_s: &Session) -> std::result::Result<(), String> {
    web_unavailable()
}

/// Start and Restart need an engine to start: the ComfyUI folder or the native folders.
fn when_configured(s: &Session) -> std::result::Result<(), String> {
    web_unavailable()?;
    let i = &s.prefs().integrations;
    if i.comfy_path.trim().is_empty() && !i.native_configured() {
        return Err("set the ComfyUI folder or the native runtime and model folders in Preferences › AI Integrations first".into());
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
            params: r#"{} → {"state":"unconfigured|stopped|starting|ready|external|failed","pid","url","root","log","detail","sinceMs","native":{"state":"off|unconfigured|stopped|starting|running|ready|failed","mode":"auto|native|comfy","pid","engine","device","vramTotal","vramFree","resident":[...],"detail","log"}} (the top level is the app-managed ComfyUI: unconfigured = no ComfyUI folder in Preferences › AI Integrations; external = a server started outside the app answers at the URL; failed = the child exited, detail has the log's last line. `native` is the native worker, asked without waiting and without starting it: off under engine comfy, unconfigured without the runtime and model folders, stopped until a generative command or generate.engine.start {engine:"native"} starts it, running while a request holds it, failed with the reason in detail (a start that failed, a card the build has no kernels for, missing runtime libraries); device is the card's name, vramFree is known while a model is loaded; a query, journal false)"#,
            enabled: always,
            run: run_status,
            journal: false,
        },
        CommandSpec {
            id: START,
            label: "Start Engine",
            menu: &[],
            shortcut: None,
            params: r#"{engine?: "comfy"|"native"} → as generate.engine.status (comfy, the default: spawns the ComfyUI install at Preferences › ComfyUI folder, hidden, bound to the host and port of Preferences › ComfyUI server, with Preferences › ComfyUI launch flags; returns at once with state starting, ready follows within 10–60 s; a server already answering is left alone (external); needs the folder set. native: starts the native worker on a thread (no model loads until a request asks); needs the runtime and model folders)"#,
            enabled: when_configured,
            run: run_start,
            journal: false,
        },
        CommandSpec {
            id: STOP,
            label: "Stop Engine",
            menu: &[],
            shortcut: None,
            params: r#"{engine?: "comfy"|"native"} → as generate.engine.status (comfy: stops the server the app started; one started by hand is left alone. native: ends the worker; the next generative command starts a fresh one)"#,
            enabled: always,
            run: run_stop,
            journal: false,
        },
        CommandSpec {
            id: RESTART,
            label: "Restart Engine",
            menu: &[],
            shortcut: None,
            params: r#"{engine?: "comfy"|"native"} → as generate.engine.status (stop, then start; the way to apply changed launch flags or recover a failed engine)"#,
            enabled: when_configured,
            run: run_restart,
            journal: false,
        },
        CommandSpec {
            id: PURGE,
            label: "Purge Engine Models",
            menu: &[],
            shortcut: None,
            params: r#"{engine?: "all"|"comfy"|"native"} → as generate.engine.status (unloads the models: the native worker's, if one runs, and the ComfyUI server's, if it answers; starts nothing; the worker stays and loads again on the next request)"#,
            enabled: always,
            run: run_purge,
            journal: false,
        },
    ]
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// The commands read the preferences: unconfigured without a folder, a sentence for a folder
    /// that is no install, `external` for a server already answering (nothing spawned), and the
    /// launch flags split into arguments.
    #[test]
    fn engine_commands_follow_the_preferences() {
        let fake = photocraft_genai::fake::FakeComfy::start().unwrap();
        let mut s = Session::new();
        s.edit_prefs(|p| p.integrations.set_comfy_route(fake.url.clone()));
        let st = s.execute(STATUS, json!({})).unwrap();
        assert_eq!(st["state"], "unconfigured");
        assert!(s.execute(START, json!({})).is_err(), "start needs a folder");
        assert!(!start_on_launch(&s) && stop_on_quit(&s));
        let d = std::env::temp_dir().join(format!("pex-engine-cmds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        s.edit_prefs(|p| {
            p.integrations.comfy_path = d.join("nowhere").to_string_lossy().into_owned();
            p.integrations.comfy_args = "--fast  fp8_matrix_mult --lowvram".into();
            p.integrations.engine_start_on_launch = true;
        });
        assert!(start_on_launch(&s));
        let l = launch_from_prefs(&s).unwrap();
        assert_eq!(l.args, vec!["--fast", "fp8_matrix_mult", "--lowvram"]);
        assert_eq!(l.url, fake.url);
        // A closed port and a folder that is no install: a sentence, nothing spawned.
        s.edit_prefs(|p| p.integrations.set_comfy_route("http://127.0.0.1:9"));
        assert_eq!(s.execute(STATUS, json!({})).unwrap()["state"], "stopped");
        let e = s.execute(START, json!({})).unwrap_err().to_string();
        assert!(e.contains("not a ComfyUI install"), "{e}");
        assert_eq!(s.execute(STATUS, json!({})).unwrap()["state"], "failed");
        // A server already answering at the URL is used as it is, whatever the folder holds.
        std::fs::write(d.join("main.py"), "").unwrap();
        s.edit_prefs(|p| {
            p.integrations.comfy_path = d.to_string_lossy().into_owned();
            p.integrations.set_comfy_route(fake.url.clone());
        });
        let st = s.execute(START, json!({})).unwrap();
        assert_eq!((st["state"].as_str(), st["pid"].is_null()), (Some("external"), true), "{st}");
        let st = s.execute(STOP, json!({})).unwrap();
        assert_eq!(st["state"], "external");
        let _ = std::fs::remove_dir_all(d);
    }
}
