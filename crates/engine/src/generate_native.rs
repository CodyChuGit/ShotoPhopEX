//! The native worker's lifecycle: the shared backend, its endpoint, status, start, warm-up,
//! stop and purge.

use std::sync::Arc;

use photocraft_genai::GenerativeBackend;
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

use super::{default_image_template, gen_err, speed};

/// The native engine's backend for this process: one worker (`photocraft --engine …`), started
/// on first use, shared by every command (plan-native-engine §3.4). A change of the runtime or
/// model folders (`integrations.runtimePath`, `integrations.modelsPath`) stops the worker so the
/// next call starts one with the new flags.
#[cfg(not(target_arch = "wasm32"))]
pub(super) static NATIVE: std::sync::OnceLock<std::sync::Mutex<Option<Arc<photocraft_genai::native::NativeBackend>>>> = std::sync::OnceLock::new();

/// SAM 3.1's precision on the worker's GPU: `f16` (1.9x the speed of f32; select by text on a
/// new picture 273–278 ms through the app against the retired ONNX Runtime route's 249–346 ms, every mask at an
/// IoU of at least 0.993 against ComfyUI's own f32 node: genai-native gn-segment F16.md), or
/// `f32` with `PHOTOCRAFT_SAM_PRECISION=f32` (equal to ComfyUI's within 3 pixels). Not yet a
/// preference: the owner decides whether it gets one.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn sam_precision() -> &'static str {
    match std::env::var("PHOTOCRAFT_SAM_PRECISION").ok().as_deref().map(str::trim) {
        Some(p) if p.eq_ignore_ascii_case("f32") => "f32",
        _ => "f16",
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn native_endpoint(spec: &str) -> Result<(u16, String)> {
    let invalid = || EngineError::Other("PHOTOCRAFT_NATIVE_ENGINE must be port:token (a nonzero port and nonempty token)".into());
    let (port, token) = spec.trim().split_once(':').ok_or_else(invalid)?;
    let port = port.trim().parse::<u16>().ok().filter(|p| *p != 0).ok_or_else(invalid)?;
    let token = token.trim();
    if token.is_empty() || token.len() > 512 {
        return Err(invalid());
    }
    Ok((port, token.to_string()))
}

/// `integrations.previewWhileGenerating` as the backend takes it; an unknown spelling (a
/// hand-edited file) reads as the default, `fast`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn preview_mode(prefs: &crate::prefs::Integrations) -> photocraft_genai::PreviewMode {
    photocraft_genai::PreviewMode::parse(&prefs.preview_while_generating).unwrap_or(photocraft_genai::PreviewMode::Fast)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_backend(s: &Session) -> Result<Arc<dyn GenerativeBackend>> {
    let integrations = &s.prefs().integrations;
    let timeout = std::time::Duration::from_secs(u64::from(integrations.generative_timeout_secs.max(5)));
    let preview = preview_mode(integrations);
    // Speed Fast and Balanced take 4-bit weights where the worker can (Krea 2's NVFP4 file on an
    // RTX 50-series card); Full keeps the full models.
    let nvfp4 = speed(s) != "best";
    // The deadline, the preview mode and the precision are read at each run: a change takes
    // effect on the next generation without restarting the worker.
    let configure = |b: &photocraft_genai::native::NativeBackend| {
        b.set_timeout(timeout);
        b.set_preview(preview);
        b.set_nvfp4(nvfp4);
    };
    if let Some(b) = photocraft_genai::native::test_backend(&integrations.models_root()).map_err(gen_err)? {
        configure(&b);
        return Ok(b);
    }
    let endpoint = match std::env::var("PHOTOCRAFT_NATIVE_ENGINE") {
        Ok(spec) => Some(native_endpoint(&spec)?),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err(EngineError::Other("PHOTOCRAFT_NATIVE_ENGINE must be valid Unicode port:token".into())),
    };
    if !integrations.native_configured() && endpoint.is_none() {
        let what = if integrations.models_root().is_empty() {
            "choose your Models folder (the folder with diffusion_models, text_encoders, vae and loras) in Preferences › AI Integrations first"
        } else {
            "set the runtime folders in Preferences › AI Integrations first"
        };
        return Err(EngineError::Other(what.into()));
    }
    let mut wanted = photocraft_genai::native::worker_args(&integrations.runtime_folders(), &integrations.models_root());
    if !wanted.is_empty() {
        wanted.extend(["--sam-precision".to_string(), sam_precision().to_string()]);
    }
    let slot = NATIVE.get_or_init(|| std::sync::Mutex::new(None));
    let mut g = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = g.as_ref() {
        // The flags after `--engine` are the preferences; a change restarts the worker. A
        // worker this process joined rather than started is kept as it is.
        if b.is_attached() || b.args().get(1..).unwrap_or(&[]) == wanted.as_slice() {
            configure(b);
            return Ok(Arc::clone(b) as Arc<dyn GenerativeBackend>);
        }
        b.stop();
        *g = None;
    }
    // `PHOTOCRAFT_NATIVE_ENGINE=port:token` joins a worker already running (one started by
    // hand with `photocraft-cli --engine …` and `GN_ENGINE_TOKEN`), so a bench of many CLI
    // processes shares one resident set of models instead of loading them in every process.
    if let Some((port, token)) = endpoint {
        let b = Arc::new(photocraft_genai::native::NativeBackend::attached(port, &token).map_err(gen_err)?);
        configure(&b);
        *g = Some(Arc::clone(&b));
        return Ok(b);
    }
    let dirs = photocraft_genai::native::runtime_dirs(&integrations.runtime_folders());
    let b = photocraft_genai::native::NativeBackend::this_executable(&wanted).map_err(gen_err)?.with_path_dirs(dirs);
    let b = Arc::new(b.with_log(crate::generate_engine_cmds::native_log_path()));
    configure(&b);
    *g = Some(Arc::clone(&b));
    Ok(b)
}

/// The worker this process holds now, if any (never starts one).
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn native_current(s: &Session) -> Option<Arc<photocraft_genai::native::NativeBackend>> {
    if let Ok(Some(b)) = photocraft_genai::native::test_backend(&s.prefs().integrations.models_root()) {
        return Some(b);
    }
    NATIVE.get().and_then(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

/// The native worker status:
/// `unconfigured` without a runtime and a model folder, else the worker's own state
/// (`NativeBackend::status`: stopped / starting / running / ready / failed), without waiting
/// and without starting one.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_status(s: &Session) -> Value {
    let log = crate::generate_engine_cmds::native_log_path().to_string_lossy().into_owned();
    let mode = "native";
    let state = |st: &str| json!({"state": st, "mode": mode, "log": log});
    if !s.prefs().integrations.native_configured() && native_current(s).is_none_or(|b| !b.is_attached()) {
        return state("unconfigured");
    }
    let st = native_current(s).map(|b| b.status()).unwrap_or_else(|| photocraft_genai::native::NativeStatus { state: "stopped".into(), ..Default::default() });
    let mut v = serde_json::to_value(st).unwrap_or_else(|_| json!({"state": "failed"}));
    v["mode"] = json!(mode);
    v["log"] = json!(log);
    v
}

/// Start the worker on a thread of its own (the call returns at once; the status says
/// `starting`, then `ready`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_start(s: &Session) -> Result<()> {
    if !s.prefs().integrations.native_configured() {
        return Err(EngineError::Other("set the runtime and model folders in Preferences › AI Integrations first".into()));
    }
    let b = native_backend(s)?;
    std::thread::spawn(move || {
        let _ = b.health();
    });
    Ok(())
}

/// The card memory a warm-up wants free before it loads `template`'s models: what that family
/// holds resident on the RTX 5090 plus 2 GB for the first generation's working set (measured
/// 2026-10-29 from the worker's free memory before and after a warm-up).
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn warm_need(template: &str) -> u64 {
    const GB: u64 = 1_000_000_000;
    match template.split('/').next().unwrap_or("") {
        t if t.starts_with("krea2") => WARM_KREA2_GB * GB,
        t if t.starts_with("moss") => WARM_MOSS_GB * GB,
        t if t.starts_with("p-h3") || t.starts_with("minimax-h3") => WARM_H3_GB * GB,
        _ => WARM_QWEN21_GB * GB,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) const WARM_KREA2_GB: u64 = 20; // 18.2 GB resident

#[cfg(not(target_arch = "wasm32"))]
pub(super) const WARM_QWEN21_GB: u64 = 27; // 25.0 GB resident

/// MOSS in the worker's 12 GB mode keeps one model at a time (11.4–12.0 GiB peak, 2026-11-05):
/// a warm-up on a 12 GB card wants most of it free.
#[cfg(not(target_arch = "wasm32"))]
pub(super) const WARM_MOSS_GB: u64 = 10;

/// P-H3-Image: the int8 DiT (21.2 GB) stays, the encoder or the VAE (5–6 GB) beside it (an
/// estimate until measured on the 5090).
#[cfg(not(target_arch = "wasm32"))]
pub(super) const WARM_H3_GB: u64 = 28;

/// Start the worker and load the default image template's models (`generate.engine.warmUp`):
/// on a thread of its own, or with `wait` before returning (the bench). It is skipped on a card
/// without the room ([`warm_need`]); any generation that starts meanwhile cancels it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_warm_up(s: &Session, wait: bool) -> Result<Value> {
    use photocraft_genai::native::WarmUp;
    let template = default_image_template(s);
    photocraft_genai::template::find(&template).map_err(gen_err)?;
    native_backend(s)?;
    let b = native_current(s).ok_or_else(|| EngineError::Other("no native worker".into()))?;
    let need = warm_need(&template);
    let warm = move || {
        let started = std::time::Instant::now();
        let r = b.warm_up(&template, need);
        let ms = started.elapsed().as_millis() as u64;
        match r {
            Ok(WarmUp::Warm) => json!({"template": template, "state": "warm", "ms": ms}),
            Ok(WarmUp::Skipped(why)) => json!({"template": template, "state": "skipped", "detail": why, "ms": ms}),
            Ok(WarmUp::Cancelled) => json!({"template": template, "state": "cancelled", "ms": ms}),
            Err(e) => json!({"template": template, "state": "failed", "detail": e.to_string(), "ms": ms}),
        }
    };
    if wait {
        return Ok(warm());
    }
    std::thread::Builder::new()
        .name("warm-up".into())
        .spawn(move || drop(warm()))
        .map_err(|e| EngineError::Other(format!("could not start the warm-up: {e}")))?;
    Ok(json!({"state": "started"}))
}

/// Stop the worker this process started (the next generative command starts a fresh one).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_stop(s: &Session) {
    if let Some(b) = native_current(s) {
        b.stop();
    }
}

/// Unload the worker's models, if it runs.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_purge(s: &Session) -> Result<()> {
    match native_current(s) {
        Some(b) => b.purge().map_err(gen_err),
        None => Ok(()),
    }
}

/// The native worker is the only backend. The web build has none yet.
pub(crate) fn backend(s: &Session) -> Result<Arc<dyn GenerativeBackend>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native_backend(s)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        Err(EngineError::Other("generative features are not available in the web build yet".into()))
    }
}

pub(crate) fn backend_for(s: &Session, _template: &str) -> Result<(Arc<dyn GenerativeBackend>, String)> {
    Ok((backend(s)?, "native".to_string()))
}

pub(crate) fn web_unavailable() -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") { Err("not available in the web build yet".into()) } else { Ok(()) }
}
