//! `QIR_PROFILE=1`: time the DiT's linears and attention by kind, the device synchronised
//! around each (which slows the run a little; the shares are what it is for). [`report`] logs
//! the totals and the rate each kind reached, then clears them. Off, it costs one atomic read.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::Device;

use crate::Result;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_PROFILE").is_ok_and(|v| v.trim() == "1"))
}

/// Time a pipeline phase without synchronizing every kernel (`QIR_PHASE_PROFILE=1`).
pub fn phase<T>(kind: &str, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| enabled() || std::env::var("QIR_PHASE_PROFILE").is_ok_and(|v| v == "1")) {
        return f();
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    tracing::info!(phase = kind, seconds = t0.elapsed().as_secs_f64(), "pipeline phase");
    Ok(out)
}

/// Per kind: calls, seconds, floating-point operations.
static TOTALS: Mutex<Vec<(&'static str, u64, f64, f64)>> = Mutex::new(Vec::new());

/// Run `f` as one `kind` operation of `flops` operations.
pub fn scope<T>(kind: &'static str, flops: f64, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled() {
        return f();
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    let mut g = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    match g.iter_mut().find(|e| e.0 == kind) {
        Some(e) => {
            e.1 += 1;
            e.2 += dt;
            e.3 += flops;
        }
        None => g.push((kind, 1, dt, flops)),
    }
    Ok(out)
}

/// Log and clear the totals (nothing when profiling is off).
pub fn report(what: &str) {
    if !enabled() {
        return;
    }
    let mut g = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    for (kind, calls, secs, flops) in g.drain(..) {
        tracing::info!(what, kind, calls, secs = format!("{secs:.3}"), tflops = format!("{:.0}", flops / secs.max(1e-9) / 1e12), "profile");
    }
}
