//! Opt-in stage timings (`GN_MLX_PROFILE=1`): each timed stage is evaluated and synchronized, so
//! the totals attribute device time to stages (and lose the overlap a single eval keeps).
//! Off, a stage is the closure alone.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use mlx_rs::Array;

thread_local! {
    static TIMES: RefCell<Option<BTreeMap<String, (f64, u64)>>> = const { RefCell::new(None) };
}

pub fn enabled() -> bool {
    std::env::var("GN_MLX_PROFILE").is_ok_and(|v| v == "1")
}

/// Start collecting on this thread.
pub fn start() {
    if enabled() {
        TIMES.with(|t| *t.borrow_mut() = Some(BTreeMap::new()));
    }
}

/// Stage → (milliseconds, calls) since [`start`].
pub fn finish() -> BTreeMap<String, (f64, u64)> {
    TIMES.with(|t| t.borrow_mut().take().unwrap_or_default())
}

/// Run `f`, and when collecting, evaluate its result and add the time to `name`.
pub fn stage(name: &str, f: impl FnOnce() -> Result<Array>) -> Result<Array> {
    if !TIMES.with(|t| t.borrow().is_some()) {
        return f();
    }
    let t0 = Instant::now();
    let a = f()?;
    a.eval()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    TIMES.with(|t| {
        if let Some(m) = t.borrow_mut().as_mut() {
            let e = m.entry(name.into()).or_default();
            e.0 += ms;
            e.1 += 1;
        }
    });
    Ok(a)
}

/// Add `ms` to `name` when collecting (host-side work timed by the caller).
pub fn record(name: &str, ms: f64) {
    TIMES.with(|t| {
        if let Some(m) = t.borrow_mut().as_mut() {
            let e = m.entry(name.into()).or_default();
            e.0 += ms;
            e.1 += 1;
        }
    });
}
