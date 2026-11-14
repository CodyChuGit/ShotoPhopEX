//! Opt-in timing of the flow model's parts (`QIR_IDOL_PROFILE=1`): each timed part waits for the
//! device before and after, so the totals are true GPU time per part (and the run is slower).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use candle_core::Device;

thread_local! {
    static TOTALS: RefCell<BTreeMap<&'static str, (Duration, u64)>> = const { RefCell::new(BTreeMap::new()) };
}

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QIR_IDOL_PROFILE").is_ok_and(|v| v == "1"))
}

/// Run `f`, adding its device time to `name` when profiling.
pub fn time<T>(dev: &Device, name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let _ = dev.synchronize();
    let t = Instant::now();
    let out = f();
    let _ = dev.synchronize();
    let d = t.elapsed();
    TOTALS.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry(name).or_insert((Duration::ZERO, 0));
        e.0 += d;
        e.1 += 1;
    });
    out
}

/// Print and clear the totals.
pub fn report() {
    if !enabled() {
        return;
    }
    TOTALS.with(|m| {
        let mut m = m.borrow_mut();
        let all: f64 = m.values().map(|v| v.0.as_secs_f64()).sum();
        eprintln!("idol profile (device time, synced per part):");
        let mut rows: Vec<_> = m.iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (k, (d, n)) in rows {
            eprintln!("  {k:<14} {:8.3} s  {:5.1}%  {n} calls", d.as_secs_f64(), 100.0 * d.as_secs_f64() / all.max(1e-9));
        }
        m.clear();
    });
}
