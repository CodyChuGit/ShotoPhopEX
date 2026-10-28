//! Opt-in synchronized stage timings; disabled inference does not synchronize.
use anyhow::Result;
use candle_core::Device;
use std::{cell::RefCell, collections::BTreeMap, time::Instant};
thread_local! {
    static TIMES: RefCell<Option<BTreeMap<String, f64>>> = const { RefCell::new(None) };
}
/// Start collecting milliseconds on the current inference thread.
pub fn start() {
    TIMES.with(|t| *t.borrow_mut() = Some(BTreeMap::new()));
}
/// Finish collection and return accumulated milliseconds per stage.
pub fn finish() -> BTreeMap<String, f64> {
    TIMES.with(|t| t.borrow_mut().take().unwrap_or_default())
}
/// Attribute device work to a stage only when profiling is enabled.
pub(crate) fn measure<T>(device: &Device, name: &str, run: impl FnOnce() -> Result<T>) -> Result<T> {
    if !TIMES.with(|t| t.borrow().is_some()) {
        return run();
    }
    device.synchronize()?;
    let start = Instant::now();
    let value = run()?;
    device.synchronize()?;
    TIMES.with(|t| {
        if let Some(times) = t.borrow_mut().as_mut() {
            *times.entry(name.into()).or_default() += start.elapsed().as_secs_f64() * 1000.0;
        }
    });
    Ok(value)
}
