//! Device selection. Picks CUDA when the `cuda` feature is built and a GPU is
//! present; otherwise CPU (the default build, used for Mac-side `cargo check`).

use candle_core::Device;

use crate::Result;

/// Best available device: CUDA:0 under the `cuda` feature, else CPU.
pub fn best_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => return Ok(d),
            Err(e) => {
                tracing::warn!("CUDA requested but unavailable ({e}); falling back to CPU");
            }
        }
    }
    Ok(Device::Cpu)
}

/// Make `device`'s CUDA context current on this thread. The worker serves each
/// client on a thread of its own, and the first CUDA call of a request may be a
/// free (the previous request's LoRA adapters dropped, a resident model let
/// go): cudarc frees without binding, the driver answers
/// `CUDA_ERROR_INVALID_CONTEXT`, and the error is handed to the request's next
/// call (a no-LoRA edit after a LoRA one failed in 0.3 s, 2026-10-27). No-op on
/// CPU.
pub fn bind_thread(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        let _ = d.cuda_stream().context().bind_to_thread();
    }
    let _ = device;
}

/// `(free, total)` device memory in bytes (cuMemGetInfo) on CUDA; `None` on
/// CPU or if the query fails.
pub fn free_vram(device: &Device) -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        use candle_core::cuda_backend::cudarc::driver::result::mem_get_info;
        d.cuda_stream().context().bind_to_thread().ok()?;
        return mem_get_info().ok();
    }
    let _ = device;
    None
}

/// The default CUDA memory pool of `device` (the one candle's
/// `cuMemAllocAsync` allocations come from), bound to this thread.
#[cfg(feature = "cuda")]
fn default_pool(
    d: &candle_core::CudaDevice,
) -> Option<candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let ctx = d.cuda_stream().context().clone();
    ctx.bind_to_thread().ok()?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }
        .result()
        .ok()?;
    Some(pool)
}

/// Release threshold set by [`retain_pool`] so far (bytes; process-wide,
/// only ever raised).
#[cfg(feature = "cuda")]
static RETAINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(RESERVED_MEM_CURRENT, USED_MEM_CURRENT)` of the default pool.
#[cfg(feature = "cuda")]
fn pool_usage(
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
) -> Option<(u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use sys::CUmemPool_attribute as A;
    let get = |attr| -> Option<u64> {
        let mut v: sys::cuuint64_t = 0;
        unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
            .result()
            .ok()?;
        Some(v)
    };
    Some((
        get(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        get(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
    ))
}

/// Let the default CUDA memory pool keep up to `extra` bytes of freed memory
/// on top of what is allocated right now, across stream synchronizations
/// (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`, default 0; the threshold counts ALL
/// reserved memory, used or not, so it is set to `used + extra`).
///
/// With threshold 0 every sync hands all freed pool memory back to the driver,
/// so the next VAE decode re-maps its whole multi-GB working set: 267
/// `cuMemAllocAsync` calls took 85 ms of a 281 ms 1024² decode and the GPU sat
/// idle waiting on them. The threshold is bounded (the caller passes its
/// working-set estimate) and only ever raised; `QIR_POOL_RETAIN=0` opts out.
/// No-op on CPU or if the driver call fails.
pub fn retain_pool(device: &Device, extra: usize) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use std::sync::atomic::Ordering;
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("QIR_POOL_RETAIN").map_or(true, |v| v != "0")) {
            return;
        }
        let Some(pool) = default_pool(d) else { return };
        let Some((_, used)) = pool_usage(pool) else {
            return;
        };
        let want = (used as usize).saturating_add(extra);
        if want <= RETAINED.load(Ordering::Relaxed) {
            return;
        }
        let mut v: sys::cuuint64_t = want as u64;
        let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
        let ok = unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
            .result()
            .is_ok();
        if ok {
            RETAINED.fetch_max(want, Ordering::Relaxed);
            tracing::debug!(
                threshold_mib = want >> 20,
                used_mib = used >> 20,
                "cuda mempool release threshold"
            );
        }
    }
    let _ = (device, extra);
}

/// Freed-but-retained bytes in the default pool (`RESERVED - USED`), which a
/// new allocation reuses but cuMemGetInfo does not count as free. 0 on CPU,
/// on a failed query, or while [`retain_pool`] has never run.
pub fn pool_slack(device: &Device) -> usize {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        if RETAINED.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return 0;
        }
        let Some(pool) = default_pool(d) else {
            return 0;
        };
        let Some((res, used)) = pool_usage(pool) else {
            return 0;
        };
        return res.saturating_sub(used) as usize;
    }
    let _ = device;
    0
}

/// Hand the default pool's freed-but-retained memory back to the driver
/// (`cuMemPoolTrimTo(pool, 0)` after a sync), so other processes on the card
/// see it free between requests. [`retain_pool`] keeps that memory for speed
/// within a run (the VAE's allocations); on a card shared with other processes
/// (ComfyUI, the desktop) 8–11 GB of slack held between requests pushed them,
/// and the worker's own next model load, into paging (2026-10-26). The next
/// run maps its working set again (on the order of 100 ms). No-op on CPU.
pub fn trim_pool(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = device {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let _ = device.synchronize();
        let Some(pool) = default_pool(d) else { return };
        let before = pool_usage(pool);
        let ok = unsafe { sys::cuMemPoolTrimTo(pool, 0) }.result().is_ok();
        if let (true, Some((res, used))) = (ok, before) {
            tracing::debug!(released_mib = res.saturating_sub(used) >> 20, "cuda mempool trimmed");
        }
    }
    let _ = device;
}

/// Free device memory as the worker can use it: the driver's free count plus
/// what the pool retains ([`pool_slack`]); `None` where [`free_vram`] is.
pub fn usable_vram(device: &Device) -> Option<(usize, usize)> {
    free_vram(device).map(|(free, total)| (free.saturating_add(pool_slack(device)), total))
}

/// Driver-free memory a working set must leave: under it Windows (WDDM) does not fail an
/// allocation but pages device memory to the host, silently 2–5x slower.
pub const PAGING_HEADROOM: usize = 256 << 20;

/// Log a warning when the driver's free memory and the pool's own slack are both under
/// [`PAGING_HEADROOM`] as `what` runs, so a run that Windows pages shows as paging in the log
/// rather than as a slow kernel. A full driver count beside a large slack is not paging: a whole
/// VAE decode grows the pool into the driver's last free memory and later work reuses it
/// (2026-10-29: 0 MiB driver-free with 6.4 GB slack, every step at its usual speed).
pub fn warn_if_paging(device: &Device, what: &str) {
    let slack = pool_slack(device);
    if let Some((free, _)) = free_vram(device).filter(|(free, _)| *free < PAGING_HEADROOM && slack < PAGING_HEADROOM) {
        tracing::warn!(what, driver_free_mib = free >> 20, slack_mib = slack >> 20, "device memory nearly full: Windows may page it");
    }
}

/// The compute capability the kernels were built for (`CUDA_COMPUTE_CAP` at build time, as
/// `major * 10 + minor`: 120 for the RTX 50 series, 89 for Ada).
pub fn built_compute_cap() -> u32 {
    env!("QIR_COMPUTE_CAP").trim().parse().unwrap_or(89)
}

/// The first CUDA card as the driver describes it, before any context exists.
#[derive(Clone, Debug, PartialEq)]
pub struct Gpu {
    pub name: String,
    /// Compute capability, `major * 10 + minor`.
    pub compute_cap: u32,
    pub total_bytes: usize,
}

/// Ask the driver about card 0 without creating a context (no device memory is taken). The
/// sentence says why there is none: a build without CUDA, no NVIDIA driver (checked before the
/// driver library is touched, since loading a missing one aborts in cudarc), no card.
pub fn probe() -> std::result::Result<Gpu, String> {
    #[cfg(feature = "cuda")]
    {
        if cfg!(windows) {
            let system = std::env::var_os("SystemRoot").map(std::path::PathBuf::from).unwrap_or_else(|| "C:\\Windows".into());
            if !system.join("System32").join("nvcuda.dll").is_file() {
                return Err("no NVIDIA driver is installed (nvcuda.dll not found)".into());
            }
        }
        // cudarc panics when the driver library cannot be loaded; elsewhere than Windows that
        // is the only check, so it is caught here.
        match std::panic::catch_unwind(probe_driver) {
            Ok(r) => r,
            Err(_) => Err("the NVIDIA driver library could not be loaded".into()),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("this build has no CUDA support".into())
    }
}

#[cfg(feature = "cuda")]
fn probe_driver() -> std::result::Result<Gpu, String> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};
    result::init().map_err(|e| format!("the NVIDIA driver did not start ({e})"))?;
    if result::device::get_count().map_err(|e| e.to_string())? < 1 {
        return Err("no NVIDIA card was found".into());
    }
    let dev = result::device::get(0).map_err(|e| e.to_string())?;
    let name = result::device::get_name(dev).map_err(|e| e.to_string())?;
    // SAFETY: `dev` is a device handle the driver just returned; the calls only read attributes.
    let (major, minor, total) = unsafe {
        (
            result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).map_err(|e| e.to_string())?,
            result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).map_err(|e| e.to_string())?,
            result::device::total_mem(dev).map_err(|e| e.to_string())?,
        )
    };
    Ok(Gpu { name: name.trim().to_string(), compute_cap: (major.max(0) * 10 + minor.max(0)) as u32, total_bytes: total })
}

/// Whether the kernels of this build run on `gpu`: its compute capability at least the built
/// one (the cubins carry PTX, which the driver compiles for newer cards). The sentence names
/// the card and what the build needs.
pub fn check(gpu: &Gpu, built: u32) -> std::result::Result<(), String> {
    if gpu.compute_cap >= built {
        return Ok(());
    }
    let cc = |c: u32| format!("{}.{}", c / 10, c % 10);
    Err(format!("{} (compute capability {}) is older than this build's kernels need ({} or newer)", gpu.name, cc(gpu.compute_cap), cc(built)))
}

/// Human-readable device label for logs.
pub fn label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_older_than_the_kernels_is_refused_with_its_name() {
        let gpu = |cc| Gpu { name: "Card".into(), compute_cap: cc, total_bytes: 1 };
        assert!(check(&gpu(120), 120).is_ok() && check(&gpu(121), 120).is_ok());
        let e = check(&gpu(89), 120).unwrap_err();
        assert!(e.contains("Card") && e.contains("8.9") && e.contains("12.0"), "{e}");
        assert!(built_compute_cap() >= 50);
        // The probe answers (a card or a sentence) and never aborts the process.
        match probe() {
            Ok(g) => assert!(!g.name.is_empty() && g.compute_cap >= 30, "{g:?}"),
            Err(e) => assert!(!e.is_empty()),
        }
    }
}
