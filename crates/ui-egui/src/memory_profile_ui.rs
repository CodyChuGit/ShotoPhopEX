//! The first-launch memory check: which memory profile this machine gets
//! ([`photocraft_engine::memory_profile`]), and a notice saying what it picked the first time
//! (and again on another graphics card).

use photocraft_engine::memory_profile::{self, Profile};

use crate::PhotocraftApp;

/// The memory the AI models can use here: the largest NVIDIA card's, or on an Apple Silicon Mac
/// its unified memory. `(nvidia, unified)`.
pub fn machine_memory() -> (Option<u64>, Option<u64>) {
    let nvidia = crate::gpu_canvas::nvidia_card_memory();
    let unified = if cfg!(all(target_os = "macos", target_arch = "aarch64")) { crate::gpu_canvas::physical_memory() } else { None };
    (nvidia, unified)
}

/// Detect the profile, record it, and post a notice when it is new. Returns the new profile.
pub fn check_at_launch(app: &mut PhotocraftApp) -> Option<Profile> {
    let (nvidia, unified) = machine_memory();
    let detected = memory_profile::detect(nvidia, unified);
    let p = memory_profile::record_detected(&mut app.session, detected)?;
    if let Err(e) = crate::prefs_ui::save_preferences(app) {
        log::warn!("couldn't save the detected memory profile: {e}");
    }
    let gb = nvidia.or(unified).map_or_else(String::new, |b| format!("{}", (b as f64 / (1u64 << 30) as f64).round()));
    log::info!("memory profile: {} ({gb} GB detected)", p.name());
    let models = if p.small() { tl!("MOSS Turbo for images and MOSS Edit Turbo for edits") } else { tl!("Krea 2 and Qwen-Image-2.1") };
    let chosen = app.session.prefs().integrations.memory_profile.trim() != "auto";
    let mut lines = vec![crate::i18n::fmt(tl!("{gb} GB of graphics memory: the AI models default to {models}."), &[("gb", &gb), ("models", models)])];
    if chosen {
        lines.push(tl!("Your own choice in Preferences › AI Integrations › Memory profile still applies.").to_string());
    } else {
        lines.push(tl!("Models that may need more memory are marked “may not fit”. Change it in Preferences › AI Integrations › Memory profile.").to_string());
    }
    lines.push(tl!("Help › Get AI Models… downloads the models for this profile.").to_string());
    if matches!(p, Profile::Gb8 | Profile::Mac8) {
        lines.push(tl!("Image generation needs a smaller model that isn't ready yet; selection tools work.").to_string());
    }
    crate::notices::post(app, crate::i18n::fmt(tl!("Memory profile: {profile}"), &[("profile", p.label())]), lines, false, None);
    Some(p)
}
