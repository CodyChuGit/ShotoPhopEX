//! Memory profiles: the AI defaults matched to the graphics card's memory (or a Mac's unified
//! memory), picked once on the first launch (the owner's call, 2026-11-05: "a gpu check … so you
//! get matched settings/models for your vram/unified ram amount").
//!
//! `integrations.memoryProfile` is `auto` (use the profile detected at launch,
//! `integrations.detectedMemoryProfile`) or one of [`Profile::NAMES`]. A profile only changes
//! defaults: which image model Auto runs ([`image_template`]) and which edit model
//! ([`edit_template`]). A model chosen in the picker or in Preferences always wins, and no model
//! is hidden or locked on a smaller profile: the picker marks it `may not fit` ([`may_not_fit`]).
//!
//! The 32 GB and 24 GB profiles keep the defaults the app has always had (Krea 2, Qwen-Image-2.1).
//! The smaller ones run MOSS, the low-VRAM family (Kromoss 8-step turbo and MOSS Edit turbo,
//! measured at 11.4 / 12.0 GiB peak in the worker's 12 GB mode, 2026-11-05).

use crate::Session;

/// Memory a profile is matched to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Gb32,
    Gb24,
    Gb16,
    Gb12,
    Gb8,
    Mac16,
    Mac8,
}

/// `integrations.memoryProfile`'s values: `auto`, then every profile.
pub const PREF_NAMES: &[&str] = &["auto", "32gb", "24gb", "16gb", "12gb", "8gb", "mac16", "mac8"];

/// The image template of the big profiles (and of a machine nothing was detected on).
pub const BIG_IMAGE_TEMPLATE: &str = "krea2-turbo/image";
/// The image template of the small profiles: Kromoss with its 8-step DMD2 LoRA.
pub const SMALL_IMAGE_TEMPLATE: &str = "moss-image/turbo";
/// The edit template of the small profiles: MOSS Edit with the 8-step LoRA stacked.
pub const SMALL_EDIT_TEMPLATE: &str = "moss-image/edit-turbo";

impl Profile {
    pub const ALL: [Profile; 7] = [Profile::Gb32, Profile::Gb24, Profile::Gb16, Profile::Gb12, Profile::Gb8, Profile::Mac16, Profile::Mac8];

    pub fn name(self) -> &'static str {
        match self {
            Profile::Gb32 => "32gb",
            Profile::Gb24 => "24gb",
            Profile::Gb16 => "16gb",
            Profile::Gb12 => "12gb",
            Profile::Gb8 => "8gb",
            Profile::Mac16 => "mac16",
            Profile::Mac8 => "mac8",
        }
    }

    pub fn from_name(name: &str) -> Option<Profile> {
        Profile::ALL.into_iter().find(|p| p.name() == name.trim())
    }

    /// What people read: "32 GB", "Mac, 16 GB".
    pub fn label(self) -> &'static str {
        match self {
            Profile::Gb32 => "32 GB",
            Profile::Gb24 => "24 GB",
            Profile::Gb16 => "16 GB",
            Profile::Gb12 => "12 GB",
            Profile::Gb8 => "8 GB",
            Profile::Mac16 => "Mac, 16 GB",
            Profile::Mac8 => "Mac, 8 GB",
        }
    }

    /// Gigabytes a model may use at its peak on this profile: the card's memory less what the
    /// desktop and the canvas keep, or on a Mac what macOS lets the GPU have once the system and
    /// the app have theirs (`rust-rebuild/mac-memory-budget.md`).
    pub fn budget_gb(self) -> f32 {
        match self {
            Profile::Gb32 => 30.0,
            Profile::Gb24 => 22.0,
            Profile::Gb16 => 15.0,
            Profile::Gb12 => 11.5,
            Profile::Gb8 => 7.0,
            Profile::Mac16 => 9.0,
            Profile::Mac8 => 3.5,
        }
    }

    /// The low-VRAM profiles, which run MOSS by default.
    pub fn small(self) -> bool {
        !matches!(self, Profile::Gb32 | Profile::Gb24)
    }
}

/// The profile for a machine: `nvidia` = the largest NVIDIA card's memory, `unified` = an Apple
/// Silicon Mac's memory (bytes). `None` when neither is known (no NVIDIA card, an AMD or Intel
/// GPU): the app keeps its usual defaults.
pub fn detect(nvidia: Option<u64>, unified: Option<u64>) -> Option<Profile> {
    const GIB: f64 = (1u64 << 30) as f64;
    if let Some(b) = nvidia.filter(|b| *b > 0) {
        let gb = b as f64 / GIB;
        return Some(match gb {
            g if g >= 30.0 => Profile::Gb32,
            g if g >= 22.0 => Profile::Gb24,
            g if g >= 15.0 => Profile::Gb16,
            g if g >= 11.0 => Profile::Gb12,
            _ => Profile::Gb8,
        });
    }
    let gb = unified.filter(|b| *b > 0)? as f64 / GIB;
    Some(if gb >= 15.0 { Profile::Mac16 } else { Profile::Mac8 })
}

/// The profile in use: the chosen one, or under `auto` the one detected at launch.
pub fn active(s: &Session) -> Option<Profile> {
    let i = &s.prefs().integrations;
    match i.memory_profile.trim() {
        "" | "auto" => Profile::from_name(&i.detected_memory_profile),
        p => Profile::from_name(p),
    }
}

/// The image template Auto runs (Preferences › Default Image Template set to `auto`).
pub fn image_template(p: Option<Profile>) -> &'static str {
    if p.is_some_and(Profile::small) { SMALL_IMAGE_TEMPLATE } else { BIG_IMAGE_TEMPLATE }
}

/// The edit template Auto runs on this profile, when it differs from the usual choice.
pub fn edit_template(p: Option<Profile>) -> Option<&'static str> {
    p.is_some_and(Profile::small).then_some(SMALL_EDIT_TEMPLATE)
}

/// The fill template Auto runs on this profile, when it differs from the usual choice: a Mac
/// fills with FLUX 3 Klein on MLX (the Qwen fills are CUDA-only).
pub fn fill_template(p: Option<Profile>) -> Option<&'static str> {
    matches!(p, Some(Profile::Mac16 | Profile::Mac8)).then_some(crate::generate_cmds::KLEIN_FILL_TEMPLATE)
}

/// Gigabytes of card memory a template's family needs at its peak, in the formats it runs in
/// today (the worker drops the 2.1 text encoder to Q8_0 under 28 GB and runs MOSS in its 12 GB
/// mode on a small card). `None` for a family not listed.
pub fn need_gb(template: &str) -> Option<f32> {
    let family = template.split('/').next().unwrap_or("");
    Some(match family {
        // fp8 DiT 12.8 GB + Qwen3-VL 4B fp8 4.4 GB + VAE (18.2 GB resident on the 5090).
        f if f.starts_with("krea2") => 18.0,
        // int8 DiT 7.1 GB + Qwen3-VL 8B Q8_0 ~9 GB + VAE.
        "qwen-2.1" => 17.0,
        // fp8 DiT ~20 GB + Qwen2.5-VL 7B fp8 ~8 GB.
        f if f.starts_with("qwen-edit-2511") || f.starts_with("qwen-image-layered") => 28.0,
        // 12 GB mode: encoder then DiT, 11.4–12.0 GiB peak.
        f if f.starts_with("moss") => 11.0,
        f if f.starts_with("sam") => 2.0,
        _ => return None,
    })
}

/// The picker's `may not fit` badge: the template needs more than the profile's budget.
pub fn may_not_fit(p: Option<Profile>, template: &str) -> bool {
    // On a Mac MOSS runs on MLX: the 8-bit encoder, then the 8-bit DiT, 7.6 GB at the peak
    // (gn-mlx, measured on a 16 GB M1 Air, 2026-11-07).
    let need = if matches!(p, Some(Profile::Mac16 | Profile::Mac8)) && template.starts_with("moss") { Some(8.0) } else { need_gb(template) };
    match (p, need) {
        (Some(p), Some(need)) => need > p.budget_gb(),
        _ => false,
    }
}

/// Record the profile detected at this launch (`integrations.detectedMemoryProfile`). Returns
/// it when it is new (the first launch, or another graphics card since the last one), so the
/// app can say what it picked; `None` when it is the same as last time or nothing was detected.
pub fn record_detected(s: &mut Session, detected: Option<Profile>) -> Option<Profile> {
    let p = detected?;
    if s.prefs().integrations.detected_memory_profile == p.name() {
        return None;
    }
    s.edit_prefs(|prefs| prefs.integrations.detected_memory_profile = p.name().to_string());
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn detection_rounds_cards_to_their_profile() {
        // DXGI reports a 5090 as about 31.5 GiB, a 3090 as about 23.6.
        assert_eq!(detect(Some(32_607 << 20), None), Some(Profile::Gb32));
        assert_eq!(detect(Some(24_268 << 20), None), Some(Profile::Gb24));
        assert_eq!(detect(Some(16_303 << 20), None), Some(Profile::Gb16));
        assert_eq!(detect(Some(12_282 << 20), None), Some(Profile::Gb12));
        assert_eq!(detect(Some(8 * GIB), None), Some(Profile::Gb8));
        assert_eq!(detect(Some(6 * GIB), None), Some(Profile::Gb8));
        assert_eq!(detect(None, Some(16 * GIB)), Some(Profile::Mac16));
        assert_eq!(detect(None, Some(64 * GIB)), Some(Profile::Mac16));
        assert_eq!(detect(None, Some(8 * GIB)), Some(Profile::Mac8));
        assert_eq!(detect(None, None), None);
        assert_eq!(detect(Some(0), None), None, "zero reads as unknown");
    }

    #[test]
    fn big_profiles_keep_todays_models_and_small_ones_run_moss() {
        for p in [None, Some(Profile::Gb32), Some(Profile::Gb24)] {
            assert_eq!(image_template(p), BIG_IMAGE_TEMPLATE);
            assert_eq!(edit_template(p), None);
        }
        for p in [Profile::Gb16, Profile::Gb12, Profile::Gb8, Profile::Mac16, Profile::Mac8] {
            assert_eq!(image_template(Some(p)), SMALL_IMAGE_TEMPLATE);
            assert_eq!(edit_template(Some(p)), Some(SMALL_EDIT_TEMPLATE));
        }
    }

    #[test]
    fn badges_follow_the_budget() {
        let g = Some(Profile::Gb32);
        assert!(!may_not_fit(g, "qwen-edit-2511/fill"));
        let g = Some(Profile::Gb24);
        assert!(may_not_fit(g, "qwen-edit-2511/fill"));
        assert!(!may_not_fit(g, "krea2-turbo/image"));
        assert!(!may_not_fit(g, "qwen-2.1/fill"));
        let g = Some(Profile::Gb16);
        assert!(may_not_fit(g, "krea2-turbo/image"));
        assert!(may_not_fit(g, "qwen-2.1/fill"));
        assert!(!may_not_fit(g, "moss-image/turbo"));
        let g = Some(Profile::Gb12);
        assert!(!may_not_fit(g, "moss-image/edit-turbo"));
        assert!(may_not_fit(Some(Profile::Gb8), "moss-image/turbo"));
        assert!(!may_not_fit(None, "qwen-edit-2511/fill"), "nothing detected: no badges");
        assert!(!may_not_fit(Some(Profile::Gb8), "unknown/thing"));
    }

    #[test]
    fn auto_image_template_follows_the_profile_and_a_pick_wins() {
        use crate::generate_cmds::default_image_template;
        let mut s = Session::new();
        assert_eq!(default_image_template(&s), BIG_IMAGE_TEMPLATE, "nothing detected: today's default");
        record_detected(&mut s, Some(Profile::Gb12));
        assert_eq!(default_image_template(&s), SMALL_IMAGE_TEMPLATE);
        s.edit_prefs(|p| p.integrations.memory_profile = "24gb".into());
        assert_eq!(default_image_template(&s), BIG_IMAGE_TEMPLATE);
        s.edit_prefs(|p| p.integrations.default_image_template = "qwen-2.1/image".into());
        assert_eq!(default_image_template(&s), "qwen-2.1/image", "a chosen template stays");
    }

    #[test]
    fn names_round_trip_and_match_the_preference_choices() {
        for p in Profile::ALL {
            assert_eq!(Profile::from_name(p.name()), Some(p));
            assert!(PREF_NAMES.contains(&p.name()));
        }
        assert_eq!(PREF_NAMES.len(), Profile::ALL.len() + 1);
    }

    #[test]
    fn the_detected_profile_is_recorded_once_and_a_choice_wins() {
        let mut s = Session::new();
        assert_eq!(active(&s), None);
        assert_eq!(record_detected(&mut s, Some(Profile::Gb12)), Some(Profile::Gb12), "first launch");
        assert_eq!(record_detected(&mut s, Some(Profile::Gb12)), None, "same card: nothing new");
        assert_eq!(active(&s), Some(Profile::Gb12));
        assert_eq!(record_detected(&mut s, None), None, "a launch that sees nothing keeps the record");
        assert_eq!(active(&s), Some(Profile::Gb12));
        s.edit_prefs(|p| p.integrations.memory_profile = "32gb".into());
        assert_eq!(active(&s), Some(Profile::Gb32), "the chosen profile wins");
        assert_eq!(record_detected(&mut s, Some(Profile::Gb24)), Some(Profile::Gb24), "a new card");
        assert_eq!(active(&s), Some(Profile::Gb32));
    }
}
