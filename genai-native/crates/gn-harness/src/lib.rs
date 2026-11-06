//! The parity and performance harness of `genai-native` (plan-native-engine §12.1).
//!
//! Every engine the plan names (the ComfyUI route through PhotoshopEX's CLI today; sd.cpp's
//! `sd-cli` and the native worker later) runs the same **fixtures** ([`fixtures`]) and writes a
//! **run folder** (`<case>.png` + `<case>.json` per case, `summary.md`); [`report::compare`]
//! diffs two run folders with the **metrics** ([`metrics`]: PSNR, SSIM, mask IoU, boundary F)
//! and prints the Markdown table the dev log and `benchmarks.md` carry.
//!
//! Nothing here panics: a case that fails is a row that says so.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

pub mod engines;
pub mod fixtures;
pub mod image;
pub mod metrics;
pub mod report;

pub use fixtures::{Case, Kind, Prompt, Suite};
pub use image::Image;

/// A harness error: a sentence for the report.
pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;
