//! MOSS-Image on Apple's MLX: qir's MOSS port (`qwen_image_rs::moss`) re-run on the Apple
//! Silicon GPU. The same math as qir's candle chains; the weights are read from the same
//! ComfyUI files through qir (ConvRot INT8 dequantized on the CPU, one tensor at a time) and
//! held as MLX f16 arrays, the large linears quantized to 4-bit (group 64). Activations run in
//! f16, the residual streams, norms and softmax in f32.
//!
//! macOS only: elsewhere the crate is empty.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(target_os = "macos")]
pub mod convert;
#[cfg(target_os = "macos")]
pub mod dit;
#[cfg(target_os = "macos")]
pub mod klein;
#[cfg(target_os = "macos")]
pub mod nn;
#[cfg(target_os = "macos")]
pub mod pipeline;
#[cfg(target_os = "macos")]
pub mod text;
#[cfg(target_os = "macos")]
pub mod vae;
