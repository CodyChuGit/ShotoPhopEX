//! MOSS-Image (OpenMOSS, Apache-2.0): text to image with a 5B single-stream DiT, Qwen3.5-9B's
//! language model as the text encoder and the FLUX.2 VAE, run from Comfy-Org's repack of the
//! weights as it is (ConvRot INT8 linears on CUDA). See [`pipeline`].

pub mod dit;
#[cfg(feature = "fusednorm")]
pub mod fused;
pub mod pipeline;
pub mod text;
pub mod vae;
pub mod weights;
