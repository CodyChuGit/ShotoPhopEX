//! MiniMax H3 image models (P-H3-Image by PrunaAI, MiniMax-H3-Image): the DiT, its packed
//! sequence, the text encoder's conditioning and the VAE, run from the ComfyUI repacks.
//! Port of ComfyUI 74eb0c4 `comfy/ldm/minimax`, `comfy/text_encoders/minimax.py`.

pub mod dit;
pub mod layout;

/// P-H3-Image's 8-step sigma grid (the Pruna workflows' `ManualSigmas`), at shift 1.
pub const PH3_SIGMAS: [f32; 9] = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0];

#[cfg(test)]
mod parity;
