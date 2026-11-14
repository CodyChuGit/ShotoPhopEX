//! Idol (TripoSplat): image to 3D gaussian splats, natively.

pub mod dino;
pub mod decoder;
pub mod dit;
#[cfg(feature = "flash")]
pub mod fused;
pub mod nn;
pub mod pipeline;
pub mod prof;
pub mod sobol;

#[cfg(test)]
mod parity;
