//! Idol (TripoSplat): image to 3D gaussian splats, natively.

pub mod dino;
pub mod decoder;
pub mod dit;
pub mod nn;
pub mod pipeline;
pub mod prof;
pub mod sobol;

#[cfg(test)]
mod parity;
