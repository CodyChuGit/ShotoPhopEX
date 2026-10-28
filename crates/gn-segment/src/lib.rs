//! Checkpoint validation, parity primitives, CLIP, vision, geometry, the detector and the interactive heads for the
//! SAM 3.1 port.

pub mod attention;
pub mod detector;
mod detector_heads;
pub mod geometry;
pub mod mask_decoder;
pub mod ops;
pub mod position;
pub mod prompt;
pub mod text;
pub mod vision;
pub mod weights;

mod resample;
pub mod sam;

pub mod profile;

mod linear;
