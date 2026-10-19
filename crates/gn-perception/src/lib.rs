//! Perception models of genai-native, run through ONNX Runtime.
//!
//! The first resident is SAM 3: a vision encoder that runs once per picture, a text encoder
//! and a geometry (box) encoder that run once per prompt, and a decoder that turns both into
//! masks, boxes and scores. ONNX Runtime itself is loaded at run time from a directory the
//! user points at, exactly like the weights: the application ships neither.
#![cfg_attr(test, allow(clippy::panic, clippy::unwrap_used, clippy::expect_used))]

pub mod resample;
pub mod runtime;
pub mod sam3;
pub mod tokenizer;
