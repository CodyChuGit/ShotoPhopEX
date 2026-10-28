//! Perception models of genai-native: SAM 3.1 on candle from ComfyUI's checkpoint (`candle`),
//! or the exported graphs through ONNX Runtime (`onnx`, on by default).
//!
//! The first resident is SAM 3: a vision encoder that runs once per picture, a text encoder
//! and a geometry (box) encoder that run once per prompt, and a decoder that turns both into
//! masks, boxes and scores. ONNX Runtime itself is loaded at run time from a directory the
//! user points at, exactly like the weights: the application ships neither. Without the
//! `onnx` feature the crate has no `ort` at all (Rust rebuild R1).
#![cfg_attr(test, allow(clippy::panic, clippy::unwrap_used, clippy::expect_used))]

#[cfg(all(feature = "engine", not(any(feature = "onnx", feature = "candle"))))]
compile_error!("the perception engine needs a backend: the `onnx` feature, `candle`, or both");

#[cfg(feature = "engine")]
pub mod engine;
pub mod graph;
pub mod resample;
#[cfg(feature = "onnx")]
pub mod runtime;
#[cfg(feature = "onnx")]
pub mod sam3;
#[cfg(feature = "onnx")]
pub mod sam31;
pub mod tokenizer;
