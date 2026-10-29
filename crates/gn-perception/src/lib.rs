//! Perception models of genai-native: SAM 3.1 on candle from ComfyUI's checkpoint (`candle`,
//! through `gn-segment`), served over `gn-engine`'s worker protocol (`engine`).
//!
//! SAM 3.1 is a vision encoder that runs once per picture, a text encoder and a geometry (box)
//! encoder that run once per prompt, and a decoder that turns both into masks, boxes and
//! scores. The weights come from a directory the user points at; the application ships none.
#![cfg_attr(test, allow(clippy::panic, clippy::unwrap_used, clippy::expect_used))]

#[cfg(feature = "engine")]
pub mod engine;
pub mod graph;
