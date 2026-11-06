//! The BYOM weight loader (plan-native-engine §3.2, layer 4). The app ships no weights; the user
//! arrives with official safetensors, ComfyUI's repacks (fp8-scaled, NVFP4, INT8 convrot) or
//! GGUF, and this crate reads whatever that is: the file formats ([`safetensors`], [`gguf`]),
//! what each tensor is ([`dtype`]), which known model a file holds ([`index`]), and a folder
//! index in ComfyUI's layout for "what is installed".
//!
//! Model files are untrusted input: every length comes from the file and is checked against the
//! file before it is used; nothing here allocates from a header's number, panics, or evaluates
//! anything. No pickle, ever (`.pt`/`.ckpt` are refused with a sentence).

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

pub mod dtype;
pub mod gguf;
pub mod index;
pub mod io;
pub mod safetensors;

pub use dtype::DType;
pub use index::{Family, Fingerprint, Installed, ModelIndex, Precision};

/// A loader error: a sentence naming the file and what was wrong with it.
pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

/// Caps on what a header may claim before any of it is believed.
pub mod limits {
    /// A safetensors JSON header larger than this is not a model file (the largest real ones
    /// are a few megabytes).
    pub const SAFETENSORS_HEADER: u64 = 256 * 1024 * 1024;
    /// Tensors per file (Qwen-Image's transformer has ~1 800; a merged checkpoint a few thousand).
    pub const TENSORS: u64 = 1 << 20;
    /// GGUF metadata entries per file.
    pub const GGUF_KV: u64 = 1 << 20;
    /// Elements in one GGUF array value (tokenizer vocabularies run to ~150 k).
    pub const GGUF_ARRAY: u64 = 1 << 24;
    /// Bytes in one GGUF string.
    pub const GGUF_STRING: u64 = 1 << 20;
    /// Dimensions of one tensor.
    pub const DIMS: usize = 8;
}
