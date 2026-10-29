//! Native worker protocol and model metadata for PhotoCraft.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(all(feature = "fake-server", not(target_arch = "wasm32")))]
pub mod fake;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
pub mod png;
pub mod template;

use serde::{Deserialize, Serialize};

/// Why a generation did not happen. Messages are written for the user and for agents.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// No server to talk to (unreachable, refused, or this build has no client).
    #[error("the generative backend is not available: {0}")]
    Unavailable(String),
    /// The server answered, but not in the shape the protocol promises.
    #[error("the generative server answered unexpectedly: {0}")]
    Protocol(String),
    /// The server ran the request and reported a failure (a missing model file or runtime failure).
    #[error("the generative server reported an error: {0}")]
    Server(String),
    #[error("workflow template: {0}")]
    Template(String),
    #[error("invalid generative request: {0}")]
    Request(String),
    #[error("image data: {0}")]
    Image(String),
    #[error("cancelled")]
    Cancelled,
    #[error("the generation did not finish within {0} s")]
    Timeout(u64),
    /// The workflow ran but produced no image (a detector found nothing, for instance).
    #[error("the workflow produced no output ({0})")]
    NoOutput(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Straight-alpha RGBA, 8 bits per channel, row-major, top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgba8 {
    /// `data` must hold exactly `width × height × 4` bytes.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let need = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(4)).ok_or_else(|| Error::Image("image size overflows".into()))?;
        if width == 0 || height == 0 {
            return Err(Error::Image("an image needs a non-zero size".into()));
        }
        if data.len() != need {
            return Err(Error::Image(format!("{}×{} RGBA needs {need} bytes, got {}", width, height, data.len())));
        }
        Ok(Self { width, height, data })
    }

    /// A solid-colour image (tests and placeholders).
    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Result<Self> {
        let n = (width as usize).checked_mul(height as usize).ok_or_else(|| Error::Image("image size overflows".into()))?;
        let mut data = Vec::with_capacity(n * 4);
        for _ in 0..n {
            data.extend_from_slice(&rgba);
        }
        Self::new(width, height, data)
    }

    /// The pixel at `(x, y)`, or `None` outside the image.
    pub fn get(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.data.get(i..i + 4).map(|p| [p[0], p[1], p[2], p[3]])
    }
}

/// An 8-bit coverage mask: 255 inside, 0 outside, same geometry as the image it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Gray8 {
    /// `data` must hold exactly `width × height` bytes.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let need = (width as usize).checked_mul(height as usize).ok_or_else(|| Error::Image("mask size overflows".into()))?;
        if width == 0 || height == 0 {
            return Err(Error::Image("a mask needs a non-zero size".into()));
        }
        if data.len() != need {
            return Err(Error::Image(format!("{}×{} mask needs {need} bytes, got {}", width, height, data.len())));
        }
        Ok(Self { width, height, data })
    }
}

/// What a request asks a model to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// Regenerate the masked part of an image from a prompt (Generative Fill).
    Fill,
    /// Extend an image beyond its edges (Generative Expand).
    Expand,
    /// Text to image.
    Image,
    /// Change an image by instruction (no mask).
    Edit,
    /// Foreground matte / background removal.
    Matte,
    Upscale,
    /// Instance masks for a text (or point) prompt; the result images are coverage, not pictures.
    Segment,
    /// Decompose a picture into RGBA layers; one result image per layer.
    Layers,
    /// Text in, text out (a prompt rewritten by a language model); no image results.
    Text,
}

/// One generation, backend-agnostic. The engine builds it; a backend runs it.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// A [`template::Template`] id such as `qwen-edit-2511/fill`.
    pub template: String,
    pub prompt: String,
    pub negative: String,
    /// Kept below 2^53 so every JSON reader keeps it exact.
    pub seed: u64,
    pub steps: u32,
    pub guidance: f32,
    /// The source pixels (the composite around the selection, or the layer being edited).
    pub image: Option<Rgba8>,
    /// Coverage of the area to regenerate, same size as `image`.
    pub mask: Option<Gray8>,
    /// Model-file overrides by placeholder name (`unet`, `clip`, `vae`, …); the template's
    /// defaults apply otherwise.
    pub models: Vec<(String, String)>,
    /// Output size for text-to-image templates (`width`/`height` placeholders).
    pub size: Option<(u32, u32)>,
    /// Template-specific placeholder values (`threshold`, …). The standard bindings win on a
    /// name clash.
    pub params: std::collections::BTreeMap<String, serde_json::Value>,
    /// Reference images in native request order (at most `references.max`, at least
    /// `references.min`).
    pub references: Vec<Rgba8>,
}

impl Request {
    /// A request with the template's defaults and no pixels.
    pub fn new(template: &str, prompt: &str) -> Self {
        Self {
            template: template.to_string(),
            prompt: prompt.to_string(),
            negative: String::new(),
            seed: random_seed(),
            steps: 0,
            guidance: 0.0,
            image: None,
            mask: None,
            models: Vec::new(),
            size: None,
            params: std::collections::BTreeMap::new(),
            references: Vec::new(),
        }
    }
}

/// What came back.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    /// The result images, in the order the server produced them (empty for a text template).
    pub images: Vec<Rgba8>,
    /// Chat texts returned by the worker; empty for image templates.
    pub texts: Vec<String>,
    pub seed: u64,
    /// The worker's id for the run, for logs and reproduction.
    pub run_id: String,
    pub elapsed_ms: u64,
    /// Where the time went (benchmarks, `generate.*` results).
    pub timings: Timings,
}

/// Milliseconds spent in each stage of a run. `queue` is the wait between queueing the prompt
/// and the server starting on it (other jobs, model loading counts as `run`); `run` is the
/// server's execution; the rest is the client's own work and the transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Timings {
    pub encode_ms: u64,
    pub upload_ms: u64,
    pub queue_ms: u64,
    pub run_ms: u64,
    pub download_ms: u64,
}

/// Progress and cancellation, implemented by the engine's job context.
pub trait Progress {
    /// `fraction` in `0.0..=1.0` with a short status (empty keeps the previous one).
    fn report(&self, fraction: f32, message: &str);
    /// Checked between steps; a backend stops and returns [`Error::Cancelled`] when true.
    fn cancelled(&self) -> bool;
}

/// A [`Progress`] that ignores everything (tests, the CLI's inline runs).
pub struct NoProgress;

impl Progress for NoProgress {
    fn report(&self, _fraction: f32, _message: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

/// The server's state, for `generate.health`, the picker and the preferences page.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Health {
    pub ok: bool,
    pub backend: String,
    pub url: String,
    pub version: String,
    /// The server's first GPU as it names it ("NVIDIA GeForce RTX 5090 : cudaMallocAsync").
    #[serde(default)]
    pub device: String,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    pub queue_remaining: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A generative engine. Implementations are shared between the UI thread and job workers.
pub trait GenerativeBackend: Send + Sync {
    /// Reachability and resources; never an `Err` (the answer carries the error).
    fn health(&self) -> Health;
    /// Run one request to completion, reporting progress and honouring cancellation.
    fn run(&self, req: &Request, progress: &dyn Progress) -> Result<Response>;
    /// Model files the server has in `folder` (`diffusion_models`, `text_encoders`, `vae`, …).
    fn model_files(&self, folder: &str) -> Result<Vec<String>>;
    /// Ask the server to unload its models and free their memory (the next run reloads them).
    /// Backends without resident models do nothing.
    fn free(&self) -> Result<()> {
        Ok(())
    }
}

/// Random bytes: the OS source on native targets; on wasm (no client, so seeds never reach a
/// server) a counter run through a mixer, which is enough to keep ids distinct.
fn random_bytes(out: &mut [u8]) {
    #[cfg(not(target_arch = "wasm32"))]
    if getrandom::fill(out).is_ok() {
        return;
    }
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    for chunk in out.chunks_mut(8) {
        // splitmix64
        let mut z = COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        for (dst, src) in chunk.iter_mut().zip(z.to_le_bytes()) {
            *dst = src;
        }
    }
}

/// A seed below 2^53 (exact in every JSON implementation).
pub fn random_seed() -> u64 {
    let mut b = [0u8; 8];
    random_bytes(&mut b);
    u64::from_le_bytes(b) & ((1u64 << 53) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_and_mask_sizes_are_checked() {
        assert!(Rgba8::new(2, 2, vec![0; 16]).is_ok());
        assert!(Rgba8::new(2, 2, vec![0; 15]).is_err());
        assert!(Rgba8::new(0, 2, vec![]).is_err());
        assert!(Gray8::new(3, 1, vec![0; 3]).is_ok());
        assert!(Gray8::new(3, 1, vec![0; 2]).is_err());
        let s = Rgba8::solid(2, 1, [1, 2, 3, 4]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.get(1, 0), Some([1, 2, 3, 4]));
        assert_eq!(s.get(2, 0), None);
    }

    #[test]
    fn seeds_stay_exact_in_json() {
        for _ in 0..100 {
            assert!(random_seed() < (1u64 << 53));
        }
    }
}
