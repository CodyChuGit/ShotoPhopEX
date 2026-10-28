//! Small shared pieces for graphs run through ONNX Runtime: copying outputs out of a run, the
//! error type, and mask bookkeeping. The device, the node's knobs and the found objects are
//! shared with the candle backend, so they build without the `onnx` feature too.

use std::path::PathBuf;

#[cfg(feature = "onnx")]
use crate::runtime::RuntimeError;

/// The device a backend runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// The CUDA execution provider on GPU `ordinal`; falls back to the CPU provider for operators
    /// CUDA cannot run.
    Cuda { ordinal: i32 },
    /// The CPU provider only.
    Cpu,
}

/// The node's knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// `sigmoid(score)` above this keeps a detection (ComfyUI: 0.5).
    pub threshold: f32,
    /// Detections kept per text prompt, best first (ComfyUI: 1 unless the prompt says `:N`).
    pub max_detections: usize,
    /// Tracker refinement passes (ComfyUI: 2; 0 keeps the detector's masks).
    pub refine_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { threshold: 0.5, max_detections: 1, refine_iterations: 2 }
    }
}

/// A tensor copied out of a run: its shape and its data.
pub type Feat = (Vec<i64>, Vec<f32>);

/// A missing file, a runtime failure, or a graph that is not what a module expects.
#[derive(Debug)]
pub enum Error {
    Missing(PathBuf),
    #[cfg(feature = "onnx")]
    Runtime(RuntimeError),
    /// A graph output did not have the shape or type the module relies on.
    Shape(String),
    Tokenizer(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(p) => write!(f, "model file missing: {}", p.display()),
            #[cfg(feature = "onnx")]
            Error::Runtime(e) => write!(f, "{e}"),
            Error::Shape(s) => write!(f, "unexpected graph output: {s}"),
            Error::Tokenizer(s) => write!(f, "tokenizer: {s}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "onnx")]
impl From<RuntimeError> for Error {
    fn from(e: RuntimeError) -> Self {
        Error::Runtime(e)
    }
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Runtime(RuntimeError::Ort(e.to_string()))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Copy a named `f32` output out of a run.
#[cfg(feature = "onnx")]
pub fn take_f32(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Feat> {
    let v = outputs.get(name).ok_or_else(|| Error::Shape(format!("no output `{name}`")))?;
    let (shape, data) = v.try_extract_tensor::<f32>().map_err(|e| Error::Shape(format!("`{name}`: {e}")))?;
    Ok((shape.to_vec(), data.to_vec()))
}

#[cfg(feature = "onnx")]
pub fn take_bool(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<(Vec<i64>, Vec<bool>)> {
    let v = outputs.get(name).ok_or_else(|| Error::Shape(format!("no output `{name}`")))?;
    let (shape, data) = v.try_extract_tensor::<bool>().map_err(|e| Error::Shape(format!("`{name}`: {e}")))?;
    Ok((shape.to_vec(), data.to_vec()))
}

/// One dimension of a shape, or a readable error.
pub fn dim(shape: &[i64], i: usize, what: &str) -> Result<usize> {
    shape.get(i).and_then(|d| usize::try_from(*d).ok()).ok_or_else(|| Error::Shape(format!("{what}: shape {shape:?} has no dimension {i}")))
}

/// One found object, at picture resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub score: f32,
    /// x0, y0, x1, y1 in picture pixels (zeros when the prompt carried no box prediction).
    pub bbox: [f32; 4],
    /// Row-major `width × height` coverage, 0 or 1.
    pub mask: Vec<f32>,
    /// Pixel bounds of the mask as x0, y0, width, height, and the pixel count (both 0 when empty).
    pub bounds: [u32; 4],
    pub pixels: u64,
}

impl Instance {
    /// Wrap a coverage plane, measuring it.
    pub fn from_mask(mask: Vec<f32>, width: usize, height: usize, score: f32, bbox: [f32; 4]) -> Instance {
        let (bounds, pixels) = coverage_bounds(&mask, width, height);
        Instance { score, bbox, mask, bounds, pixels }
    }
}

/// Pixel bounds (x0, y0, w, h) and count of a coverage plane.
pub fn coverage_bounds(mask: &[f32], width: usize, height: usize) -> ([u32; 4], u64) {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut n = 0u64;
    for y in 0..height {
        for x in 0..width {
            if mask.get(y * width + x).is_some_and(|v| *v > 0.5) {
                n += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if n == 0 {
        return ([0; 4], 0);
    }
    ([x0 as u32, y0 as u32, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32], n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_of_a_block() {
        let mut m = vec![0.0; 16];
        m[5] = 1.0;
        m[6] = 1.0;
        m[10] = 1.0;
        assert_eq!(coverage_bounds(&m, 4, 4), ([1, 1, 2, 2], 3));
        assert_eq!(coverage_bounds(&[0.0; 4], 2, 2), ([0; 4], 0));
    }

    #[test]
    fn sigmoid_is_sane() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99 && sigmoid(-10.0) < 0.01);
    }
}
