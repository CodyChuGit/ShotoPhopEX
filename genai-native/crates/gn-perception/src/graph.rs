//! Small shared pieces of the perception engine: the device, the node's knobs, and the found
//! objects with their mask bookkeeping.

/// The device a backend runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// CUDA GPU `ordinal`.
    Cuda { ordinal: i32 },
    /// The CPU only.
    Cpu,
    /// The Apple GPU through candle's Metal backend (macOS).
    Metal,
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
    /// Soft text-mask edges: `sigmoid(logit / soft)`; 0 cuts at logit 0 (ComfyUI).
    pub soft: f32,
    /// A single click's other meanings (`gn_segment::sam::Options::candidate`).
    pub candidate: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { threshold: 0.5, max_detections: 1, refine_iterations: 2, soft: 0.0, candidate: None }
    }
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
        let inst = Instance::from_mask(m, 4, 4, 0.9, [0.0; 4]);
        assert_eq!((inst.bounds, inst.pixels), ([1, 1, 2, 2], 3));
    }
}
