//! SAM's spatial positions: axial interleaved rotary and fixed DETR sine features.

use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};

/// Interleaved-pair cosine and sine tables in x-then-y axis order.
pub struct Rope2d {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope2d {
    /// Build `[H*W,D/2]` tables. Global SAM uses scale 24/72, windows use one.
    /// Each axis owns half the head channels, so its exponent denominator is D/2.
    pub fn new(w: usize, h: usize, dim: usize, scale: f32, device: &Device) -> Result<Self> {
        ensure!(w > 0 && h > 0 && dim > 0 && dim.is_multiple_of(4) && scale.is_finite(), "invalid RoPE dimensions or scale");
        let mut cos = Vec::with_capacity(w * h * dim / 2);
        let mut sin = Vec::with_capacity(w * h * dim / 2);
        for t in 0..w * h {
            for axis in [t % w, t / w] {
                for pair in 0..dim / 4 {
                    let pos = axis as f32 * scale;
                    let angle = f64::from(pos) / 10000_f64.powf((2 * pair) as f64 / (dim / 2) as f64);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        Ok(Self { cos: Tensor::from_vec(cos, (w * h, dim / 2), device)?, sin: Tensor::from_vec(sin, (w * h, dim / 2), device)? })
    }

    /// Rotate adjacent channel pairs, preserving `[B,H,N,D]` layout.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &self.cos, &self.sin)?)
    }
}

/// Encode normalized image coordinates as `[1,256,H,W]`, y channels before x.
/// The size-minus-one denominator and epsilon reproduce ComfyUI's pixel-grid endpoints.
pub fn image_position(h: usize, w: usize, device: &Device) -> Result<Tensor> {
    ensure!(h > 0 && w > 0, "image position dimensions must be positive");
    let mut values = Vec::with_capacity(256 * h * w);
    for channel in 0..256 {
        let i = channel % 128;
        let freq = 10000_f32.powf((2 * (i / 2)) as f32 / 128.0);
        for y in 0..h {
            for x in 0..w {
                let (index, size) = if channel < 128 { (y, h) } else { (x, w) };
                let angle = (index as f32 / ((size - 1) as f32 + 1e-6)) * std::f32::consts::TAU / freq;
                values.push(if i % 2 == 0 { angle.sin() } else { angle.cos() });
            }
        }
    }
    Ok(Tensor::from_vec(values, (1, 256, h, w), device)?)
}
