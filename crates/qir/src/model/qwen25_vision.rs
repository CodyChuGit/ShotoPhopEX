//! Qwen2.5-VL image conditioning with ComfyUI patch and window ordering.

use crate::layered_files::{LayeredLinear, TensorFile};
use crate::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};
use std::path::Path;

const WIDTH: usize = 1280;
const HEADS: usize = 16;
const HEAD_DIM: usize = 80;
const PATCH: usize = 14;
const PATCH_WIDTH: usize = 3 * 2 * PATCH * PATCH;

fn rgb_tensor(rgb: &image::RgbImage) -> Result<Tensor> {
    anyhow::ensure!(rgb.width() > 0 && rgb.height() > 0, "empty image");
    let values: Vec<f32> = rgb.as_raw().iter().map(|v| *v as f32 / 255.0).collect();
    Ok(Tensor::from_vec(values, (rgb.height() as usize, rgb.width() as usize, 3), &Device::Cpu)?)
}

/// EditPlus's approximately 384^2 area resize, retaining fractional RGB values in HWC.
pub fn resize_editplus(rgb: &image::RgbImage) -> Result<Tensor> {
    let pixels = rgb_tensor(rgb)?;
    let (h, w, _) = pixels.dims3()?;
    let scale = (384.0 * 384.0 / (h * w) as f64).sqrt();
    let oh = (h as f64 * scale).round_ties_even() as usize;
    let ow = (w as f64 * scale).round_ties_even() as usize;
    anyhow::ensure!(oh > 0 && ow > 0, "invalid EditPlus aspect ratio");
    let source = pixels.to_vec3::<f32>()?;
    let mut out = vec![0f32; oh * ow * 3];
    // PyTorch area interpolation is adaptive average pooling, including overlapping bins.
    for y in 0..oh {
        let a = y * h / oh;
        let b = ((y + 1) * h).div_ceil(oh);
        for x in 0..ow {
            let c = x * w / ow;
            let d = ((x + 1) * w).div_ceil(ow);
            for ch in 0..3 {
                let mut sum = 0f32;
                for row in &source[a..b] {
                    for pixel in &row[c..d] {
                        sum += pixel[ch];
                    }
                }
                out[(y * ow + x) * 3 + ch] = sum / ((b - a) * (d - c)) as f32;
            }
        }
    }
    Ok(Tensor::from_vec(out, (oh, ow, 3), &Device::Cpu)?)
}

/// Normalize and pack two identical frames in the tower's 2x2 spatial order.
pub fn preprocess(pixels: &Tensor) -> Result<(Tensor, [usize; 3])> {
    let (h, w, c) = pixels.dims3()?;
    anyhow::ensure!(h > 0 && w > 0 && c == 3 && pixels.dtype() == DType::F32, "expected nonempty f32 HWC RGB");
    let factor = 28.0;
    let mut oh = ((h as f64 / factor).round_ties_even() * factor) as usize;
    let mut ow = ((w as f64 / factor).round_ties_even() * factor) as usize;
    if oh * ow > 12845056 {
        let beta = ((h * w) as f64 / 12845056.0).sqrt();
        oh = ((h as f64 / beta / factor).floor() as usize).max(1) * 28;
        ow = ((w as f64 / beta / factor).floor() as usize).max(1) * 28;
    } else if oh * ow < 3136 {
        let beta = (3136.0 / (h * w) as f64).sqrt();
        oh = (h as f64 * beta / factor).ceil() as usize * 28;
        ow = (w as f64 * beta / factor).ceil() as usize * 28;
    }
    let source = pixels.to_vec3::<f32>()?;
    let mut normalized = vec![0f32; oh * ow * 3];
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    // CLIP's published constants as ComfyUI writes them (f32 rounds the last digits away).
    #[allow(clippy::excessive_precision)]
    let std = [0.26862954f32, 0.26130258, 0.27577711];
    for y in 0..oh {
        let sy = ((y as f32 + 0.5) * (h as f64 / oh as f64) as f32 - 0.5).max(0.0);
        let ya = (sy as usize).min(h - 1);
        let yb = (ya + 1).min(h - 1);
        let dy = sy - ya as f32;
        for x in 0..ow {
            let sx = ((x as f32 + 0.5) * (w as f64 / ow as f64) as f32 - 0.5).max(0.0);
            let xa = (sx as usize).min(w - 1);
            let xb = (xa + 1).min(w - 1);
            let dx = sx - xa as f32;
            for ch in 0..3 {
                let top = (1.0 - dx) * source[ya][xa][ch] + dx * source[ya][xb][ch];
                let bottom = (1.0 - dx) * source[yb][xa][ch] + dx * source[yb][xb][ch];
                normalized[(y * ow + x) * 3 + ch] = ((1.0 - dy) * top + dy * bottom - mean[ch]) / std[ch];
            }
        }
    }
    let gh = oh / PATCH;
    let gw = ow / PATCH;
    let mut patches = Vec::with_capacity(gh * gw * PATCH_WIDTH);
    for by in 0..gh / 2 {
        for bx in 0..gw / 2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    for ch in 0..3 {
                        for _ in 0..2 {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    patches.push(normalized[(((by * 2 + iy) * PATCH + py) * ow + (bx * 2 + ix) * PATCH + px) * 3 + ch]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((Tensor::from_vec(patches, (gh * gw, PATCH_WIDTH), pixels.device())?, [1, gh, gw]))
}

/// Merged-token permutation and nonempty cumulative patch counts for 112-pixel windows.
pub fn window_index(grid: [usize; 3]) -> Result<(Vec<u32>, Vec<usize>)> {
    let [t, h, w] = grid;
    anyhow::ensure!(t > 0 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0, "invalid vision grid");
    let (h, w) = (h / 2, w / 2);
    anyhow::ensure!(t.checked_mul(h).and_then(|n| n.checked_mul(w)).is_some_and(|n| n <= u32::MAX as usize), "vision grid too large");
    let mut indices = Vec::new();
    let mut cu = vec![0];
    for frame in 0..t {
        for y in (0..h).step_by(4) {
            for x in (0..w).step_by(4) {
                for iy in y..(y + 4).min(h) {
                    for ix in x..(x + 4).min(w) {
                        indices.push((frame * h * w + iy * w + ix) as u32);
                    }
                }
                cu.push(indices.len() * 4);
            }
        }
    }
    Ok((indices, cu))
}

struct Block {
    norm1: RmsNorm,
    norm2: RmsNorm,
    qkv: LayeredLinear,
    proj: LayeredLinear,
    gate: LayeredLinear,
    up: LayeredLinear,
    down: LayeredLinear,
}
impl Block {
    fn load(vb: VarBuilder) -> Result<Self> {
        let linear = |i, o, name| -> Result<LayeredLinear> {
            let linear = LayeredLinear::load(i, o, true, true, None, vb.pp(name))?;
            Ok(if vb.device().is_cuda() { linear.precise_scale() } else { linear })
        };
        Ok(Self {
            norm1: candle_nn::rms_norm(WIDTH, 1e-6, vb.pp("norm1"))?,
            norm2: candle_nn::rms_norm(WIDTH, 1e-6, vb.pp("norm2"))?,
            qkv: linear(WIDTH, WIDTH * 3, "attn.qkv")?,
            proj: linear(WIDTH, WIDTH, "attn.proj")?,
            gate: linear(WIDTH, 3420, "mlp.gate_proj")?,
            up: linear(WIDTH, 3420, "mlp.up_proj")?,
            down: linear(3420, WIDTH, "mlp.down_proj")?,
        })
    }
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, cu: &[usize]) -> Result<Tensor> {
        let n = x.dim(0)?;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?.reshape((n, 3, HEADS, HEAD_DIM))?;
        let get = |i| -> Result<Tensor> { Ok(qkv.narrow(1, i, 1)?.squeeze(1)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?) };
        let q = candle_nn::rotary_emb::rope(&get(0)?.to_dtype(DType::F32)?, cos, sin)?.to_dtype(x.dtype())?;
        let k = candle_nn::rotary_emb::rope(&get(1)?.to_dtype(DType::F32)?, cos, sin)?.to_dtype(x.dtype())?;
        let v = get(2)?;
        let mut attended = Vec::new();
        for bounds in cu.windows(2) {
            let (a, len) = (bounds[0], bounds[1] - bounds[0]);
            let scores = (q.narrow(2, a, len)?.contiguous()?.matmul(&k.narrow(2, a, len)?.transpose(2, 3)?.contiguous()?)? / (HEAD_DIM as f64).sqrt())?;
            attended.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.narrow(2, a, len)?.contiguous()?)?);
        }
        let attention = Tensor::cat(&attended, 2)?.squeeze(0)?.transpose(0, 1)?.reshape((n, WIDTH))?;
        let x = (x + self.proj.forward(&attention)?)?;
        let norm = self.norm2.forward(&x)?;
        Ok((&x + self.down.forward(&(candle_nn::ops::silu(&self.gate.forward(&norm)?)? * self.up.forward(&norm)?)?)?)?)
    }
}

/// Streams vision blocks so the language decoder need not share their residency.
pub struct Qwen25Vision {
    file: TensorFile,
}
impl Qwen25Vision {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { file: TensorFile::open(path)? })
    }

    /// Return merged image tokens and the unmerged patch grid needed by text M-RoPE.
    pub fn encode(&self, rgb: &image::RgbImage) -> Result<(Tensor, [usize; 3])> {
        self.encode_pixels(&rgb_tensor(rgb)?)
    }

    /// Accept HWC float RGB, including the fractional output of `resize_editplus`.
    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<(Tensor, [usize; 3])> {
        self.run(pixels, &Device::Cpu, |_, _| Ok(()))
    }

    /// CUDA stages use bf16; the existing CPU parity route retains f32.
    pub fn encode_pixels_on_device(&self, pixels: &Tensor, device: &Device) -> Result<(Tensor, [usize; 3])> {
        self.run(pixels, device, |_, _| Ok(()))
    }

    fn run(&self, pixels: &Tensor, device: &Device, mut tap: impl FnMut(&str, &Tensor) -> Result<()>) -> Result<(Tensor, [usize; 3])> {
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let (patches, grid) = preprocess(pixels)?;
        let patches = patches.to_device(device)?.to_dtype(dtype)?;
        tap("patches", &patches)?;
        let n = patches.dim(0)?;
        let weight = self.file.tensor("visual.patch_embed.proj.weight", device)?.to_dtype(dtype)?.reshape((WIDTH, PATCH_WIDTH))?;
        let mut x = patches.matmul(&weight.t()?)?;
        tap("patch_embed", &x)?;
        let (indices, cu) = window_index(grid)?;
        let index = Tensor::from_vec(indices.clone(), indices.len(), device)?;
        tap("window_index", &index)?;
        x = x.reshape((n / 4, 4, WIDTH))?.index_select(&index, 0)?.reshape((n, WIDTH))?;
        let mut angles = Vec::with_capacity(n * HEAD_DIM / 2);
        for &idx in &indices {
            for iy in 0..2 {
                for ix in 0..2 {
                    let y = (idx as usize / (grid[2] / 2)) * 2 + iy;
                    let x = (idx as usize % (grid[2] / 2)) * 2 + ix;
                    for pos in [y, x] {
                        for j in 0..20 {
                            angles.push(pos as f32 * (1.0 / 10000f32.powf((2 * j) as f32 / 40.0)));
                        }
                    }
                }
            }
        }
        let angles = Tensor::from_vec(angles, (n, HEAD_DIM / 2), device)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        for i in 0..32 {
            let block = Block::load(self.file.builder(&format!("visual.blocks.{i}."), dtype, device)?)?;
            let full = [0, n];
            x = block.forward(&x, &cos, &sin, if [7, 15, 23, 31].contains(&i) { &full } else { &cu })?;
            tap(&format!("block{i}"), &x)?;
        }
        let vb = self.file.builder("visual.merger.", dtype, device)?;
        x = candle_nn::rms_norm(WIDTH, 1e-6, vb.pp("ln_q"))?.forward(&x)?.reshape((n / 4, WIDTH * 4))?;
        x = LayeredLinear::load(WIDTH * 4, WIDTH * 4, true, true, None, vb.pp("mlp.0"))?.precise_scale().forward(&x)?.gelu_erf()?;
        x = LayeredLinear::load(WIDTH * 4, 3584, true, true, None, vb.pp("mlp.2"))?.precise_scale().forward(&x)?;
        let mut inverse = vec![0u32; indices.len()];
        for (i, &j) in indices.iter().enumerate() {
            inverse[j as usize] = i as u32;
        }
        x = x.index_select(&Tensor::from_vec(inverse, indices.len(), device)?, 0)?;
        tap("merged", &x)?;
        Ok((x, grid))
    }

    /// Visit complete stage tensors without retaining every block's activation.
    pub fn encode_with_taps(&self, pixels: &Tensor, tap: impl FnMut(&str, &Tensor) -> Result<()>) -> Result<(Tensor, [usize; 3])> {
        self.run(pixels, &Device::Cpu, tap)
    }
}
