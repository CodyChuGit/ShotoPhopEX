//! Pictures of a generation while it samples. After a step the pipeline hands its denoised
//! estimate x0 (flow matching: `x0 = x − σ·v`, in the sampler's normalised latent space) to a
//! [`StepPreview`]; [`Previewer`] turns it into RGBA:
//!
//! - **Fast**: a linear map from the latent's channels to RGB plus a bias, at the latent's grid
//!   (1/8 of the picture for the Wan 2.1 family, 1/16 for Qwen-Image-2.1's 64-channel VAE).
//!   The factors are ComfyUI's (`comfy/latent_formats.py`, `Wan21` and `QwenImage21`), which
//!   apply to the same normalised latents.
//! - **Sharp**: a tiny decoder at the picture's size, every [`SHARP_EVERY`] steps: madebyollin's
//!   TAEHV `taew2_1` for the Wan 2.1 family (Krea 2, Qwen-Image-Edit-2511, Layered) and TAESD's
//!   16× variant `taeqi2_1` for Qwen-Image-2.1 (both MIT, files under the model root's
//!   `vae_approx/`). Without the file, or without room on the card, Sharp shows Fast.
//!
//! A preview reads the latents and never writes them: the sampled picture is the same with
//! previews off, fast or sharp. A preview that fails is skipped (and Sharp falls back to Fast
//! for the rest of the run); it never fails the generation. Each preview copies its pixels to
//! the host, which waits for the queued step; previews are throttled to [`MIN_GAP`] apart.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::Result;

/// What a run shows while it samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PreviewMode {
    #[default]
    Off,
    Fast,
    Sharp,
}

/// Sharp decodes every this many steps (Fast shows every step).
pub const SHARP_EVERY: usize = 3;
/// At most ten previews a second.
pub const MIN_GAP: Duration = Duration::from_millis(100);
/// The model root's folder for tiny decoders (ComfyUI's name for it).
pub const TINY_DIR: &str = "vae_approx";

/// The latent space a pipeline samples in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatentSpace {
    /// 16 channels, 8× (the Wan 2.1 VAE and its relatives: Krea 2, Qwen-Image-Edit-2511),
    /// normalised by Wan 2.1's per-channel statistics.
    Wan21,
    /// 64 channels, 16× (Qwen-Image-2.1's VAE), normalised by its own statistics.
    QwenImage21,
    /// Qwen-Image-Layered's RGBA VAE: 16 channels, 8×, but its own latent scale (Wan 2.1's
    /// factors and taew2_1 give grey). Fast only: no tiny decoder matches it.
    Layered,
    /// The FLUX.2 VAE's (MOSS-Image): 128 channels at 16×, its 32 channels 2×2-packed and
    /// batch-normalised; unpacked to 32 channels at 8× for the factors and the tiny decoder.
    Flux2,
}

impl LatentSpace {
    /// The tiny decoder's file name under [`TINY_DIR`], when one decodes this space.
    pub fn tiny_file(self) -> Option<&'static str> {
        match self {
            Self::Wan21 => Some("taew2_1.safetensors"),
            Self::QwenImage21 => Some("taeqi2_1.safetensors"),
            Self::Layered => None,
            Self::Flux2 => Some("taef2.safetensors"),
        }
    }

    pub fn channels(self) -> usize {
        match self {
            Self::Wan21 | Self::Layered => 16,
            Self::QwenImage21 => 64,
            Self::Flux2 => 128,
        }
    }

    /// Pixels per latent cell along a side.
    pub fn scale(self) -> usize {
        match self {
            Self::Wan21 | Self::Layered => 8,
            Self::QwenImage21 | Self::Flux2 => 16,
        }
    }

    /// The latent-to-RGB factors (one row per channel, after [`unpack`] for FLUX.2) and the
    /// bias; RGB in `[-1, 1]`.
    fn factors(self) -> (&'static [[f32; 3]], [f32; 3]) {
        match self {
            Self::Wan21 => (&WAN21_RGB, WAN21_RGB_BIAS),
            Self::QwenImage21 => (&QWEN21_RGB, QWEN21_RGB_BIAS),
            Self::Layered => (&LAYERED_RGB, LAYERED_RGB_BIAS),
            Self::Flux2 => (&FLUX2_RGB, FLUX2_RGB_BIAS),
        }
    }
}

/// A FLUX.2 latent `(…, 128, h, w)` as its 32 channels at `(2h, 2w)`: channel `4c + 2i + j` of a
/// cell goes to `(2y + i, 2x + j)` of channel `c` (the packing MOSS-Image samples in).
fn unpack(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let lead: Vec<usize> = dims[..dims.len() - 3].to_vec();
    let (c, h, w) = (dims[dims.len() - 3], dims[dims.len() - 2], dims[dims.len() - 1]);
    let n: usize = lead.iter().product();
    let y = x.reshape((n, c / 4, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.contiguous()?;
    let mut out = lead;
    out.extend([c / 4, 2 * h, 2 * w]);
    Ok(y.reshape(out)?)
}

/// ComfyUI `latent_formats.Flux2.latent_rgb_factors` / `_bias` (read 2026-10-30, ComfyUI 0.43),
/// for the unpacked 32 channels.
const FLUX2_RGB: [[f32; 3]; 32] = [
    [0.0058, 0.0113, 0.0073],
    [0.0495, 0.0443, 0.0836],
    [-0.0099, 0.0096, 0.0644],
    [0.2144, 0.3009, 0.3652],
    [0.0166, -0.0039, -0.0054],
    [0.0157, 0.0103, -0.0160],
    [-0.0398, 0.0902, -0.0235],
    [-0.0052, 0.0095, 0.0109],
    [-0.3527, -0.2712, -0.1666],
    [-0.0301, -0.0356, -0.0180],
    [-0.0107, 0.0078, 0.0013],
    [0.0746, 0.0090, -0.0941],
    [0.0156, 0.0169, 0.0070],
    [-0.0034, -0.0040, -0.0114],
    [0.0032, 0.0181, 0.0080],
    [-0.0939, -0.0008, 0.0186],
    [0.0018, 0.0043, 0.0104],
    [0.0284, 0.0056, -0.0127],
    [-0.0024, -0.0022, -0.0030],
    [0.1207, -0.0026, 0.0065],
    [0.0128, 0.0101, 0.0142],
    [0.0137, -0.0072, -0.0007],
    [0.0095, 0.0092, -0.0059],
    [0.0000, -0.0077, -0.0049],
    [-0.0465, -0.0204, -0.0312],
    [0.0095, 0.0012, -0.0066],
    [0.0290, -0.0034, 0.0025],
    [0.0220, 0.0169, -0.0048],
    [-0.0332, -0.0457, -0.0468],
    [-0.0085, 0.0389, 0.0609],
    [-0.0076, 0.0003, -0.0043],
    [-0.0111, -0.0460, -0.0614],
];
const FLUX2_RGB_BIAS: [f32; 3] = [-0.0329, -0.0718, -0.0851];

/// Fitted here by least squares (`tests/fit_layered_rgb.rs`, 2026-10-29): two 864×1152 pictures
/// split into three layers, each decoded layer averaged over the latent's 8×8 cells, the 63,966
/// opaque cells against the normalised latent.
const LAYERED_RGB: [[f32; 3]; 16] = [
    [1.6850, 2.9788, 4.7393],
    [-0.2873, -0.2436, -0.1137],
    [1.1271, 0.5936, -0.0878],
    [2.1219, 1.9036, 0.9615],
    [-0.0528, -0.2833, -0.4493],
    [1.2324, 0.3905, 0.0373],
    [-1.9189, -1.7780, -1.1246],
    [-0.6648, 0.0147, 0.3365],
    [-3.3269, -3.6531, -4.2630],
    [-1.1779, -0.3111, 0.1830],
    [-0.7592, -0.1260, -0.4620],
    [0.6080, 1.0463, 1.4395],
    [-1.7267, -0.6338, -0.0953],
    [0.0456, -0.1541, -0.0593],
    [2.6103, 2.1575, 2.6603],
    [1.7832, 1.5612, 1.4954],
];
const LAYERED_RGB_BIAS: [f32; 3] = [-1.1025, -1.2532, -1.8779];

/// ComfyUI `latent_formats.Wan21.latent_rgb_factors` / `_bias` (read 2026-10-29, ComfyUI 0.43).
const WAN21_RGB: [[f32; 3]; 16] = [
    [-0.1299, -0.1692, 0.2932],
    [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747],
    [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328],
    [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364],
    [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128],
    [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699],
    [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950],
    [0.1984, 0.0913, 0.1861],
];
const WAN21_RGB_BIAS: [f32; 3] = [-0.1835, -0.0868, -0.3360];

/// ComfyUI `latent_formats.QwenImage21.latent_rgb_factors` / `_bias` (read 2026-10-29, 0.43).
const QWEN21_RGB: [[f32; 3]; 64] = [
    [-0.0158, -0.0115, -0.0174],
    [0.0030, 0.0120, 0.0027],
    [0.0637, 0.0470, -0.0127],
    [0.0360, 0.0661, -0.0030],
    [0.0159, 0.0181, 0.0082],
    [0.0132, 0.0326, 0.0169],
    [0.0191, 0.0261, 0.0136],
    [-0.0146, -0.0276, -0.0361],
    [0.0187, -0.0024, -0.0072],
    [-0.1059, -0.0090, 0.0350],
    [-0.0195, -0.0226, -0.0138],
    [-0.0295, 0.0024, -0.0215],
    [0.0191, -0.0393, -0.0001],
    [-0.0144, -0.0166, -0.0272],
    [0.0389, 0.0430, 0.0445],
    [-0.0153, -0.0336, 0.0031],
    [0.0339, 0.0122, 0.0220],
    [-0.0136, -0.0078, -0.0120],
    [-0.0340, -0.0282, -0.0245],
    [-0.0133, -0.0176, -0.0133],
    [0.0109, -0.0087, 0.0096],
    [-0.0010, 0.0044, 0.0016],
    [0.0301, 0.0053, 0.0361],
    [-0.0281, -0.0205, -0.0032],
    [-0.0725, 0.0002, 0.0160],
    [-0.0036, 0.0158, 0.0807],
    [0.0087, 0.0040, -0.0053],
    [-0.0260, 0.0183, -0.0077],
    [-0.0039, -0.0035, -0.0107],
    [-0.0026, 0.0172, 0.0237],
    [0.0088, 0.0078, 0.0078],
    [-0.0087, -0.0310, -0.0122],
    [-0.0027, 0.0018, 0.0094],
    [-0.0064, 0.0292, -0.0256],
    [0.0594, 0.1049, 0.1180],
    [0.0103, -0.0103, -0.0026],
    [-0.0091, 0.0025, -0.0015],
    [0.0178, 0.0243, 0.0292],
    [-0.0063, -0.0012, 0.0202],
    [0.0452, 0.0246, 0.0143],
    [0.0149, 0.0270, 0.0052],
    [0.1484, 0.0801, 0.0804],
    [-0.0120, 0.0040, 0.0010],
    [0.0181, 0.0051, -0.0021],
    [0.0132, 0.0050, 0.0019],
    [0.0291, 0.0020, 0.0092],
    [0.0066, -0.0410, -0.1314],
    [-0.1153, -0.0629, -0.0802],
    [0.0258, 0.0378, 0.0298],
    [0.0375, 0.1139, 0.0468],
    [-0.0142, -0.0126, -0.0276],
    [0.0339, 0.0153, 0.0138],
    [0.0346, 0.0211, 0.0267],
    [0.0369, -0.0431, -0.0993],
    [-0.0052, -0.0092, 0.0056],
    [-0.0279, 0.0410, -0.0357],
    [0.0036, 0.0017, -0.0083],
    [-0.0441, -0.0367, -0.0454],
    [-0.0001, -0.0092, -0.0001],
    [-0.0222, -0.0183, -0.0051],
    [0.0039, 0.0053, -0.0184],
    [-0.0094, -0.0075, -0.0143],
    [-0.0066, -0.0088, -0.0063],
    [0.0220, 0.0074, 0.0100],
];
const QWEN21_RGB_BIAS: [f32; 3] = [-0.1228, -0.1869, -0.3083];

/// One preview picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Steps finished, of `steps`.
    pub step: usize,
    pub steps: usize,
    /// Which result it shows (a split's layer, background first); 0 for a single picture.
    pub output: usize,
    pub width: usize,
    pub height: usize,
    /// Straight RGBA8, row-major, opaque.
    pub rgba: Vec<u8>,
}

/// What a pipeline's step loop calls after a step.
pub trait StepPreview {
    /// Whether the step that just finished (`step` of `steps`, from 1) should be shown. Cheap:
    /// the pipeline computes x0 only when this says yes.
    fn wants(&mut self, step: usize, steps: usize) -> bool;
    /// The denoised estimate after `step`: `(N, C, H, W)`, one row per result, in the
    /// sampler's normalised space. Never fails the run.
    fn show(&mut self, step: usize, steps: usize, x0: &Tensor);
    /// A batch of separate results (Upscale's tiles) is starting result `first`: the rows of
    /// the next pictures are numbered from it, and its first step may be shown at once.
    fn begin_output(&mut self, first: usize) {
        let _ = first;
    }
}

/// [`LatentSpace::factors`] as `(3, C)` and `(3, 1)` tensors on `dev`, f32.
fn factor_tensors(space: LatentSpace, dev: &Device) -> Result<(Tensor, Tensor)> {
    let (rows, bias) = space.factors();
    let c = rows.len();
    let mut t = vec![0f32; 3 * c];
    for (i, row) in rows.iter().enumerate() {
        for (k, v) in row.iter().enumerate() {
            if let Some(slot) = t.get_mut(k * c + i) {
                *slot = *v;
            }
        }
    }
    Ok((Tensor::from_vec(t, (3, c), dev)?, Tensor::from_slice(&bias, (3, 1), dev)?))
}

/// One latent `(C, H, W)` as RGB `(3, H, W)` in `[0, 1]` through the factors.
pub fn latent_to_rgb(space: LatentSpace, x0: &Tensor) -> Result<Tensor> {
    let (c, _, _) = x0.dims3()?;
    anyhow::ensure!(c == space.channels(), "a {c}-channel latent for a {}-channel space", space.channels());
    let x0 = if space == LatentSpace::Flux2 { unpack(x0)? } else { x0.clone() };
    let (c, h, w) = x0.dims3()?;
    let (m, b) = factor_tensors(space, x0.device())?;
    let flat = x0.to_dtype(DType::F32)?.reshape((c, h * w))?;
    let rgb = m.matmul(&flat)?.broadcast_add(&b)?;
    Ok(((rgb + 1.0)? * 0.5)?.clamp(0.0, 1.0)?.reshape((3, h, w))?)
}

/// `(3 or more, H, W)` in `[0, 1]` (the first three channels) as opaque RGBA8 bytes.
fn rgba_bytes(rgb: &Tensor) -> Result<(usize, usize, Vec<u8>)> {
    let (_, h, w) = rgb.dims3()?;
    let bytes = (rgb.narrow(0, 0, 3)?.to_dtype(DType::F32)?.clamp(0.0, 1.0)? * 255.0)?
        .round()?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let n = h * w;
    let mut rgba = vec![255u8; n * 4];
    for (i, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        for (k, slot) in px.iter_mut().take(3).enumerate() {
            *slot = bytes.get(k * n + i).copied().unwrap_or(0);
        }
    }
    Ok((w, h, rgba))
}

fn conv(vb: &VarBuilder, cin: usize, cout: usize, k: usize, bias: bool) -> Result<Conv2d> {
    let w = vb.get((cout, cin, k, k), "weight")?;
    let b = if bias { Some(vb.get(cout, "bias")?) } else { None };
    Ok(Conv2d::new(w, b, Conv2dConfig { padding: k / 2, ..Default::default() }))
}

/// `tanh(x / 3) · 3`, both decoders' first layer.
fn soft_clamp(x: &Tensor) -> Result<Tensor> {
    Ok(((x / 3.0)?.tanh()? * 3.0)?)
}

fn up2(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    Ok(x.upsample_nearest2d(h * 2, w * 2)?)
}

/// TAESD's residual block: `relu(conv(relu(conv(relu(conv(x))))) + x)` (`n → n`).
struct Block {
    c0: Conv2d,
    c2: Conv2d,
    c4: Conv2d,
}

impl Block {
    fn load(vb: &VarBuilder, n: usize) -> Result<Self> {
        Ok(Self { c0: conv(&vb.pp("conv.0"), n, n, 3, true)?, c2: conv(&vb.pp("conv.2"), n, n, 3, true)?, c4: conv(&vb.pp("conv.4"), n, n, 3, true)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.c4.forward(&self.c2.forward(&self.c0.forward(x)?.relu()?)?.relu()?)?;
        Ok((y + x)?.relu()?)
    }
}

/// TAEHV's block with memory: its first convolution sees `[x, past]` (`2n → n`). With no
/// past (the first frame) `past` is zeros, so only the convolution's `x` half matters:
/// `zero` keeps that half alone (half the work, no zeros to make).
struct MemBlock {
    c0: Conv2d,
    zero: Conv2d,
    c2: Conv2d,
    c4: Conv2d,
}

impl MemBlock {
    fn load(vb: &VarBuilder, n: usize) -> Result<Self> {
        let c0 = conv(&vb.pp("conv.0"), 2 * n, n, 3, true)?;
        let zero = Conv2d::new(c0.weight().narrow(1, 0, n)?.contiguous()?, c0.bias().cloned(), *c0.config());
        Ok(Self { c0, zero, c2: conv(&vb.pp("conv.2"), n, n, 3, true)?, c4: conv(&vb.pp("conv.4"), n, n, 3, true)? })
    }

    fn forward(&self, x: &Tensor, past: Option<&Tensor>) -> Result<Tensor> {
        let first = match past {
            Some(p) => self.c0.forward(&Tensor::cat(&[x, p], 1)?)?,
            None => self.zero.forward(x)?,
        };
        let y = self.c4.forward(&self.c2.forward(&first.relu()?)?.relu()?)?;
        Ok((y + x)?.relu()?)
    }
}

/// TAEHV's decoder for Wan 2.1 latents (`taew2_1`), one latent frame: its time upsampling
/// makes four frames of which the last is the picture (ComfyUI trims the first three).
struct Taehv {
    c1: Conv2d,
    s1: [MemBlock; 3],
    g7: Conv2d,
    c8: Conv2d,
    s2: [MemBlock; 3],
    g13: Conv2d,
    c14: Conv2d,
    s3: [MemBlock; 3],
    g19: Conv2d,
    c20: Conv2d,
    c22: Conv2d,
}

impl Taehv {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let mb = |i: usize, n: usize| MemBlock::load(&vb.pp(i.to_string()), n);
        Ok(Self {
            c1: conv(&vb.pp("1"), 16, 256, 3, true)?,
            s1: [mb(3, 256)?, mb(4, 256)?, mb(5, 256)?],
            g7: conv(&vb.pp("7").pp("conv"), 256, 256, 1, false)?,
            c8: conv(&vb.pp("8"), 256, 128, 3, false)?,
            s2: [mb(9, 128)?, mb(10, 128)?, mb(11, 128)?],
            g13: conv(&vb.pp("13").pp("conv"), 128, 256, 1, false)?,
            c14: conv(&vb.pp("14"), 128, 64, 3, false)?,
            s3: [mb(15, 64)?, mb(16, 64)?, mb(17, 64)?],
            g19: conv(&vb.pp("19").pp("conv"), 64, 128, 1, false)?,
            c20: conv(&vb.pp("20"), 64, 64, 3, false)?,
            c22: conv(&vb.pp("22"), 64, 3, 3, true)?,
        })
    }

    /// `(1, 16, h, w)` → `(1, 3, 8h, 8w)` in `[0, 1]`-ish.
    fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.c1.forward(&soft_clamp(z)?)?.relu()?;
        for b in &self.s1 {
            x = b.forward(&x, None)?;
        }
        x = self.c8.forward(&self.g7.forward(&up2(&x)?)?)?;
        for b in &self.s2 {
            x = b.forward(&x, None)?;
        }
        // Two frames from here: the first is the second's memory in each block.
        let g = self.g13.forward(&up2(&x)?)?;
        let mut f0 = self.c14.forward(&g.narrow(1, 0, 128)?)?;
        let mut f1 = self.c14.forward(&g.narrow(1, 128, 128)?)?;
        for b in &self.s3 {
            let next0 = b.forward(&f0, None)?;
            f1 = b.forward(&f1, Some(&f0))?;
            f0 = next0;
        }
        drop(f0);
        // The second frame's second half is the last of the four.
        let last = self.g19.forward(&up2(&f1)?)?.narrow(1, 64, 64)?;
        Ok(self.c22.forward(&self.c20.forward(&last)?.relu()?)?)
    }
}

/// TAESD's 16× decoder for Qwen-Image-2.1 latents (`taeqi2_1`, diffusers' key layout:
/// `decoder.layers.N` is TAESD's layer N + 1). RGBA out; the preview keeps RGB.
struct TaesdF16 {
    c0: Conv2d,
    s1: [Block; 3],
    c6: Conv2d,
    s2: [Block; 3],
    c11: Conv2d,
    s3: [Block; 3],
    c16: Conv2d,
    b17: Block,
    c18: Conv2d,
}

impl TaesdF16 {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let b = |i: usize, n: usize| Block::load(&vb.pp(i.to_string()), n);
        Ok(Self {
            c0: conv(&vb.pp("0"), 64, 256, 3, true)?,
            s1: [b(2, 256)?, b(3, 256)?, b(4, 256)?],
            c6: conv(&vb.pp("6"), 256, 128, 3, false)?,
            s2: [b(7, 128)?, b(8, 128)?, b(9, 128)?],
            c11: conv(&vb.pp("11"), 128, 64, 3, false)?,
            s3: [b(12, 64)?, b(13, 64)?, b(14, 64)?],
            c16: conv(&vb.pp("16"), 64, 64, 3, false)?,
            b17: b(17, 64)?,
            c18: conv(&vb.pp("18"), 64, 16, 3, true)?,
        })
    }

    /// `(1, 64, h, w)` → `(1, 3, 16h, 16w)` in `[0, 1]`-ish.
    fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.c0.forward(&soft_clamp(z)?)?.relu()?;
        for b in &self.s1 {
            x = b.forward(&x)?;
        }
        x = self.c6.forward(&up2(&x)?)?;
        for b in &self.s2 {
            x = b.forward(&x)?;
        }
        x = self.c11.forward(&up2(&x)?)?;
        for b in &self.s3 {
            x = b.forward(&x)?;
        }
        x = self.b17.forward(&self.c16.forward(&up2(&x)?)?)?;
        // PixelShuffle(2) of the RGB channels alone: channel 4c + 2i + j → (c, 2y + i, 2x + j).
        let rgb = self.c18.forward(&x)?.narrow(1, 0, 12)?;
        let (b, _, h, w) = rgb.dims4()?;
        Ok(rgb.reshape((b, 3, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.contiguous()?.reshape((b, 3, h * 2, w * 2))?)
    }
}

/// TAESD's block with a mid-block pooled branch (`x += conv(relu(gn(conv(x))))`, the 4-group
/// GroupNorm at 4× the width) in front of the plain block.
struct PoolBlock {
    block: Block,
    p0: Conv2d,
    gn: candle_nn::GroupNorm,
    p3: Conv2d,
}

impl PoolBlock {
    fn load(vb: &VarBuilder, n: usize) -> Result<Self> {
        Ok(Self {
            block: Block::load(vb, n)?,
            p0: conv(&vb.pp("pool.0"), n, 4 * n, 1, false)?,
            gn: candle_nn::group_norm(4, 4 * n, 1e-5, vb.pp("pool.1"))?,
            p3: conv(&vb.pp("pool.3"), 4 * n, n, 1, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pooled = self.p3.forward(&self.gn.forward(&self.p0.forward(x)?)?.relu()?)?;
        self.block.forward(&(x + pooled)?)
    }
}

/// madebyollin's TAEF2 decoder for FLUX.2 latents (`taef2`, MIT; `decoder.layers.N` is TAESD's
/// layer N + 1): the unpacked 32 channels at 8× in, RGB out.
struct Taef2 {
    c0: Conv2d,
    s1: [PoolBlock; 3],
    c6: Conv2d,
    s2: [Block; 3],
    c11: Conv2d,
    s3: [Block; 3],
    c16: Conv2d,
    b17: Block,
    c18: Conv2d,
}

impl Taef2 {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let b = |i: usize| Block::load(&vb.pp(i.to_string()), 64);
        let pb = |i: usize| PoolBlock::load(&vb.pp(i.to_string()), 64);
        Ok(Self {
            c0: conv(&vb.pp("0"), 32, 64, 3, true)?,
            s1: [pb(2)?, pb(3)?, pb(4)?],
            c6: conv(&vb.pp("6"), 64, 64, 3, false)?,
            s2: [b(7)?, b(8)?, b(9)?],
            c11: conv(&vb.pp("11"), 64, 64, 3, false)?,
            s3: [b(12)?, b(13)?, b(14)?],
            c16: conv(&vb.pp("16"), 64, 64, 3, false)?,
            b17: b(17)?,
            c18: conv(&vb.pp("18"), 64, 3, 3, true)?,
        })
    }

    /// `(1, 128, h, w)` → `(1, 3, 16h, 16w)` in `[0, 1]`-ish.
    fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.c0.forward(&soft_clamp(&unpack(z)?)?)?.relu()?;
        for b in &self.s1 {
            x = b.forward(&x)?;
        }
        x = self.c6.forward(&up2(&x)?)?;
        for b in &self.s2 {
            x = b.forward(&x)?;
        }
        x = self.c11.forward(&up2(&x)?)?;
        for b in &self.s3 {
            x = b.forward(&x)?;
        }
        x = self.b17.forward(&self.c16.forward(&up2(&x)?)?)?;
        Ok(self.c18.forward(&x)?)
    }
}

enum TinyNet {
    Taehv(Box<Taehv>),
    TaesdF16(Box<TaesdF16>),
    Taef2(Box<Taef2>),
}

/// A tiny decoder on the device, for Sharp previews.
pub struct TinyDecoder {
    space: LatentSpace,
    dtype: DType,
    dev: Device,
    net: TinyNet,
}

impl TinyDecoder {
    /// The file for `space` under `root/vae_approx/`, when there.
    pub fn file(root: &Path, space: LatentSpace) -> Option<PathBuf> {
        Some(root.join(TINY_DIR).join(space.tiny_file()?)).filter(|p| p.is_file())
    }

    /// Load `space`'s decoder from `file` onto `dev` (bf16 on CUDA, f32 on the CPU).
    pub fn load(file: &Path, space: LatentSpace, dev: &Device) -> Result<Self> {
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[file], dtype, dev)? };
        Self::from_vb(&vb, space, dtype, dev)
    }

    /// From weights already in a `VarBuilder` (`decoder.*` for TAEHV, `decoder.layers.*` for TAESD).
    pub fn from_vb(vb: &VarBuilder, space: LatentSpace, dtype: DType, dev: &Device) -> Result<Self> {
        let net = match space {
            LatentSpace::Wan21 => TinyNet::Taehv(Box::new(Taehv::load(&vb.pp("decoder"))?)),
            LatentSpace::QwenImage21 => TinyNet::TaesdF16(Box::new(TaesdF16::load(&vb.pp("decoder").pp("layers"))?)),
            LatentSpace::Layered => anyhow::bail!("no tiny decoder decodes the Layered VAE's latents"),
            LatentSpace::Flux2 => TinyNet::Taef2(Box::new(Taef2::load(&vb.pp("decoder").pp("layers"))?)),
        };
        Ok(Self { space, dtype, dev: dev.clone(), net })
    }

    pub fn space(&self) -> LatentSpace {
        self.space
    }

    /// Whether the weights live on `dev` (a pipeline loaded again has a device of its own).
    pub fn on(&self, dev: &Device) -> bool {
        self.dev.same_device(dev)
    }

    /// One latent `(C, h, w)` as RGB `(3, s·h, s·w)` in `[0, 1]`.
    pub fn decode(&self, x0: &Tensor) -> Result<Tensor> {
        Ok(self.decode_raw(x0)?.clamp(0.0, 1.0)?)
    }

    /// [`Self::decode`] before the clamp, f32.
    fn decode_raw(&self, x0: &Tensor) -> Result<Tensor> {
        let z = x0.to_dtype(self.dtype)?.unsqueeze(0)?;
        let rgb = match &self.net {
            TinyNet::Taehv(n) => n.decode(&z)?,
            TinyNet::TaesdF16(n) => n.decode(&z)?,
            TinyNet::Taef2(n) => n.decode(&z)?,
        };
        Ok(rgb.squeeze(0)?.to_dtype(DType::F32)?)
    }

    /// Device bytes a decode of a `pixels`-pixel picture may take at its peak (candle's
    /// im2col of the widest convolution at full size, its output and a margin).
    pub fn peak_bytes(&self, pixels: usize) -> usize {
        let per_pixel = match self.space {
            // 64 channels × 9 taps at full size (bf16), plus two 64-channel activations.
            LatentSpace::Wan21 | LatentSpace::Layered | LatentSpace::Flux2 => 64 * 9 * 2 + 4 * 64 * 2,
            // Its widest convolutions run at half size.
            LatentSpace::QwenImage21 => (64 * 9 * 2 + 4 * 64 * 2) / 4 + 16 * 4,
        };
        pixels.saturating_mul(per_pixel).saturating_add(512 << 20)
    }
}

/// The [`StepPreview`] the engine hands a pipeline: renders x0 as `mode` asks and gives each
/// picture to `sink`.
pub struct Previewer<'a> {
    mode: PreviewMode,
    space: LatentSpace,
    tiny: Option<&'a TinyDecoder>,
    sink: &'a mut dyn FnMut(Frame),
    last: Option<Instant>,
    /// The output number of row 0 ([`StepPreview::begin_output`]).
    base: usize,
    /// Set when a Sharp decode failed: Fast for the rest of the run.
    sharp_failed: bool,
    /// Pictures sent and the time spent making them (the run's log line).
    pub shown: usize,
    pub spent: Duration,
}

impl<'a> Previewer<'a> {
    /// `tiny` is used only in Sharp mode and only when it decodes `space`.
    pub fn new(mode: PreviewMode, space: LatentSpace, tiny: Option<&'a TinyDecoder>, sink: &'a mut dyn FnMut(Frame)) -> Self {
        let tiny = tiny.filter(|t| t.space == space && mode == PreviewMode::Sharp);
        Self { mode, space, tiny, sink, last: None, base: 0, sharp_failed: false, shown: 0, spent: Duration::ZERO }
    }

    fn sharp(&self) -> bool {
        self.tiny.is_some() && !self.sharp_failed
    }

    fn render(&self, x0: &Tensor, use_tiny: bool) -> Result<(usize, usize, Vec<u8>)> {
        match (use_tiny, self.tiny) {
            (true, Some(t)) => rgba_bytes(&t.decode(x0)?),
            _ => rgba_bytes(&latent_to_rgb(self.space, x0)?),
        }
    }
}

impl StepPreview for Previewer<'_> {
    fn wants(&mut self, step: usize, steps: usize) -> bool {
        if self.mode == PreviewMode::Off || step == 0 || step >= steps {
            // The last step's picture is the result itself, decoded properly.
            return false;
        }
        if self.sharp() && !step.is_multiple_of(SHARP_EVERY) {
            return false;
        }
        self.last.is_none_or(|t| t.elapsed() >= MIN_GAP)
    }

    fn show(&mut self, step: usize, steps: usize, x0: &Tensor) {
        let t0 = Instant::now();
        self.last = Some(t0);
        let Ok((n, _, h, w)) = x0.dims4() else {
            tracing::warn!(shape = ?x0.dims(), "preview: x0 is not (N, C, H, W)");
            return;
        };
        let pixels = h * w * self.space.scale() * self.space.scale();
        for output in 0..n {
            // Sharp only when the card has room for the decode; else Fast for the rest of the run
            // (alternating sharp and coarse pictures would flicker).
            let room = self.tiny.is_some_and(|t| crate::device::free_vram(x0.device()).is_none_or(|(free, _)| free >= t.peak_bytes(pixels)));
            if self.sharp() && !room {
                tracing::info!("preview: no room on the card for the tiny decoder; Fast for the rest of the run");
                self.sharp_failed = true;
            }
            let use_tiny = self.sharp();
            let one = match x0.get(output) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "preview: no row {output}");
                    return;
                }
            };
            let made = match self.render(&one, use_tiny) {
                Err(e) if use_tiny => {
                    tracing::warn!(error = %e, "preview: the tiny decoder failed; Fast for the rest of the run");
                    self.sharp_failed = true;
                    self.render(&one, false)
                }
                other => other,
            };
            match made {
                Ok((width, height, rgba)) => {
                    self.shown += 1;
                    (self.sink)(Frame { step, steps, output: self.base + output, width, height, rgba });
                }
                Err(e) => tracing::warn!(error = %e, "preview skipped"),
            }
        }
        self.spent += t0.elapsed();
    }

    fn begin_output(&mut self, first: usize) {
        self.base = first;
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn fast_is_the_factor_map() {
        let x0 = rand(&[16, 2, 3], 1);
        let rgb = latent_to_rgb(LatentSpace::Wan21, &x0).unwrap();
        assert_eq!(rgb.dims(), &[3, 2, 3]);
        let got = rgb.to_vec3::<f32>().unwrap();
        let x = x0.to_vec3::<f32>().unwrap();
        for k in 0..3 {
            for y in 0..2 {
                for xx in 0..3 {
                    let v: f32 = WAN21_RGB_BIAS[k] + (0..16).map(|c| WAN21_RGB[c][k] * x[c][y][xx]).sum::<f32>();
                    let want = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                    assert!((got[k][y][xx] - want).abs() < 1e-6);
                }
            }
        }
        let (w, h, rgba) = rgba_bytes(&rgb).unwrap();
        assert_eq!((w, h, rgba.len()), (3, 2, 24));
        assert!(rgba.chunks(4).all(|p| p[3] == 255));
        assert_eq!(rgba[0], (got[0][0][0] * 255.0).round() as u8);
        assert!(latent_to_rgb(LatentSpace::QwenImage21, &x0).is_err(), "16 channels are not Qwen-Image-2.1's 64");
    }

    /// Random weights for every tensor a decoder asks for.
    fn weights(space: LatentSpace) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let mut seed = 7;
        let mut put = |name: String, shape: &[usize]| {
            seed += 1;
            map.insert(name, rand(shape, seed));
        };
        let convw = |put: &mut dyn FnMut(String, &[usize]), p: &str, cin: usize, cout: usize, k: usize, bias: bool| {
            put(format!("{p}.weight"), &[cout, cin, k, k]);
            if bias {
                put(format!("{p}.bias"), &[cout]);
            }
        };
        match space {
            LatentSpace::Wan21 => {
                convw(&mut put, "decoder.1", 16, 256, 3, true);
                for (i, n) in [(3, 256), (4, 256), (5, 256), (9, 128), (10, 128), (11, 128), (15, 64), (16, 64), (17, 64)] {
                    convw(&mut put, &format!("decoder.{i}.conv.0"), 2 * n, n, 3, true);
                    convw(&mut put, &format!("decoder.{i}.conv.2"), n, n, 3, true);
                    convw(&mut put, &format!("decoder.{i}.conv.4"), n, n, 3, true);
                }
                convw(&mut put, "decoder.7.conv", 256, 256, 1, false);
                convw(&mut put, "decoder.8", 256, 128, 3, false);
                convw(&mut put, "decoder.13.conv", 128, 256, 1, false);
                convw(&mut put, "decoder.14", 128, 64, 3, false);
                convw(&mut put, "decoder.19.conv", 64, 128, 1, false);
                convw(&mut put, "decoder.20", 64, 64, 3, false);
                convw(&mut put, "decoder.22", 64, 3, 3, true);
            }
            LatentSpace::QwenImage21 => {
                convw(&mut put, "decoder.layers.0", 64, 256, 3, true);
                for (i, n) in [(2, 256), (3, 256), (4, 256), (7, 128), (8, 128), (9, 128), (12, 64), (13, 64), (14, 64), (17, 64)] {
                    for c in [0, 2, 4] {
                        convw(&mut put, &format!("decoder.layers.{i}.conv.{c}"), n, n, 3, true);
                    }
                }
                convw(&mut put, "decoder.layers.6", 256, 128, 3, false);
                convw(&mut put, "decoder.layers.11", 128, 64, 3, false);
                convw(&mut put, "decoder.layers.16", 64, 64, 3, false);
                convw(&mut put, "decoder.layers.18", 64, 16, 3, true);
            }
            LatentSpace::Flux2 => {
                convw(&mut put, "decoder.layers.0", 32, 64, 3, true);
                for i in [2, 3, 4, 7, 8, 9, 12, 13, 14, 17] {
                    for c in [0, 2, 4] {
                        convw(&mut put, &format!("decoder.layers.{i}.conv.{c}"), 64, 64, 3, true);
                    }
                }
                for i in [2, 3, 4] {
                    convw(&mut put, &format!("decoder.layers.{i}.pool.0"), 64, 256, 1, false);
                    put(format!("decoder.layers.{i}.pool.1.weight"), &[256]);
                    put(format!("decoder.layers.{i}.pool.1.bias"), &[256]);
                    convw(&mut put, &format!("decoder.layers.{i}.pool.3"), 256, 64, 1, false);
                }
                for i in [6, 11, 16] {
                    convw(&mut put, &format!("decoder.layers.{i}"), 64, 64, 3, false);
                }
                convw(&mut put, "decoder.layers.18", 64, 3, 3, true);
            }
            LatentSpace::Layered => {}
        }
        map
    }

    /// FLUX.2's Fast picture is the factor map of the unpacked latent: channel `4c + 2i + j` of
    /// a cell at `(2y + i, 2x + j)` of channel `c`, then the 32-channel factors.
    #[test]
    fn flux2_fast_unpacks_then_maps() {
        let x0 = rand(&[128, 2, 3], 3);
        let rgb = latent_to_rgb(LatentSpace::Flux2, &x0).unwrap();
        assert_eq!(rgb.dims(), &[3, 4, 6]);
        let got = rgb.to_vec3::<f32>().unwrap();
        let x = x0.to_vec3::<f32>().unwrap();
        for (k, plane) in got.iter().enumerate() {
            for (py, row) in plane.iter().enumerate() {
                for (px, value) in row.iter().enumerate() {
                    let (y, i, xx, j) = (py / 2, py % 2, px / 2, px % 2);
                    let v: f32 = FLUX2_RGB_BIAS[k] + (0..32).map(|c| FLUX2_RGB[c][k] * x[4 * c + 2 * i + j][y][xx]).sum::<f32>();
                    let want = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                    assert!((value - want).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn the_taef2_decodes_to_the_picture_size() {
        let vb = VarBuilder::from_tensors(weights(LatentSpace::Flux2), DType::F32, &Device::Cpu);
        let tiny = TinyDecoder::from_vb(&vb, LatentSpace::Flux2, DType::F32, &Device::Cpu).unwrap();
        let out = tiny.decode(&rand(&[128, 2, 3], 5)).unwrap();
        assert_eq!(out.dims(), &[3, 32, 48]);
    }

    /// TAEHV the way its reference runs a clip: every frame through every layer in turn, each
    /// memory block seeing the previous frame's input (zeros for the first), the four output
    /// frames of one latent frame, the last kept.
    fn taehv_reference(w: &HashMap<String, Tensor>, z: &Tensor) -> Tensor {
        let c = |name: &str, x: &Tensor, pad: usize| {
            let y = x.conv2d(&w[&format!("decoder.{name}.weight")], pad, 1, 1, 1).unwrap();
            match w.get(&format!("decoder.{name}.bias")) {
                Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0).unwrap(), 1, 1)).unwrap()).unwrap(),
                None => y,
            }
        };
        let memblock = |i: usize, x: &Tensor, past: &Tensor| {
            let cat = Tensor::cat(&[x, past], 1).unwrap();
            let y = c(&format!("{i}.conv.0"), &cat, 1).relu().unwrap();
            let y = c(&format!("{i}.conv.2"), &y, 1).relu().unwrap();
            let y = c(&format!("{i}.conv.4"), &y, 1);
            (y + x).unwrap().relu().unwrap()
        };
        // Frames as a list; a time-grow layer splits each frame's channels into frames.
        let mut frames = vec![c("1", &soft_clamp(z).unwrap(), 1).relu().unwrap()];
        let stage = |frames: Vec<Tensor>, blocks: [usize; 3]| -> Vec<Tensor> {
            let mut frames = frames;
            for i in blocks {
                let mut past = frames[0].zeros_like().unwrap();
                frames = frames
                    .into_iter()
                    .map(|f| {
                        let y = memblock(i, &f, &past);
                        past = f;
                        y
                    })
                    .collect();
            }
            frames
        };
        let grow = |frames: Vec<Tensor>, i: usize, stride: usize| -> Vec<Tensor> {
            frames
                .iter()
                .flat_map(|f| {
                    let g = c(&format!("{i}.conv"), &up2(f).unwrap(), 0);
                    let n = g.dim(1).unwrap() / stride;
                    (0..stride).map(move |k| g.narrow(1, k * n, n).unwrap()).collect::<Vec<_>>()
                })
                .collect()
        };
        frames = stage(frames, [3, 4, 5]);
        frames = grow(frames, 7, 1).iter().map(|f| c("8", f, 1)).collect();
        frames = stage(frames, [9, 10, 11]);
        frames = grow(frames, 13, 2).iter().map(|f| c("14", f, 1)).collect();
        frames = stage(frames, [15, 16, 17]);
        let out: Vec<Tensor> = grow(frames, 19, 2).iter().map(|f| c("22", &c("20", f, 1).relu().unwrap(), 1)).collect();
        assert_eq!(out.len(), 4);
        out[3].clone()
    }

    #[test]
    fn the_single_frame_taehv_is_the_last_of_the_clips_four() {
        let w = weights(LatentSpace::Wan21);
        let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        let tiny = TinyDecoder::from_vb(&vb, LatentSpace::Wan21, DType::F32, &Device::Cpu).unwrap();
        let z = (rand(&[16, 3, 2], 3) * 20.0).unwrap();
        let ours = tiny.decode_raw(&z).unwrap();
        assert_eq!(ours.dims(), &[3, 24, 16]);
        let want = taehv_reference(&w, &z.unsqueeze(0).unwrap()).squeeze(0).unwrap();
        let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let diff = (ours - want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(scale > 0.0 && diff <= 1e-5 * scale, "max difference {diff} of {scale}");
    }

    #[test]
    fn the_16x_taesd_decodes_to_the_picture_size() {
        let vb = VarBuilder::from_tensors(weights(LatentSpace::QwenImage21), DType::F32, &Device::Cpu);
        let tiny = TinyDecoder::from_vb(&vb, LatentSpace::QwenImage21, DType::F32, &Device::Cpu).unwrap();
        let out = tiny.decode(&rand(&[64, 2, 3], 5)).unwrap();
        assert_eq!(out.dims(), &[3, 32, 48]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|p| (0.0..=1.0).contains(p)));
    }

    #[test]
    fn the_previewer_keeps_its_cadence_and_sends_one_picture_per_result() {
        let mut got = Vec::new();
        {
            let mut sink = |f: Frame| got.push((f.step, f.output, f.width, f.height));
            let mut p = Previewer::new(PreviewMode::Fast, LatentSpace::Wan21, None, &mut sink);
            assert!(p.wants(1, 8) && !p.wants(8, 8), "every step but the last");
            p.show(1, 8, &rand(&[2, 16, 3, 4], 9));
            assert!(!p.wants(2, 8), "throttled right after a picture");
        }
        assert_eq!(got, vec![(1, 0, 4, 3), (1, 1, 4, 3)]);
        let mut sink = |_: Frame| {};
        let mut off = Previewer::new(PreviewMode::Off, LatentSpace::Wan21, None, &mut sink);
        assert!(!off.wants(1, 8));
        // Sharp with no tiny decoder is Fast: every step.
        let mut sink = |_: Frame| {};
        let mut sharp = Previewer::new(PreviewMode::Sharp, LatentSpace::Wan21, None, &mut sink);
        assert!(sharp.wants(1, 8));
    }
}
