//! The FLUX.2 VAE's decoder on MLX, as `qwen_image_rs::moss::vae::Flux2Decoder` runs it: the
//! latent's batch-norm statistics undone and its 2×2 packing unpacked, then `post_quant_conv`
//! and an SD-style decoder (a mid block with one single-head attention, up blocks of three
//! resnets). Channels last (MLX's convolutions); the widths are read from the file.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype, fast, ops};

use crate::nn::{Params, from_candle, get};

/// Packed latent channels and the decoder's (before the 2×2 packing).
const PACKED: i32 = 128;
const LATENT: i32 = 32;
const BN_EPS: f32 = 1e-4;
const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;

/// The decoder's and encoder's tensors from the VAE file, in `dtype` (statistics in f32),
/// convolution weights as MLX takes them `(out, kh, kw, in)`.
pub fn convert(path: &Path, dtype: Dtype) -> Result<Params> {
    let file: HashMap<String, candle_core::Tensor> =
        candle_core::safetensors::load(path, &candle_core::Device::Cpu).with_context(|| format!("read {}", path.display()))?;
    let mut p = Params::new();
    for (name, t) in file {
        if !["decoder.", "encoder.", "post_quant_conv.", "quant_conv.", "bn.running_"].iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let a = from_candle(&t)?;
        let a = if a.ndim() == 4 { a.transpose_axes(&[0, 2, 3, 1])? } else { a };
        let a = if name.starts_with("bn.") { a } else { a.as_dtype(dtype)? };
        a.eval()?;
        p.insert(name, a);
    }
    Ok(p)
}

struct Conv {
    w: Array,
    b: Array,
    pad: i32,
}

impl Conv {
    fn load(p: &Params, m: &str) -> Result<Self> {
        let w = get(p, &format!("{m}.weight"))?;
        let pad = w.shape().get(1).copied().unwrap_or(1) / 2;
        Ok(Self { w, b: get(p, &format!("{m}.bias"))?, pad })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
        // MLX unfolds a wide 3×3 convolution into one matrix (6 GB for 192 channels at 1024²
        // in f32): bands of rows with a one-row halo each are the same sum in a fraction of it.
        if self.pad != 1 || i64::from(h) * i64::from(w) * i64::from(c) <= BAND_ELEMENTS {
            return Ok(ops::conv2d(x, &self.w, (1, 1), (self.pad, self.pad), None, None)?.add(&self.b)?);
        }
        let zero = Array::zeros::<f32>(&[n, 1, w, c])?.as_dtype(x.dtype())?;
        let xp = ops::concatenate(&[&zero, x, &zero], 1)?;
        let band = i32::try_from((BAND_ELEMENTS / (i64::from(w) * i64::from(c))).max(1))?;
        let mut outs = Vec::new();
        let mut r0 = 0;
        while r0 < h {
            let r1 = (r0 + band).min(h);
            let y = ops::conv2d(xp.try_index((.., r0..r1 + 2))?, &self.w, (1, 1), (0, 1), None, None)?;
            y.eval()?;
            outs.push(y);
            r0 = r1;
        }
        Ok(ops::concatenate(&outs, 1)?.add(&self.b)?)
    }
}

/// The largest input (elements) a 3×3 convolution takes in one go.
const BAND_ELEMENTS: i64 = 256 * 1024 * 96;

struct GroupNorm {
    w: Array,
    b: Array,
}

impl GroupNorm {
    fn load(p: &Params, m: &str) -> Result<Self> {
        Ok(Self { w: get(p, &format!("{m}.weight"))?, b: get(p, &format!("{m}.bias"))? })
    }

    /// `x (N, H, W, C)`: each group's channels over every pixel normalised (MLX's fused layer
    /// norm, f32 statistics), then the affine.
    fn forward(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
        let g = x.reshape(&[n, h * w, GROUPS, c / GROUPS])?.transpose_axes(&[0, 2, 1, 3])?.reshape(&[n, GROUPS, h * w * (c / GROUPS)])?;
        let y = fast::layer_norm(&g, None, None, EPS)?.reshape(&[n, GROUPS, h * w, c / GROUPS])?.transpose_axes(&[0, 2, 1, 3])?.reshape(&[n, h, w, c])?;
        Ok(y.multiply(&self.w)?.add(&self.b)?)
    }
}

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv,
    norm2: GroupNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl Resnet {
    fn load(p: &Params, m: &str) -> Result<Self> {
        let shortcut = if p.contains_key(&format!("{m}.conv_shortcut.weight")) { Some(Conv::load(p, &format!("{m}.conv_shortcut"))?) } else { None };
        Ok(Self {
            norm1: GroupNorm::load(p, &format!("{m}.norm1"))?,
            conv1: Conv::load(p, &format!("{m}.conv1"))?,
            norm2: GroupNorm::load(p, &format!("{m}.norm2"))?,
            conv2: Conv::load(p, &format!("{m}.conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv1.forward(&mlx_rs::nn::silu(self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&mlx_rs::nn::silu(self.norm2.forward(&h)?)?)?;
        let x = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok(x.add(&h)?)
    }
}

/// The mid block's attention: one head over every pixel.
struct Attention {
    norm: GroupNorm,
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    out: (Array, Array),
}

impl Attention {
    fn load(p: &Params, m: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<(Array, Array)> { Ok((get(p, &format!("{m}.{n}.weight"))?, get(p, &format!("{m}.{n}.bias"))?)) };
        Ok(Self { norm: GroupNorm::load(p, &format!("{m}.group_norm"))?, q: lin("to_q")?, k: lin("to_k")?, v: lin("to_v")?, out: lin("to_out.0")? })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
        let y = self.norm.forward(x)?.reshape(&[n, h * w, c])?;
        let lin = |l: &(Array, Array), t: &Array| -> Result<Array> { Ok(t.matmul(l.0.t())?.add(&l.1)?) };
        let (q, k, v) = (lin(&self.q, &y)?, lin(&self.k, &y)?, lin(&self.v, &y)?);
        // ponytail: queries in blocks of 2048: a head of 384 has no fused kernel, and at 1024² the
        // whole score matrix would be 16384² floats.
        let kt = k.transpose_axes(&[0, 2, 1])?;
        let scale = Array::from_f32(1.0 / (c as f32).sqrt()).as_dtype(q.dtype())?;
        let mut outs = Vec::new();
        let tokens = h * w;
        let mut r0 = 0;
        while r0 < tokens {
            let r1 = (r0 + 2048).min(tokens);
            let qc = q.try_index((.., r0..r1))?;
            let scores = qc.matmul(&kt)?.multiply(&scale)?.as_dtype(Dtype::Float32)?;
            let o = ops::softmax_axis(&scores, -1, None)?.as_dtype(v.dtype())?.matmul(&v)?;
            o.eval()?;
            outs.push(o);
            r0 = r1;
        }
        let o = ops::concatenate(&outs, 1)?;
        let o = lin(&self.out, &o)?.reshape(&[n, h, w, c])?;
        Ok(x.add(&o)?)
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>,
}

pub struct Flux2Decoder {
    bn_std: Array,
    bn_mean: Array,
    post_quant_conv: Conv,
    conv_in: Conv,
    mid: (Resnet, Attention, Resnet),
    up: Vec<UpBlock>,
    norm_out: GroupNorm,
    conv_out: Conv,
    dtype: Dtype,
}

impl Flux2Decoder {
    pub fn load(p: &Params, dtype: Dtype) -> Result<Self> {
        let bn_std = get(p, "bn.running_var")?.add(Array::from_f32(BN_EPS))?.sqrt()?.reshape(&[1, PACKED, 1, 1])?;
        let bn_mean = get(p, "bn.running_mean")?.reshape(&[1, PACKED, 1, 1])?;
        let d = "decoder";
        let mut up = Vec::new();
        for i in 0.. {
            let b = format!("{d}.up_blocks.{i}");
            if !p.contains_key(&format!("{b}.resnets.0.conv1.weight")) {
                break;
            }
            let resnets = (0..)
                .map(|j| format!("{b}.resnets.{j}"))
                .take_while(|r| p.contains_key(&format!("{r}.conv1.weight")))
                .map(|r| Resnet::load(p, &r))
                .collect::<Result<_>>()?;
            let upsample =
                if p.contains_key(&format!("{b}.upsamplers.0.conv.weight")) { Some(Conv::load(p, &format!("{b}.upsamplers.0.conv"))?) } else { None };
            up.push(UpBlock { resnets, upsample });
        }
        anyhow::ensure!(!up.is_empty(), "the VAE file has no decoder up blocks");
        let m = format!("{d}.mid_block");
        Ok(Self {
            bn_std,
            bn_mean,
            post_quant_conv: Conv::load(p, "post_quant_conv")?,
            conv_in: Conv::load(p, &format!("{d}.conv_in"))?,
            mid: (Resnet::load(p, &format!("{m}.resnets.0"))?, Attention::load(p, &format!("{m}.attentions.0"))?, Resnet::load(p, &format!("{m}.resnets.1"))?),
            up,
            norm_out: GroupNorm::load(p, &format!("{d}.conv_norm_out"))?,
            conv_out: Conv::load(p, &format!("{d}.conv_out"))?,
            dtype,
        })
    }

    /// RGB `(1, 3, 16h, 16w)` f32 in [-1, 1] for the sampler's latent `(1, 128, h, w)`.
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let s = z.shape();
        anyhow::ensure!(s.len() == 4 && s[1] == PACKED, "a latent of shape {s:?} (the decoder takes (1, {PACKED}, h, w))");
        let (b, h, w) = (s[0], s[2], s[3]);
        // Unpack: channel `4c + 2i + j` of pixel (y, x) to (2y + i, 2x + j) of channel c, channels last.
        let z = z.as_dtype(Dtype::Float32)?.multiply(&self.bn_std)?.add(&self.bn_mean)?;
        let z = z.reshape(&[b, LATENT, 2, 2, h, w])?.transpose_axes(&[0, 4, 2, 5, 3, 1])?.reshape(&[b, 2 * h, 2 * w, LATENT])?.as_dtype(self.dtype)?;
        let x = self.conv_in.forward(&self.post_quant_conv.forward(&z)?)?;
        let x = self.mid.2.forward(&self.mid.1.forward(&self.mid.0.forward(&x)?)?)?;
        let mut x = x;
        for block in &self.up {
            // A resnet at a time: at 1024² each one's temporaries are hundreds of MB.
            for r in &block.resnets {
                x = r.forward(&x)?;
                x.eval()?;
            }
            if let Some(conv) = &block.upsample {
                x = conv.forward(&upsample2(&x)?)?;
                x.eval()?;
            }
        }
        let x = self.conv_out.forward(&mlx_rs::nn::silu(self.norm_out.forward(&x)?)?)?;
        Ok(ops::clip(x.as_dtype(Dtype::Float32)?, (-1.0f32, 1.0f32))?.transpose_axes(&[0, 3, 1, 2])?)
    }
}

/// A down block's resnets and its stride-2 convolution's weight and bias (all but the last).
type DownBlock = (Vec<Resnet>, Option<(Array, Array)>);

/// The FLUX.2 VAE's encoder (diffusers' layout, as `qwen_image_rs::moss::vae::Flux2Encoder`
/// reads it): a picture to the posterior's mean, 2×2-packed to 128 channels at 1/16 and
/// batch-normalised, the space MOSS-Image samples in (MOSS Edit's references).
pub struct Flux2Encoder {
    bn_std: Array,
    bn_mean: Array,
    conv_in: Conv,
    down: Vec<DownBlock>,
    mid: (Resnet, Attention, Resnet),
    norm_out: GroupNorm,
    conv_out: Conv,
    quant_conv: Conv,
    dtype: Dtype,
}

impl Flux2Encoder {
    pub fn load(p: &Params, dtype: Dtype) -> Result<Self> {
        anyhow::ensure!(p.contains_key("quant_conv.weight"), "the VAE file has no top-level quant_conv: no FLUX.2 encoder (flux2-vae_small has one)");
        let e = "encoder";
        let mut down = Vec::new();
        for i in 0.. {
            let b = format!("{e}.down_blocks.{i}");
            if !p.contains_key(&format!("{b}.resnets.0.conv1.weight")) {
                break;
            }
            let resnets = (0..)
                .map(|j| format!("{b}.resnets.{j}"))
                .take_while(|r| p.contains_key(&format!("{r}.conv1.weight")))
                .map(|r| Resnet::load(p, &r))
                .collect::<Result<_>>()?;
            let ds = format!("{b}.downsamplers.0.conv");
            let downsample =
                if p.contains_key(&format!("{ds}.weight")) { Some((get(p, &format!("{ds}.weight"))?, get(p, &format!("{ds}.bias"))?)) } else { None };
            down.push((resnets, downsample));
        }
        anyhow::ensure!(!down.is_empty(), "the VAE file has no encoder (encoder.down_blocks.*)");
        let m = format!("{e}.mid_block");
        Ok(Self {
            bn_std: get(p, "bn.running_var")?.add(Array::from_f32(BN_EPS))?.sqrt()?.reshape(&[1, PACKED, 1, 1])?,
            bn_mean: get(p, "bn.running_mean")?.reshape(&[1, PACKED, 1, 1])?,
            conv_in: Conv::load(p, &format!("{e}.conv_in"))?,
            down,
            mid: (Resnet::load(p, &format!("{m}.resnets.0"))?, Attention::load(p, &format!("{m}.attentions.0"))?, Resnet::load(p, &format!("{m}.resnets.1"))?),
            norm_out: GroupNorm::load(p, &format!("{e}.conv_norm_out"))?,
            conv_out: Conv::load(p, &format!("{e}.conv_out"))?,
            quant_conv: Conv::load(p, "quant_conv")?,
            dtype,
        })
    }

    /// RGB `(1, 3, H, W)` in [-1, 1] (sides multiples of 16) to the sampler's `(1, 128, H/16,
    /// W/16)` f32.
    pub fn encode(&self, img: &Array) -> Result<Array> {
        let s = img.shape();
        anyhow::ensure!(s.len() == 4 && s[1] == 3 && s[2] % 16 == 0 && s[3] % 16 == 0, "a picture of shape {s:?}: (1, 3, H, W), sides multiples of 16");
        let mut x = self.conv_in.forward(&img.transpose_axes(&[0, 2, 3, 1])?.as_dtype(self.dtype)?)?;
        for (resnets, downsample) in &self.down {
            for r in resnets {
                x = r.forward(&x)?;
                x.eval()?;
            }
            if let Some((w, b)) = downsample {
                // diffusers' Downsample2D: one zero row and column after, then stride 2.
                let sh = x.shape();
                let (n, h, wd, c) = (sh[0], sh[1], sh[2], sh[3]);
                let x1 = ops::concatenate(&[&x, &Array::zeros::<f32>(&[n, h, 1, c])?.as_dtype(x.dtype())?], 2)?;
                let x1 = ops::concatenate(&[&x1, &Array::zeros::<f32>(&[n, 1, wd + 1, c])?.as_dtype(x.dtype())?], 1)?;
                x = ops::conv2d(&x1, w, (2, 2), (0, 0), None, None)?.add(b)?;
                x.eval()?;
            }
        }
        let x = self.mid.2.forward(&self.mid.1.forward(&self.mid.0.forward(&x)?)?)?;
        let x = self.conv_out.forward(&mlx_rs::nn::silu(self.norm_out.forward(&x)?)?)?;
        let moments = self.quant_conv.forward(&x)?.as_dtype(Dtype::Float32)?;
        // The posterior's mode: its mean, the first half of the channels; packed `(c, dy, dx)`.
        let z = crate::nn::cols(&moments, 0, LATENT as usize)?;
        let sh = z.shape();
        let (b, zh, zw) = (sh[0], sh[1], sh[2]);
        let z = z.reshape(&[b, zh / 2, 2, zw / 2, 2, LATENT])?.transpose_axes(&[0, 5, 2, 4, 1, 3])?.reshape(&[b, PACKED, zh / 2, zw / 2])?;
        Ok(z.subtract(&self.bn_mean)?.divide(&self.bn_std)?)
    }
}

/// Straight RGBA8 `w × h` to RGB `(1, 3, h, w)` in [-1, 1] f32 (alpha dropped).
pub fn from_rgba_u8(rgba: &[u8], w: usize, h: usize) -> Result<Array> {
    anyhow::ensure!(rgba.len() == w * h * 4, "{} bytes for a {w}x{h} RGBA picture", rgba.len());
    let rgb: Vec<f32> = rgba.as_chunks::<4>().0.iter().flat_map(|px| px.iter().take(3).map(|&v| f32::from(v) / 127.5 - 1.0)).collect();
    Ok(Array::from_slice(&rgb, &crate::nn::shape(&[1, h, w, 3])?).transpose_axes(&[0, 3, 1, 2])?)
}

/// Nearest-neighbour ×2, channels last.
fn upsample2(x: &Array) -> Result<Array> {
    let s = x.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let x = ops::broadcast_to(x.reshape(&[n, h, 1, w, 1, c])?, &[n, h, 2, w, 2, c])?;
    Ok(x.reshape(&[n, 2 * h, 2 * w, c])?)
}

/// `(1, 3, H, W)` in [-1, 1] to straight RGBA8 (`w`, `h`, bytes).
pub fn to_rgba_u8(img: &Array) -> Result<(usize, usize, Vec<u8>)> {
    let s = img.shape();
    anyhow::ensure!(s.len() == 4 && s[1] == 3, "a picture of shape {s:?}");
    let (h, w) = (usize::try_from(s[2])?, usize::try_from(s[3])?);
    let hwc = crate::nn::to_vec(&img.squeeze_axes(&[0])?.transpose_axes(&[1, 2, 0])?)?;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for px in hwc.chunks(3) {
        rgba.extend(px.iter().map(|v| ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8));
        rgba.push(255);
    }
    Ok((w, h, rgba))
}
