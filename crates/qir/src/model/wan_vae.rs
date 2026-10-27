//! The Wan 2.1 VAE for one frame, its decoder and (when the file has it) its encoder (Krea 2's
//! pixels and an edit's picture; also Qwen-Image 1's): 16 latent
//! channels, 8× spatial, Wan's own tensor layout (`conv2`, `decoder.conv1`, `decoder.middle.N`,
//! `decoder.upsamples.N`, `decoder.head.N`, RMS norms with a `gamma`). Ported clean-room from
//! the architecture as ComfyUI runs it (`comfy/ldm/wan/vae.py`, the Alibaba Wan team's code)
//! and `stable-diffusion.cpp`'s `wan_vae.hpp` (MIT).
//!
//! A single frame simplifies the 3-D model: every causal 3-D convolution sees two zero frames
//! before its one frame, so only the last temporal tap of its kernel touches the data and the
//! convolution is a 2-D one with that tap; the temporal `time_conv` of an `upsample3d` resample
//! runs only from the second frame on. Numerics in f32 on the device (a 1024² decode is small
//! beside the DiT); the fused bf16 path of the 2.1 VAE is for later.

use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::Result;

/// ComfyUI's `latent_formats.Wan21`: the sampler works on `(z − mean) / std`, the VAE on `z`.
pub const WAN21_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const WAN21_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];
pub const LATENT_CHANNELS: usize = 16;
/// Pixels per latent cell on a side.
pub const SCALE: usize = 8;

/// A causal 3-D convolution's last temporal tap as a 2-D convolution (`weight` is
/// `(O, I, kt, kh, kw)`).
fn conv_last_tap(in_c: usize, out_c: usize, k: usize, kt: usize, vb: VarBuilder) -> Result<Conv2d> {
    let w = vb.get((out_c, in_c, kt, k, k), "weight")?;
    let w = w.i((.., .., kt - 1))?.contiguous()?; // (O, I, k, k)
    let b = vb.get(out_c, "bias")?;
    Ok(Conv2d::new(w, Some(b), Conv2dConfig { padding: k / 2, ..Default::default() }))
}

/// A plain 2-D convolution (`weight` `(O, I, k, k)`).
fn conv2d(in_c: usize, out_c: usize, k: usize, vb: VarBuilder) -> Result<Conv2d> {
    let w = vb.get((out_c, in_c, k, k), "weight")?;
    let b = vb.get(out_c, "bias")?;
    Ok(Conv2d::new(w, Some(b), Conv2dConfig { padding: k / 2, ..Default::default() }))
}

/// Wan's `RMS_norm` over the channel axis: `x / ‖x‖₂ · √C · gamma`.
struct RmsNorm {
    gamma: Tensor, // (C, 1, 1)
    scale: f64,
}

impl RmsNorm {
    fn load(c: usize, vb: VarBuilder) -> Result<Self> {
        // `images=False` norms carry (C, 1, 1, 1), the attention norm (C, 1, 1).
        let gamma = match vb.get((c, 1, 1, 1), "gamma") {
            Ok(g) => g.reshape((c, 1, 1))?,
            Err(_) => vb.get((c, 1, 1), "gamma")?,
        };
        Ok(Self { gamma: gamma.to_dtype(DType::F32)?, scale: (c as f64).sqrt() })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let norm = x32.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
        Ok((x32.broadcast_div(&norm)? * self.scale)?.broadcast_mul(&self.gamma)?.to_dtype(dt)?)
    }
}

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// `ResidualBlock`: norm, SiLU, conv3, norm, SiLU, conv3, plus the (projected) input.
struct ResidualBlock {
    norm1: RmsNorm,
    conv1: Conv2d,
    norm2: RmsNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResidualBlock {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let r = vb.pp("residual");
        Ok(Self {
            norm1: RmsNorm::load(in_c, r.pp("0"))?,
            conv1: conv_last_tap(in_c, out_c, 3, 3, r.pp("2"))?,
            norm2: RmsNorm::load(out_c, r.pp("3"))?,
            conv2: conv_last_tap(out_c, out_c, 3, 3, r.pp("6"))?,
            shortcut: if in_c == out_c { None } else { Some(conv_last_tap(in_c, out_c, 1, 1, vb.pp("shortcut"))?) },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        let sc = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok((h + sc)?)
    }
}

/// `AttentionBlock`: one head over the frame's pixels.
struct AttentionBlock {
    norm: RmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    dim: usize,
}

impl AttentionBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { norm: RmsNorm::load(dim, vb.pp("norm"))?, to_qkv: conv2d(dim, 3 * dim, 1, vb.pp("to_qkv"))?, proj: conv2d(dim, dim, 1, vb.pp("proj"))?, dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let qkv = self.to_qkv.forward(&self.norm.forward(x)?)?.reshape((b, 3 * c, h * w))?.transpose(1, 2)?; // (B, HW, 3C)
        let q = qkv.narrow(2, 0, c)?.contiguous()?.to_dtype(DType::F32)?;
        let k = qkv.narrow(2, c, c)?.contiguous()?.to_dtype(DType::F32)?;
        let v = qkv.narrow(2, 2 * c, c)?.contiguous()?;
        // Scores in f32 whatever the model's dtype (a 1024² picture has 16 384 keys).
        let scores = (q.matmul(&k.transpose(1, 2)?)? / (self.dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(v.dtype())?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((self.proj.forward(&out)? + x)?)
    }
}

/// `Resample` upsampling for one frame: nearest 2× then a 3×3 convolution to half the
/// channels (the `upsample3d` kind's temporal convolution never runs on a first frame).
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { conv: conv2d(dim, dim / 2, 3, vb.pp("resample").pp("1"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        Ok(self.conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)?)
    }
}

enum Up {
    Res(ResidualBlock),
    Sample(Upsample),
}

/// `Resample` downsampling for one frame: `ZeroPad2d((0, 1, 0, 1))` then a 3×3 convolution with
/// stride 2 (the `downsample3d` kind's `time_conv` only runs from the second frame on: on the
/// first chunk the encoder caches the frame and skips it).
struct Downsample {
    weight: Tensor,
    bias: Tensor,
}

impl Downsample {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let r = vb.pp("resample").pp("1");
        Ok(Self { weight: r.get((dim, dim, 3, 3), "weight")?, bias: r.get(dim, "bias")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?;
        let y = x.conv2d(&self.weight, 0, 2, 1, 1)?;
        Ok(y.broadcast_add(&self.bias.reshape((1, self.bias.dim(0)?, 1, 1))?)?)
    }
}

enum Down {
    Res(ResidualBlock),
    Sample(Downsample),
}

/// The encoder for one frame: `encoder.conv1`, the downsamples, the middle, the head to
/// `2·z` channels, then the VAE's own 1×1 `conv1` (the quantisation convolution); the mean is
/// the first `z` channels.
struct WanEncoder {
    conv1: Conv2d,
    downs: Vec<Down>,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    head_norm: RmsNorm,
    head_conv: Conv2d,
    quant: Conv2d,
}

impl WanEncoder {
    /// `dim` 96, `dim_mult [1, 2, 4, 4]`, two residual blocks per stage, a downsample after
    /// every stage but the last.
    fn load(vb: VarBuilder) -> Result<Self> {
        let dim = 96usize;
        let dim_mult = [1usize, 2, 4, 4];
        let z = LATENT_CHANNELS;
        let mut mult = vec![1usize];
        mult.extend(dim_mult.iter());
        let dims: Vec<usize> = mult.iter().map(|m| dim * m).collect(); // [96, 96, 192, 384, 384]
        let e = vb.pp("encoder");
        let d = e.pp("downsamples");
        let mut downs = Vec::new();
        let mut idx = 0usize;
        for i in 0..dims.len() - 1 {
            let (mut in_c, out_c) = (dims[i], dims[i + 1]);
            for _ in 0..2 {
                downs.push(Down::Res(ResidualBlock::load(in_c, out_c, d.pp(idx))?));
                in_c = out_c;
                idx += 1;
            }
            if i != dim_mult.len() - 1 {
                downs.push(Down::Sample(Downsample::load(out_c, d.pp(idx))?));
                idx += 1;
            }
        }
        let last = dims[dims.len() - 1];
        let m = e.pp("middle");
        Ok(Self {
            conv1: conv_last_tap(3, dims[0], 3, 3, e.pp("conv1"))?,
            downs,
            middle: (ResidualBlock::load(last, last, m.pp("0"))?, AttentionBlock::load(last, m.pp("1"))?, ResidualBlock::load(last, last, m.pp("2"))?),
            head_norm: RmsNorm::load(last, e.pp("head").pp("0"))?,
            head_conv: conv_last_tap(last, 2 * z, 3, 3, e.pp("head").pp("2"))?,
            quant: conv_last_tap(2 * z, 2 * z, 1, 1, vb.pp("conv1"))?,
        })
    }

    /// `(B, 3, H, W)` in `[-1, 1]` to the posterior mean `(B, 16, H/8, W/8)` in the VAE's space.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(x)?;
        for d in &self.downs {
            x = match d {
                Down::Res(r) => r.forward(&x)?,
                Down::Sample(s) => s.forward(&x)?,
            };
        }
        x = self.middle.0.forward(&x)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward(&x)?;
        let x = self.head_conv.forward(&silu(&self.head_norm.forward(&x)?)?)?;
        let moments = self.quant.forward(&x)?;
        Ok(moments.narrow(1, 0, LATENT_CHANNELS)?)
    }
}

/// The decoder: `conv2` (the VAE's post-quantisation 1×1), `conv1`, the middle, the upsamples,
/// the head.
pub struct WanVae {
    conv2: Conv2d,
    conv1: Conv2d,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    ups: Vec<Up>,
    head_norm: RmsNorm,
    head_conv: Conv2d,
    /// Present when the file has `encoder.*` (ComfyUI's `wan_2.1_vae` does): an edit's picture
    /// goes in through it.
    encoder: Option<WanEncoder>,
    device: Device,
    dtype: DType,
}

impl WanVae {
    /// Load from the file's root (`conv2.*`, `decoder.*`): `dim` 96, `dim_mult [1, 2, 4, 4]`,
    /// two residual blocks per stage (three in the decoder), temporal upsampling on the first
    /// two stages.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let dim = 96usize;
        let dim_mult = [1usize, 2, 4, 4];
        let z = LATENT_CHANNELS;
        let mut mult = vec![dim_mult[3]];
        mult.extend(dim_mult.iter().rev());
        let dims: Vec<usize> = mult.iter().map(|m| dim * m).collect(); // [384, 384, 384, 192, 96]
        let d = vb.pp("decoder");
        let conv2 = conv_last_tap(z, z, 1, 1, vb.pp("conv2"))?;
        let conv1 = conv_last_tap(z, dims[0], 3, 3, d.pp("conv1"))?;
        let m = d.pp("middle");
        let middle = (ResidualBlock::load(dims[0], dims[0], m.pp("0"))?, AttentionBlock::load(dims[0], m.pp("1"))?, ResidualBlock::load(dims[0], dims[0], m.pp("2"))?);
        let u = d.pp("upsamples");
        let mut ups = Vec::new();
        let mut idx = 0usize;
        for i in 0..dims.len() - 1 {
            let (mut in_c, out_c) = (dims[i], dims[i + 1]);
            if i >= 1 {
                in_c /= 2; // the previous stage's resample halved the channels
            }
            for _ in 0..3 {
                ups.push(Up::Res(ResidualBlock::load(in_c, out_c, u.pp(idx))?));
                in_c = out_c;
                idx += 1;
            }
            if i != dim_mult.len() - 1 {
                ups.push(Up::Sample(Upsample::load(out_c, u.pp(idx))?));
                idx += 1;
            }
        }
        let last = dims[dims.len() - 1];
        let h = d.pp("head");
        Ok(Self {
            conv2,
            conv1,
            middle,
            ups,
            head_norm: RmsNorm::load(last, h.pp("0"))?,
            head_conv: conv_last_tap(last, 3, 3, 3, h.pp("2"))?,
            encoder: if vb.contains_tensor("encoder.conv1.weight") { Some(WanEncoder::load(vb.clone())?) } else { None },
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Whether the file came with the encoder.
    pub fn can_encode(&self) -> bool {
        self.encoder.is_some()
    }

    /// Encode a picture `(B, 3, H, W)` in `[-1, 1]` (`H`, `W` multiples of 8) into the sampler's
    /// normalised space `(B, 16, H/8, W/8)`: the posterior mean, then `(z − mean) / std` (what
    /// ComfyUI's `VAEEncode` hands a Wan-latent model after `process_latent_in`). f32 out.
    pub fn encode(&self, rgb: &Tensor) -> Result<Tensor> {
        let Some(enc) = &self.encoder else { anyhow::bail!("this VAE file has no encoder") };
        let (_, c, h, w) = rgb.dims4()?;
        anyhow::ensure!(c == 3 && h.is_multiple_of(SCALE) && w.is_multiple_of(SCALE), "encode wants (B, 3, 8k, 8k), got (_, {c}, {h}, {w})");
        let z = enc.forward(&rgb.to_device(&self.device)?.to_dtype(self.dtype)?)?.to_dtype(DType::F32)?;
        self.normalize(&z)
    }

    /// The VAE's space to the sampler's: `(z − mean) / std`.
    pub fn normalize(&self, z: &Tensor) -> Result<Tensor> {
        let mean = Tensor::from_slice(&WAN21_MEAN, (1, LATENT_CHANNELS, 1, 1), z.device())?.to_dtype(z.dtype())?;
        let std = Tensor::from_slice(&WAN21_STD, (1, LATENT_CHANNELS, 1, 1), z.device())?.to_dtype(z.dtype())?;
        Ok(z.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    /// Decode one frame's raw latent `(B, 16, h, w)` (the VAE's own space) to `(B, 3, 8h, 8w)`
    /// in `[-1, 1]`, in the dtype the weights were loaded with (norms and attention scores in
    /// f32 regardless).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.to_device(&self.device)?.to_dtype(self.dtype)?;
        let mut x = self.conv1.forward(&self.conv2.forward(&z)?)?;
        x = self.middle.0.forward(&x)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward(&x)?;
        for u in &self.ups {
            x = match u {
                Up::Res(r) => r.forward(&x)?,
                Up::Sample(s) => s.forward(&x)?,
            };
        }
        let x = silu(&self.head_norm.forward(&x)?)?;
        Ok(self.head_conv.forward(&x)?.to_dtype(DType::F32)?)
    }

    /// A sampler-space latent `(B, 16, h, w)` back to the VAE's space: `z · std + mean`.
    pub fn denormalize(&self, z: &Tensor) -> Result<Tensor> {
        let mean = Tensor::from_slice(&WAN21_MEAN, (1, LATENT_CHANNELS, 1, 1), z.device())?.to_dtype(z.dtype())?;
        let std = Tensor::from_slice(&WAN21_STD, (1, LATENT_CHANNELS, 1, 1), z.device())?.to_dtype(z.dtype())?;
        Ok(z.broadcast_mul(&std)?.broadcast_add(&mean)?)
    }
}

/// `(B, 3, H, W)` in `[-1, 1]` to straight RGBA8 (alpha opaque), row-major, with the size.
pub fn to_rgba_u8(img: &Tensor) -> Result<(usize, usize, Vec<u8>)> {
    let (_, _, h, w) = img.dims4()?;
    let hwc = ((img.i(0)?.permute((1, 2, 0))?.clamp(-1.0, 1.0)? + 1.0)? * 127.5)?.round()?.to_dtype(DType::U8)?.contiguous()?;
    let rgb = hwc.flatten_all()?.to_vec1::<u8>()?;
    let mut out = Vec::with_capacity(h * w * 4);
    for p in rgb.as_chunks::<3>().0 {
        out.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    Ok((w, h, out))
}
