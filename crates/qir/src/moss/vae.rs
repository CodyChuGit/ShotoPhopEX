//! The FLUX.2 VAE's decoder (diffusers' `AutoencoderKLFlux2` layout, the full or the small
//! decoder: the widths are read from the file) and the latent's unpacking: MOSS-Image samples
//! in the VAE's normalised, 2×2-packed space of 128 channels at 1/16 scale; decoding undoes the
//! batch-norm statistics, unpacks to 32 channels at 1/8 and runs `post_quant_conv` and the
//! decoder (an SD-style decoder: a mid block with one attention, four up blocks of three
//! resnets). Runs in f32, as the official pipeline does.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig, UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

use crate::Result;

/// The decoder's latent channels (before the 2×2 packing).
pub const LATENT_CHANNELS: usize = 32;
/// Pixels per packed latent pixel.
pub const SCALE: usize = 16;
/// `batch_norm_eps` of the VAE's config.
const BN_EPS: f64 = 1e-4;

pub struct Flux2Decoder {
    /// `sqrt(running_var + eps)` and `running_mean`, `(1, 128, 1, 1)`.
    bn_std: Tensor,
    bn_mean: Tensor,
    post_quant_conv: candle_nn::Conv2d,
    conv_in: candle_nn::Conv2d,
    mid: UNetMidBlock2D,
    up: Vec<UpDecoderBlock2D>,
    norm_out: candle_nn::GroupNorm,
    conv_out: candle_nn::Conv2d,
    /// The decoder's dtype (the builder's).
    dtype: DType,
}

impl Flux2Decoder {
    /// From the VAE file (`bn.*`, `post_quant_conv.*`, `decoder.*`); the decoder runs in the
    /// builder's dtype (the statistics in f32).
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let packed = 4 * LATENT_CHANNELS;
        let var = vb.get(packed, "bn.running_var")?.to_dtype(DType::F32)?;
        let bn_std = (var + BN_EPS)?.sqrt()?.reshape((1, packed, 1, 1))?;
        let bn_mean = vb.get(packed, "bn.running_mean")?.to_dtype(DType::F32)?.reshape((1, packed, 1, 1))?;
        let post_quant_conv = candle_nn::conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1, Default::default(), vb.pp("post_quant_conv"))?;
        let d = vb.pp("decoder");
        // Widths from the file: the deepest is conv_in's, the shallowest conv_out's input.
        let deepest = d.pp("conv_in").get_unchecked_dtype("weight", DType::F32)?.dim(0)?;
        let shallowest = d.pp("conv_out").get_unchecked_dtype("weight", DType::F32)?.dim(1)?;
        let up_vb = d.pp("up_blocks");
        let mut widths = Vec::new();
        for i in 0.. {
            let b = up_vb.pp(i.to_string()).pp("resnets").pp("0");
            if !b.contains_tensor("conv2.weight") {
                break;
            }
            widths.push(b.pp("conv2").get_unchecked_dtype("weight", DType::F32)?.dim(0)?);
        }
        anyhow::ensure!(!widths.is_empty() && widths[0] == deepest && widths.last() == Some(&shallowest), "unexpected FLUX.2 decoder widths {widths:?}");
        let pad1 = candle_nn::Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = candle_nn::conv2d(LATENT_CHANNELS, deepest, 3, pad1, d.pp("conv_in"))?;
        let mid_cfg =
            UNetMidBlock2DConfig { resnet_eps: 1e-6, output_scale_factor: 1.0, attn_num_head_channels: None, resnet_groups: Some(32), ..Default::default() };
        let mid = UNetMidBlock2D::new(d.pp("mid_block"), deepest, None, mid_cfg)?;
        let mut up = Vec::with_capacity(widths.len());
        for (i, &out) in widths.iter().enumerate() {
            let input = if i == 0 { deepest } else { widths[i - 1] };
            let cfg = UpDecoderBlock2DConfig { num_layers: 3, resnet_eps: 1e-6, resnet_groups: 32, add_upsample: i + 1 < widths.len(), ..Default::default() };
            up.push(UpDecoderBlock2D::new(up_vb.pp(i.to_string()), input, out, cfg)?);
        }
        let norm_out = candle_nn::group_norm(32, shallowest, 1e-6, d.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(shallowest, 3, 3, pad1, d.pp("conv_out"))?;
        Ok(Self { bn_std, bn_mean, post_quant_conv, conv_in, mid, up, norm_out, conv_out, dtype })
    }

    /// The sampler's latent `(1, 128, h, w)` back to the decoder's `(1, 32, 2h, 2w)`: the
    /// batch-norm statistics undone, then each pixel's channels `4c + 2i + j` placed at
    /// `(2y + i, 2x + j)` of channel `c`.
    pub fn unpack(&self, z: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = z.dims4()?;
        let z = z.to_dtype(DType::F32)?.broadcast_mul(&self.bn_std)?.broadcast_add(&self.bn_mean)?;
        let z = z.reshape((b, c / 4, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.contiguous()?;
        Ok(z.reshape((b, c / 4, 2 * h, 2 * w))?.to_dtype(self.dtype)?)
    }

    /// RGB `(1, 3, 16h, 16w)` in [-1, 1] for the sampler's latent.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(&self.unpack(z)?)?;
        let mut x = self.mid.forward(&self.conv_in.forward(&x)?, None)?;
        for block in &self.up {
            x = block.forward(&x)?;
        }
        let x = candle_nn::ops::silu(&self.norm_out.forward(&x)?)?;
        Ok(self.conv_out.forward(&x)?.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?)
    }
}

/// The FLUX.2 VAE's encoder (diffusers' layout: `encoder.*` and the top-level `quant_conv`, as
/// Comfy-Org's `flux2-vae_small` has it, the widths read from the file) and the latent's packing:
/// a picture in [-1, 1] to the posterior's mode, 2×2-packed (`c, dy, dx`) to 128 channels at 1/16
/// and batch-normalised, the space MOSS-Image samples in (MOSS Edit's references).
pub struct Flux2Encoder {
    bn_std: Tensor,
    bn_mean: Tensor,
    conv_in: candle_nn::Conv2d,
    down: Vec<DownEncoderBlock2D>,
    mid: UNetMidBlock2D,
    norm_out: candle_nn::GroupNorm,
    conv_out: candle_nn::Conv2d,
    quant_conv: candle_nn::Conv2d,
    dtype: DType,
}

impl Flux2Encoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let packed = 4 * LATENT_CHANNELS;
        let var = vb.get(packed, "bn.running_var")?.to_dtype(DType::F32)?;
        let bn_std = (var + BN_EPS)?.sqrt()?.reshape((1, packed, 1, 1))?;
        let bn_mean = vb.get(packed, "bn.running_mean")?.to_dtype(DType::F32)?.reshape((1, packed, 1, 1))?;
        anyhow::ensure!(vb.contains_tensor("quant_conv.weight"), "no top-level quant_conv: not diffusers' FLUX.2 VAE layout (flux2-vae_small has it)");
        let e = vb.pp("encoder");
        let down_vb = e.pp("down_blocks");
        let mut widths = Vec::new();
        for i in 0.. {
            let b = down_vb.pp(i.to_string()).pp("resnets").pp("0");
            if !b.contains_tensor("conv2.weight") {
                break;
            }
            widths.push(b.pp("conv2").get_unchecked_dtype("weight", DType::F32)?.dim(0)?);
        }
        anyhow::ensure!(!widths.is_empty(), "the VAE file has no encoder (encoder.down_blocks.*)");
        let layers = (0..).take_while(|j| down_vb.pp("0").pp("resnets").pp(j.to_string()).contains_tensor("conv1.weight")).count();
        let pad1 = candle_nn::Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = candle_nn::conv2d(3, widths[0], 3, pad1, e.pp("conv_in"))?;
        let mut down = Vec::with_capacity(widths.len());
        for (i, &out) in widths.iter().enumerate() {
            let input = if i == 0 { widths[0] } else { widths[i - 1] };
            let cfg = DownEncoderBlock2DConfig {
                num_layers: layers,
                resnet_eps: 1e-6,
                resnet_groups: 32,
                add_downsample: i + 1 < widths.len(),
                downsample_padding: 0,
                ..Default::default()
            };
            down.push(DownEncoderBlock2D::new(down_vb.pp(i.to_string()), input, out, cfg)?);
        }
        let deepest = *widths.last().unwrap_or(&widths[0]);
        let mid_cfg =
            UNetMidBlock2DConfig { resnet_eps: 1e-6, output_scale_factor: 1.0, attn_num_head_channels: None, resnet_groups: Some(32), ..Default::default() };
        let mid = UNetMidBlock2D::new(e.pp("mid_block"), deepest, None, mid_cfg)?;
        let norm_out = candle_nn::group_norm(32, deepest, 1e-6, e.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(deepest, 2 * LATENT_CHANNELS, 3, pad1, e.pp("conv_out"))?;
        let quant_conv = candle_nn::conv2d(2 * LATENT_CHANNELS, 2 * LATENT_CHANNELS, 1, Default::default(), vb.pp("quant_conv"))?;
        Ok(Self { bn_std, bn_mean, conv_in, down, mid, norm_out, conv_out, quant_conv, dtype })
    }

    /// RGB `(1, 3, H, W)` in [-1, 1] (sides multiples of 16) to the sampler's `(1, 128, H/16,
    /// W/16)` f32.
    pub fn encode(&self, img: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = img.dims4()?;
        anyhow::ensure!(h.is_multiple_of(SCALE) && w.is_multiple_of(SCALE), "a picture of {w}x{h}: sides must be multiples of {SCALE}");
        let mut x = self.conv_in.forward(&img.to_dtype(self.dtype)?)?;
        for block in &self.down {
            x = block.forward(&x)?;
        }
        let x = self.mid.forward(&x, None)?;
        let x = candle_nn::ops::silu(&self.norm_out.forward(&x)?)?;
        let moments = self.quant_conv.forward(&self.conv_out.forward(&x)?)?;
        // The posterior's mode: its mean, the first half of the channels.
        let z = moments.narrow(1, 0, LATENT_CHANNELS)?.to_dtype(DType::F32)?;
        let (b, c, zh, zw) = z.dims4()?;
        let z = z.reshape((b, c, zh / 2, 2, zw / 2, 2))?.permute((0, 1, 3, 5, 2, 4))?.contiguous()?.reshape((b, 4 * c, zh / 2, zw / 2))?;
        Ok(z.broadcast_sub(&self.bn_mean)?.broadcast_div(&self.bn_std)?)
    }
}

/// Straight RGBA8 `w × h` to RGB `(1, 3, h, w)` in [-1, 1] f32 on `dev` (alpha dropped).
pub fn from_rgba_u8(rgba: &[u8], w: usize, h: usize, dev: &Device) -> Result<Tensor> {
    anyhow::ensure!(rgba.len() == w * h * 4, "{} bytes for a {w}x{h} RGBA picture", rgba.len());
    let mut planes = vec![0f32; 3 * w * h];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        for c in 0..3 {
            planes[c * w * h + i] = f32::from(px[c]) / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(planes, (1, 3, h, w), dev)?)
}

/// `(1, 3, H, W)` in [-1, 1] to straight RGBA8.
pub fn to_rgba_u8(img: &Tensor) -> Result<(usize, usize, Vec<u8>)> {
    let (_, _, h, w) = img.dims4()?;
    let rgb = ((img.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)? + 1.0)? * 127.5)?.round()?.clamp(0f32, 255f32)?;
    let rgb = rgb.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for px in rgb.chunks(3) {
        rgba.extend(px.iter().map(|v| *v as u8));
        rgba.push(255);
    }
    Ok((w, h, rgba))
}
