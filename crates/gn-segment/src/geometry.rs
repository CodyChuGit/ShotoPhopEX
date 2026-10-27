//! Detector box prompts and the image-conditioned geometry CLS token.

use crate::prompt::{Loader, channel_norm};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{LayerNorm, Linear, Module};
use std::path::Path;

/// Geometry weights shared by box encoding and the separately ordered CLS path.
pub struct Geometry {
    direct: Linear,
    pool: Linear,
    position: Linear,
    positive: Tensor,
    cls_embed: Tensor,
    norm: LayerNorm,
    img_norm: LayerNorm,
    layers: Vec<Layer>,
    encode_norm: LayerNorm,
    final_proj: Linear,
}

impl Geometry {
    /// Load raw detector.geometry_encoder F16 weights and perform all math in f32.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).with_context(|| format!("read {}", path.as_ref().display()))?;
        let l = Loader { tensors: BufferedSafetensors::new(bytes)?, device };
        let p = "detector.geometry_encoder";
        Ok(Self {
            direct: l.linear(&format!("{p}.boxes_direct_project"), 4, 256)?,
            // A full 7x7 convolution on each 7x7 ROI is exactly this flattened linear.
            pool: Linear::new(
                l.get(&format!("{p}.boxes_pool_project.weight"), &[256, 256, 7, 7])?.reshape((256, 256 * 49))?,
                Some(l.get(&format!("{p}.boxes_pool_project.bias"), &[256])?),
            ),
            position: l.linear(&format!("{p}.boxes_pos_enc_project"), 258, 256)?,
            positive: l.get(&format!("{p}.label_embed.weight"), &[2, 256])?.narrow(0, 1, 1)?,
            cls_embed: l.get(&format!("{p}.cls_embed.weight"), &[1, 256])?.unsqueeze(0)?,
            norm: l.norm(&format!("{p}.norm"), 256)?,
            img_norm: l.norm(&format!("{p}.img_pre_norm"), 256)?,
            layers: (0..3).map(|i| Layer::load(&l, &format!("{p}.encode.{i}"))).collect::<Result<_>>()?,
            encode_norm: l.norm(&format!("{p}.encode_norm"), 256)?,
            final_proj: l.linear(&format!("{p}.final_proj"), 256, 256)?,
        })
    }

    /// Encode positive boxes `[1,N,4]` into detector prompt tokens `[1,N,256]`.
    /// Coordinates are cx,cy,w,h normalized from original-image pixel corners.
    /// Use detector FPN level 2 and image positions, both `[1,256,72,72]`.
    /// Positions are validated but unused by ComfyUI's box path; outside boxes are not clipped.
    /// Exact ROIAlign runs on CPU before returning pooled vectors to the input device.
    pub fn boxes(&self, fpn_lowest: &Tensor, image_pos: &Tensor, boxes: &Tensor) -> Result<Tensor> {
        let (image, _) = image_inputs(fpn_lowest, image_pos)?;
        let (b, n, c) = boxes.dims3()?;
        ensure!(b == 1 && c == 4 && n > 0, "boxes must be [1,N,4] with N > 0");
        let boxes = boxes.to_dtype(DType::F32)?;
        let coords = boxes.to_device(&Device::Cpu)?.reshape((n, 4))?.to_vec2::<f32>()?;
        ensure!(coords.iter().flatten().all(|x| x.is_finite()), "box coordinates must be finite");
        ensure!(coords.iter().all(|x| x[2] >= 0.0 && x[3] >= 0.0), "box sizes must be nonnegative");
        let corners: Vec<_> =
            coords.iter().map(|v| [(v[0] - 0.5 * v[2]) * 72.0, (v[1] - 0.5 * v[3]) * 72.0, (v[0] + 0.5 * v[2]) * 72.0, (v[1] + 0.5 * v[3]) * 72.0]).collect();
        let normalized = channel_norm(&fpn_lowest.to_dtype(DType::F32)?, &self.img_norm)?;
        let pooled = crate::ops::roi_align(&normalized.to_device(&Device::Cpu)?, &corners)?.to_device(fpn_lowest.device())?;
        let pooled = self.pool.forward(&pooled.reshape((n, 256 * 49))?)?.unsqueeze(0)?;
        let positions = box_positions(&boxes)?;
        let mut geo = (self.direct.forward(&boxes)? + pooled)?;
        geo = (geo + self.position.forward(&positions)?)?.broadcast_add(&self.positive.unsqueeze(0)?)?;
        geo = self.norm.forward(&geo)?;
        for layer in &self.layers {
            geo = layer.forward(&geo, &image, &image)?;
        }
        Ok(self.final_proj.forward(&self.encode_norm.forward(&geo)?)?)
    }

    /// Compute `[1,1,256]` CLS: project before norm, and add image positions only to keys.
    pub fn cls(&self, fpn_lowest: &Tensor, image_pos: &Tensor) -> Result<Tensor> {
        let (image, positions) = image_inputs(fpn_lowest, image_pos)?;
        let keys = (&image + positions)?;
        let mut cls = self.norm.forward(&self.final_proj.forward(&self.cls_embed)?)?;
        for layer in &self.layers {
            cls = layer.forward(&cls, &keys, &image)?;
        }
        Ok(self.encode_norm.forward(&cls)?)
    }
}

fn image_inputs(image: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
    ensure!(image.dims() == [1, 256, 72, 72] && positions.dims() == image.dims(), "image and positions must be [1,256,72,72]");
    let flatten = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_from(2)?.transpose(1, 2)?.contiguous();
    Ok((flatten(image)?, flatten(positions)?))
}

/// Interleave center sine/cosine channels in y,x order, then append raw h,w.
fn box_positions(boxes: &Tensor) -> Result<Tensor> {
    let device = boxes.device();
    let freqs: Vec<f32> = (0..64).map(|i| 10000_f32.powf((2 * i) as f32 / 128.0)).collect();
    let freqs = Tensor::from_vec(freqs, (1, 1, 64), device)?;
    let mut parts = Vec::new();
    for coord in [1, 0] {
        let angle = (boxes.narrow(2, coord, 1)? * std::f64::consts::TAU)?.broadcast_div(&freqs)?;
        parts.push(Tensor::stack(&[angle.sin()?, angle.cos()?], 3)?.flatten_from(2)?);
    }
    parts.extend([boxes.narrow(2, 3, 1)?, boxes.narrow(2, 2, 1)?]);
    Ok(Tensor::cat(&parts, 2)?)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
impl Attention {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self> {
        let weight = l.get(&format!("{p}.in_proj_weight"), &[768, 256])?;
        let bias = l.get(&format!("{p}.in_proj_bias"), &[768])?;
        let projection = |i: usize| -> Result<Linear> { Ok(Linear::new(weight.narrow(0, i * 256, 256)?, Some(bias.narrow(0, i * 256, 256)?))) };
        Ok(Self { q: projection(0)?, k: projection(1)?, v: projection(2)?, out: l.linear(&format!("{p}.out_proj"), 256, 256)? })
    }
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let heads = |x: &Tensor, linear: &Linear| -> Result<Tensor> { Ok(linear.forward(x)?.reshape((1, x.dim(1)?, 8, 32))?.transpose(1, 2)?.contiguous()?) };
        let out = crate::attention::attention(&heads(q, &self.q)?, &heads(k, &self.k)?, &heads(v, &self.v)?, None, false, None, 64)?;
        Ok(self.out.forward(&out.transpose(1, 2)?.contiguous()?.reshape((1, q.dim(1)?, 256))?)?)
    }
}

struct Layer {
    self_attn: Attention,
    cross_attn: Attention,
    norms: [LayerNorm; 3],
    linear1: Linear,
    linear2: Linear,
}
impl Layer {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(l, &format!("{p}.self_attn"))?,
            cross_attn: Attention::load(l, &format!("{p}.cross_attn_image"))?,
            norms: [l.norm(&format!("{p}.norm1"), 256)?, l.norm(&format!("{p}.norm2"), 256)?, l.norm(&format!("{p}.norm3"), 256)?],
            linear1: l.linear(&format!("{p}.linear1"), 256, 2048)?,
            linear2: l.linear(&format!("{p}.linear2"), 2048, 256)?,
        })
    }
    fn forward(&self, x: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor> {
        let norm = self.norms[0].forward(x)?;
        let x = (x + self.self_attn.forward(&norm, &norm, &norm)?)?;
        let x = (&x + self.cross_attn.forward(&self.norms[1].forward(&x)?, keys, values)?)?;
        Ok((&x + self.linear2.forward(&self.linear1.forward(&self.norms[2].forward(&x)?)?.relu()?)?)?)
    }
}
