//! SAM 3.1 ViTDet trunk and the detector and interactive feature pyramids.

use crate::prompt::Loader;
use crate::{
    attention::attention,
    position::{Rope2d, image_position},
};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use candle_nn::{LayerNorm, Module};
use std::path::Path;

/// The three independent pyramid levels, ordered from 288² to 72², and detector PE.
pub struct VisionOut {
    pub detector_fpn: [Tensor; 3],
    pub interactive_fpn: [Tensor; 3],
    pub image_pos: Tensor,
}

/// Fixed SAM vision encoder with compute parameters prepared once at load.
pub struct Vision {
    patch: Conv,
    positions: Tensor,
    norm: LayerNorm,
    blocks: Vec<Block>,
    detector: [Fpn; 3],
    interactive: [Fpn; 3],
    no_mem: Tensor,
    image_pos: Tensor,
    global_rope: Rope2d,
    window_rope: Rope2d,
}

impl Vision {
    /// Load selected raw checkpoint keys without unsafe memory mapping or loader renames.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_checkpoint(&crate::weights::Checkpoint::open(path)?, device)
    }

    /// Load this part from the shared checkpoint buffer.
    pub fn from_checkpoint(checkpoint: &crate::weights::Checkpoint, device: &Device) -> Result<Self> {
        let loader = Loader { checkpoint, device };
        let p = "detector.backbone.vision_backbone";
        let t = format!("{p}.trunk");
        let pos = loader.get(&format!("{t}.pos_embed"), &[1, 577, 1024])?;
        let pos = pos.narrow(1, 1, 576)?.reshape((1, 24, 24, 1024))?;
        let rows = Tensor::cat(&[&pos, &pos, &pos], 1)?;
        let positions = Tensor::cat(&[&rows, &rows, &rows], 2)?;
        let fpn = |name: &str| -> Result<[Fpn; 3]> {
            Ok([
                Fpn::load(&loader, &format!("{p}.{name}.0"), 0)?,
                Fpn::load(&loader, &format!("{p}.{name}.1"), 1)?,
                Fpn::load(&loader, &format!("{p}.{name}.2"), 2)?,
            ])
        };
        Ok(Self {
            patch: loader.vision_conv(&format!("{t}.patch_embed.proj"), 3, 1024, 14, 14, 0, false)?,
            positions,
            norm: loader.norm(&format!("{t}.ln_pre"), 1024)?,
            blocks: (0..32).map(|i| Block::load(&loader, &format!("{t}.blocks.{i}"), i % 8 == 7)).collect::<Result<_>>()?,
            detector: fpn("convs")?,
            interactive: fpn("interactive_convs")?,
            image_pos: image_position(72, 72, device)?.to_dtype(pos.dtype())?,
            no_mem: loader.raw("tracker.model.interactivity_no_mem_embed", &[1, 1, 256])?.to_dtype(pos.dtype())?.reshape((1, 256, 1, 1))?,
            global_rope: Rope2d::new(72, 72, 64, 24.0 / 72.0, device)?,
            window_rope: Rope2d::new(24, 24, 64, 1.0, device)?,
        })
    }

    /// Encode a single RGB image `[1,3,1008,1008]` already scaled into `[0,1]`.
    pub fn forward(&self, pixels: &Tensor) -> Result<VisionOut> {
        self.forward_with_observer(pixels, |_, _| Ok(()))
    }

    /// Encode refinement crops without computing the unused detector pyramid.
    pub fn forward_tracker(&self, pixels: &Tensor) -> Result<[Tensor; 3]> {
        let x = crate::profile::measure(pixels.device(), "crop trunk", || self.trunk(pixels, &mut |_, _| Ok(())))?;
        crate::profile::measure(pixels.device(), "crop FPN", || {
            Ok([self.interactive[0].forward(&x)?, self.interactive[1].forward(&x)?, self.interactive[2].forward(&x)?.broadcast_add(&self.no_mem)?])
        })
    }
    fn trunk(&self, pixels: &Tensor, observe: &mut impl FnMut(&str, &Tensor) -> Result<()>) -> Result<Tensor> {
        ensure!(pixels.dims() == [1, 3, 1008, 1008], "vision requires [1,3,1008,1008]");
        let patch = self.patch.forward(&pixels.to_dtype(self.patch.weight.dtype())?)?;
        observe("patch", &patch)?;
        let mut x = self.norm.forward(&patch.permute((0, 2, 3, 1))?.broadcast_add(&self.positions)?.contiguous()?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, if block.global { &self.global_rope } else { &self.window_rope })?;
            if matches!(i, 0 | 7 | 31) {
                observe(&format!("block{i}"), &x)?;
            }
        }
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        Ok(x)
    }

    /// Observe parity boundaries without retaining full intermediate activations.
    /// Trunk blocks use NHWC; patch and pyramid observations use NCHW.
    pub fn forward_with_observer(&self, pixels: &Tensor, mut observe: impl FnMut(&str, &Tensor) -> Result<()>) -> Result<VisionOut> {
        let x = crate::profile::measure(pixels.device(), "trunk", || self.trunk(pixels, &mut observe))?;
        let mut pyramid = |branches: &[Fpn; 3], name: &str, interactive: bool| -> Result<[Tensor; 3]> {
            let mut outputs = Vec::with_capacity(3);
            for (i, branch) in branches.iter().enumerate() {
                let mut feature = crate::profile::measure(x.device(), &format!("{name} FPN"), || branch.forward(&x))?;
                if interactive && i == 2 {
                    feature = feature.broadcast_add(&self.no_mem)?;
                }
                observe(&format!("{name}{i}"), &feature)?;
                outputs.push(feature);
            }
            outputs.try_into().map_err(|_| anyhow::anyhow!("expected three pyramid levels"))
        };
        let detector_fpn = pyramid(&self.detector, "detector", false)?;
        let interactive_fpn = pyramid(&self.interactive, "interactive", true)?;
        Ok(VisionOut { detector_fpn, interactive_fpn, image_pos: self.image_pos.clone() })
    }
}

/// Partition NHWC into padded square windows, preserving row-major window order.
/// Padding is needed for primitive reuse even though SAM's 72 grid divides by 24.
pub fn window_partition(x: &Tensor, size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    ensure!(size > 0 && b > 0 && h > 0 && w > 0 && c > 0, "window dimensions must be positive");
    let (ph, pw) = ((size - h % size) % size, (size - w % size) % size);
    let x = x.pad_with_zeros(1, 0, ph)?.pad_with_zeros(2, 0, pw)?;
    let (hp, wp) = (h + ph, w + pw);
    Ok((
        x.reshape((b, hp / size, size, wp / size, size, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((
            b * (hp / size) * (wp / size),
            size,
            size,
            c,
        ))?,
        (hp, wp),
    ))
}

/// Restore NHWC windows and remove the padding introduced during partitioning.
pub fn window_unpartition(x: &Tensor, size: usize, padded: (usize, usize), hw: (usize, usize)) -> Result<Tensor> {
    let (n, a, b, c) = x.dims4()?;
    let (hp, wp) = padded;
    let (h, w) = hw;
    ensure!(
        size > 0 && hp > 0 && wp > 0 && hp.is_multiple_of(size) && wp.is_multiple_of(size) && a == size && b == size && h > 0 && w > 0 && h <= hp && w <= wp,
        "invalid window geometry"
    );
    let count = (hp / size) * (wp / size);
    ensure!(n > 0 && n.is_multiple_of(count), "window count disagrees with padded dimensions");
    Ok(x.reshape((n / count, hp / size, wp / size, size, size, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((n / count, hp, wp, c))?
        .narrow(1, 0, h)?
        .narrow(2, 0, w)?
        .contiguous()?)
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    global: bool,
}
impl Block {
    fn load(l: &Loader<'_>, p: &str, global: bool) -> Result<Self> {
        Ok(Self {
            norm1: l.norm(&format!("{p}.norm1"), 1024)?,
            norm2: l.norm(&format!("{p}.norm2"), 1024)?,
            qkv: l.compact_linear(&format!("{p}.attn.qkv"), 1024, 3072)?,
            proj: l.compact_linear(&format!("{p}.attn.proj"), 1024, 1024)?,
            fc1: l.compact_linear(&format!("{p}.mlp.fc1"), 1024, 4736)?,
            fc2: l.compact_linear(&format!("{p}.mlp.fc2"), 4736, 1024)?,
            global,
        })
    }
    fn forward(&self, x: &Tensor, rope: &Rope2d) -> Result<Tensor> {
        let norm = self.norm1.forward(x)?;
        let (input, pad) = if self.global { (norm, (72, 72)) } else { window_partition(&norm, 24)? };
        let (b, h, w, _) = input.dims4()?;
        let qkv = self.qkv.forward(&input.reshape((b, h * w, 1024))?)?.reshape((b, h * w, 3, 16, 64))?;
        let head = |i| -> Result<Tensor> { Ok(qkv.narrow(2, i, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?) };
        let (q, k, v) = (rope.apply(&head(0)?)?, rope.apply(&head(1)?)?, head(2)?);
        let a = attention(&q, &k, &v, None, false, None, 256)?;
        let a = a.transpose(1, 2)?.contiguous()?.reshape((b, h * w, 1024))?;
        let a = self.proj.forward(&a)?.reshape((b, h, w, 1024))?;
        let a = if self.global { a } else { window_unpartition(&a, 24, pad, (72, 72))? };
        let x = (x + a)?;
        Ok((&x + self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?)?)
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
    fused: Option<crate::linear::Linear>,
}
impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(fused) = &self.fused {
            return Ok(fused.forward(x)?);
        }
        let dims = x.dims().to_vec();
        let input = self.weight.dim(1)?;
        let output = self.weight.dim(0)?;
        let mut shape = dims;
        let last = shape.last_mut().context("linear requires a channel dimension")?;
        ensure!(*last == input, "linear input width mismatch");
        *last = output;
        Ok(x.reshape((x.elem_count() / input, input))?.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)?.reshape(shape)?)
    }
}
struct Conv {
    pointwise: Option<crate::linear::Linear>,
    fused_weight: Option<Tensor>,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}
impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(pointwise) = &self.pointwise {
            return Ok(pointwise.forward_channels(x)?);
        }
        if let Some(weight) = &self.fused_weight {
            return Ok(crate::ops::bias_channels(x)?.conv2d(weight, self.padding, self.stride, 1, 1)?);
        }
        let y = x.conv2d(&self.weight, self.padding, self.stride, 1, 1)?;
        Ok(if let Some(bias) = &self.bias { y.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)? } else { y })
    }
}

/// Nonoverlapping kernel=stride=2 transpose convolution via a channel projection.
/// This keeps CPU and accelerator execution on the same device-generic GEMM path.
pub fn transpose_conv2(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let (ic, oc, kh, kw) = weight.dims4()?;
    ensure!(c == ic && kh == 2 && kw == 2 && bias.dims() == [oc], "transpose convolution shape mismatch");
    let input = x.permute((0, 2, 3, 1))?.contiguous()?.reshape((b * h * w, c))?;
    let weight = weight.reshape((c, oc * 4))?;
    let y = input.matmul(&weight)?.reshape((b, h, w, oc, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((b, oc, h * 2, w * 2))?;
    Ok(y.broadcast_add(&bias.reshape((1, oc, 1, 1))?)?)
}
struct Up {
    weight: Tensor,
    bias: Tensor,
}
impl Up {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        transpose_conv2(x, &self.weight, &self.bias)
    }
}
struct Fpn {
    up: Vec<Up>,
    conv1: Conv,
    conv3: Conv,
}
impl Fpn {
    fn load(l: &Loader<'_>, p: &str, level: usize) -> Result<Self> {
        let mut up = Vec::new();
        let mut add = |name: &str, ic, oc| -> Result<()> {
            up.push(Up { weight: l.get(&format!("{p}.{name}.weight"), &[ic, oc, 2, 2])?, bias: l.get(&format!("{p}.{name}.bias"), &[oc])? });
            Ok(())
        };
        if level == 0 {
            add("dconv_2x2_0", 1024, 512)?;
            add("dconv_2x2_1", 512, 256)?;
        } else if level == 1 {
            add("dconv_2x2", 1024, 512)?;
        }
        Ok(Self {
            up,
            conv1: l.vision_conv(&format!("{p}.conv_1x1"), [256, 512, 1024][level], 256, 1, 1, 0, true)?,
            conv3: l.vision_conv(&format!("{p}.conv_3x3"), 256, 256, 3, 1, 1, true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, up) in self.up.iter().enumerate() {
            x = up.forward(&x)?;
            if self.up.len() == 2 && i == 0 {
                x = x.gelu_erf()?;
            }
        }
        self.conv3.forward(&self.conv1.forward(&x)?)
    }
}
impl Loader<'_> {
    fn compact_linear(&self, p: &str, ic: usize, oc: usize) -> Result<Linear> {
        let weight = self.get(&format!("{p}.weight"), &[oc, ic])?;
        let bias = self.get(&format!("{p}.bias"), &[oc])?;
        let fused = if self.device.is_cuda() && weight.dtype() == candle_core::DType::F16 {
            Some(crate::linear::Linear::new(weight.clone(), Some(bias.clone()))?)
        } else {
            None
        };
        Ok(Linear { weight, bias, fused })
    }
    #[allow(clippy::too_many_arguments)]
    fn vision_conv(&self, p: &str, ic: usize, oc: usize, k: usize, stride: usize, padding: usize, bias: bool) -> Result<Conv> {
        let weight = self.get(&format!("{p}.weight"), &[oc, ic, k, k])?;
        let bias = if bias { Some(self.get(&format!("{p}.bias"), &[oc])?) } else { None };
        let pointwise = if self.device.is_cuda() && k == 1 { Some(crate::linear::Linear::new(weight.reshape((oc, ic))?, bias.clone())?) } else { None };
        let fused_weight = if self.device.is_cuda() && k > 1 { bias.as_ref().map(|bias| crate::ops::fuse_conv_bias(&weight, bias)).transpose()? } else { None };
        Ok(Conv { weight, bias, pointwise, fused_weight, stride, padding })
    }
}
