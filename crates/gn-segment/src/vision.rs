//! SAM 3.1 ViTDet trunk and the detector and interactive feature pyramids.

use crate::{
    attention::attention,
    position::{Rope2d, image_position},
};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{LayerNorm, Module};
use std::path::Path;

/// The three independent pyramid levels, ordered from 288² to 72², and detector PE.
pub struct VisionOut {
    pub detector_fpn: [Tensor; 3],
    pub interactive_fpn: [Tensor; 3],
    pub image_pos: Tensor,
}

/// Fixed 1008² SAM vision encoder. Large parameters remain F16 between operations;
/// each operation casts its own weights to f32 so CPU execution needs no half kernels.
pub struct Vision {
    patch: Conv,
    positions: Tensor,
    norm: LayerNorm,
    blocks: Vec<Block>,
    detector: [Fpn; 3],
    interactive: [Fpn; 3],
    no_mem: Tensor,
    global_rope: Rope2d,
    window_rope: Rope2d,
}

impl Vision {
    /// Load selected raw checkpoint keys without unsafe memory mapping or loader renames.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let loader = Loader { tensors: BufferedSafetensors::new(std::fs::read(path).with_context(|| format!("read {}", path.display()))?)?, device };
        let p = "detector.backbone.vision_backbone";
        let t = format!("{p}.trunk");
        let pos = loader.get(&format!("{t}.pos_embed"), &[1, 577, 1024])?.to_dtype(DType::F32)?;
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
            patch: loader.conv(&format!("{t}.patch_embed.proj"), 3, 1024, 14, 14, 0, false)?,
            positions,
            norm: loader.norm(&format!("{t}.ln_pre"))?,
            blocks: (0..32).map(|i| Block::load(&loader, &format!("{t}.blocks.{i}"), i % 8 == 7)).collect::<Result<_>>()?,
            detector: fpn("convs")?,
            interactive: fpn("interactive_convs")?,
            no_mem: loader.get("tracker.model.interactivity_no_mem_embed", &[1, 1, 256])?.to_dtype(DType::F32)?.reshape((1, 256, 1, 1))?,
            global_rope: Rope2d::new(72, 72, 64, 24.0 / 72.0, device)?,
            window_rope: Rope2d::new(24, 24, 64, 1.0, device)?,
        })
    }

    /// Encode a single RGB image `[1,3,1008,1008]` already scaled into `[0,1]`.
    pub fn forward(&self, pixels: &Tensor) -> Result<VisionOut> {
        self.forward_with_observer(pixels, |_, _| Ok(()))
    }

    /// Observe parity boundaries without retaining full intermediate activations.
    /// Trunk blocks use NHWC; patch and pyramid observations use NCHW.
    pub fn forward_with_observer(&self, pixels: &Tensor, mut observe: impl FnMut(&str, &Tensor) -> Result<()>) -> Result<VisionOut> {
        ensure!(pixels.dims() == [1, 3, 1008, 1008], "vision requires [1,3,1008,1008]");
        let patch = self.patch.forward(&pixels.to_dtype(DType::F32)?)?;
        observe("patch", &patch)?;
        let mut x = self.norm.forward(&patch.permute((0, 2, 3, 1))?.broadcast_add(&self.positions)?.contiguous()?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, if block.global { &self.global_rope } else { &self.window_rope })?;
            if matches!(i, 0 | 7 | 31) {
                observe(&format!("block{i}"), &x)?;
            }
        }
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let mut pyramid = |branches: &[Fpn; 3], name: &str, interactive: bool| -> Result<[Tensor; 3]> {
            let mut outputs = Vec::with_capacity(3);
            for (i, branch) in branches.iter().enumerate() {
                let mut feature = branch.forward(&x)?;
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
        Ok(VisionOut { detector_fpn, interactive_fpn, image_pos: image_position(72, 72, x.device())? })
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
            norm1: l.norm(&format!("{p}.norm1"))?,
            norm2: l.norm(&format!("{p}.norm2"))?,
            qkv: l.linear(&format!("{p}.attn.qkv"), 1024, 3072)?,
            proj: l.linear(&format!("{p}.attn.proj"), 1024, 1024)?,
            fc1: l.linear(&format!("{p}.mlp.fc1"), 1024, 4736)?,
            fc2: l.linear(&format!("{p}.mlp.fc2"), 4736, 1024)?,
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
        let a = attention(&q, &k, &v, None, false, None, 256)?.transpose(1, 2)?.contiguous()?.reshape((b, h * w, 1024))?;
        let a = self.proj.forward(&a)?.reshape((b, h, w, 1024))?;
        let a = if self.global { a } else { window_unpartition(&a, 24, pad, (72, 72))? };
        let x = (x + a)?;
        Ok((&x + self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?)?)
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}
impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let input = self.weight.dim(1)?;
        let output = self.weight.dim(0)?;
        let mut shape = dims;
        let last = shape.last_mut().context("linear requires a channel dimension")?;
        ensure!(*last == input, "linear input width mismatch");
        *last = output;
        Ok(x.reshape((x.elem_count() / input, input))?
            .matmul(&self.weight.to_dtype(DType::F32)?.t()?)?
            .broadcast_add(&self.bias.to_dtype(DType::F32)?)?
            .reshape(shape)?)
    }
}
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}
impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(&self.weight.to_dtype(DType::F32)?, self.padding, self.stride, 1, 1)?;
        Ok(if let Some(bias) = &self.bias { y.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, bias.elem_count(), 1, 1))?)? } else { y })
    }
}

/// Nonoverlapping kernel=stride=2 transpose convolution via a channel projection.
/// This keeps CPU and accelerator execution on the same device-generic GEMM path.
pub fn transpose_conv2(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let (ic, oc, kh, kw) = weight.dims4()?;
    ensure!(c == ic && kh == 2 && kw == 2 && bias.dims() == [oc], "transpose convolution shape mismatch");
    let input = x.permute((0, 2, 3, 1))?.contiguous()?.reshape((b * h * w, c))?;
    let weight = weight.to_dtype(DType::F32)?.reshape((c, oc * 4))?;
    let y =
        input.to_dtype(DType::F32)?.matmul(&weight)?.reshape((b, h, w, oc, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((b, oc, h * 2, w * 2))?;
    Ok(y.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, oc, 1, 1))?)?)
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
            conv1: l.conv(&format!("{p}.conv_1x1"), [256, 512, 1024][level], 256, 1, 1, 0, true)?,
            conv3: l.conv(&format!("{p}.conv_3x3"), 256, 256, 3, 1, 1, true)?,
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
struct Loader<'a> {
    tensors: BufferedSafetensors,
    device: &'a Device,
}
impl Loader<'_> {
    fn get(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let x = self.tensors.load(key, self.device).with_context(|| format!("load {key}"))?;
        ensure!(x.dims() == shape, "{key}: expected {shape:?}, got {:?}", x.dims());
        ensure!(x.dtype() == DType::F16, "{key}: SAM checkpoint must store F16 weights");
        Ok(x)
    }
    fn norm(&self, p: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get(&format!("{p}.weight"), &[1024])?.to_dtype(DType::F32)?,
            self.get(&format!("{p}.bias"), &[1024])?.to_dtype(DType::F32)?,
            1e-5,
        ))
    }
    fn linear(&self, p: &str, ic: usize, oc: usize) -> Result<Linear> {
        Ok(Linear { weight: self.get(&format!("{p}.weight"), &[oc, ic])?, bias: self.get(&format!("{p}.bias"), &[oc])? })
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, p: &str, ic: usize, oc: usize, k: usize, stride: usize, padding: usize, bias: bool) -> Result<Conv> {
        Ok(Conv {
            weight: self.get(&format!("{p}.weight"), &[oc, ic, k, k])?,
            bias: if bias { Some(self.get(&format!("{p}.bias"), &[oc])?) } else { None },
            stride,
            padding,
        })
    }
}
