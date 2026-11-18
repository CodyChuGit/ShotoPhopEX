//! A green- and blue-screen keyer: a picture plus a rough hint of the subject to a clean alpha
//! and the subject's straight (un-mixed) foreground colour, in one pass.
//!
//! The design is CorridorKey's (Corridor Digital; its code and weights are CC BY-NC-SA
//! 4.0 with additional terms). This module is a clean-room engine for it, written from a
//! description of the computation, timm's Hiera (Apache-2.0) and the weight file's tensor names
//! and shapes, never from CorridorKey's code. It ships no weights: the user downloads
//! CorridorKey's checkpoint from the authors' Hugging Face page and loads it themselves (BYOM).
//!
//! The computation:
//! 1. The picture (sRGB 0-1) and the hint (0-1, high = subject) are squashed (bilinear) to a
//!    2048 x 2048 square; RGB is normalised with ImageNet's mean and std, the hint is not, and
//!    becomes the fourth channel.
//! 2. A Hiera Base+ backbone (4 input channels) gives four feature maps (strides 4 to 32).
//! 3. Two SegFormer-style heads (alpha: 1 channel, colour: 3) each bring every level to 256
//!    channels with a per-pixel linear layer, upsample to 1/4 size, concatenate deepest first,
//!    fuse with a 1x1 conv, batch norm (inference statistics), ReLU and a 1x1 classifier: logits.
//! 4. Both logit maps go up to full size (bilinear); sigmoid gives the coarse alpha and colour.
//! 5. A full-resolution refiner (normalised RGB + coarse alpha + coarse colour, 7 channels; a
//!    3x3 stem to 64 channels with group norm and ReLU, four residual blocks of dilation 1, 2, 4
//!    and 8, a 1x1 conv to 4 channels times 10) gives logit-space corrections, scaled by the
//!    user's refiner strength.
//! 6. alpha = sigmoid(alpha logits + correction 0), colour = sigmoid(colour logits +
//!    corrections 1-3); both are resized back to the picture (Lanczos 3).
//!
//! Norms and the sigmoids run in f32; the rest in the compute dtype (f16/bf16 on a GPU).
//! The finishing steps (despeckle, despill, linear-light premultiply) are the caller's.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::hiera::{Hiera, HieraConfig, Lin};
use crate::store::Store;

/// ImageNet's mean and standard deviation (RGB).
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// CorridorKey v1's checkpoint as its authors name it.
pub const CHECKPOINT: &str = "CorridorKey_v1.0.safetensors";

#[derive(Clone, Debug, PartialEq)]
pub struct GreenKeyConfig {
    pub hiera: HieraConfig,
    /// The decoder heads' common width.
    pub decoder_dim: usize,
    pub refiner_channels: usize,
    pub refiner_groups: usize,
    pub refiner_dilations: Vec<usize>,
    /// The refiner's output is multiplied by this before it is added to the logits.
    pub refiner_scale: f32,
    pub batch_norm_eps: f64,
    pub group_norm_eps: f64,
}

impl Default for GreenKeyConfig {
    fn default() -> Self {
        GreenKeyConfig {
            hiera: HieraConfig::base_plus(2048, 4),
            decoder_dim: 256,
            refiner_channels: 64,
            refiner_groups: 8,
            refiner_dilations: vec![1, 2, 4, 8],
            refiner_scale: 10.0,
            batch_norm_eps: 1e-5,
            group_norm_eps: 1e-5,
        }
    }
}

impl GreenKeyConfig {
    /// The square side the model works at.
    pub fn size(&self) -> usize {
        self.hiera.img_size
    }
}

fn names(prefix: &str, alts: &[&str]) -> Vec<String> {
    alts.iter().map(|a| format!("{prefix}{a}")).collect()
}

/// A 1x1 convolution as a matrix product on (1, C, H, W).
struct Pointwise {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Pointwise {
    fn load(st: &Store, weight: &[String], bias: Option<&[String]>, cin: usize, cout: usize, dtype: DType) -> Result<Pointwise> {
        let w = st.get_any(weight, &[cout, cin], dtype)?;
        let b = match bias {
            Some(b) => st.get_opt(b, &[cout], dtype)?,
            None => None,
        };
        Ok(Pointwise { weight: w, bias: b })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        let y = self.weight.matmul(&x.to_dtype(self.weight.dtype())?.reshape((c, h * w))?)?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(&b.unsqueeze(1)?)?,
            None => y,
        };
        Ok(y.reshape((1, (), h, w))?)
    }
}

/// Batch norm with stored running statistics, in f32, as one scale and shift per channel.
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn load(st: &Store, prefixes: &[String], c: usize, eps: f64) -> Result<BatchNorm> {
        let p = st.first_of(&prefixes.iter().map(|p| format!("{p}.running_mean")).collect::<Vec<_>>()).context("no batch norm statistics")?;
        let p = p.trim_end_matches(".running_mean");
        let g = st.get(&format!("{p}.weight"), &[c], DType::F32)?;
        let b = st.get(&format!("{p}.bias"), &[c], DType::F32)?;
        let mean = st.get(&format!("{p}.running_mean"), &[c], DType::F32)?;
        let var = st.get(&format!("{p}.running_var"), &[c], DType::F32)?;
        let scale = g.div(&(var + eps)?.sqrt()?)?;
        let shift = (b - mean.mul(&scale)?)?;
        Ok(BatchNorm { scale: scale.reshape((1, c, 1, 1))?, shift: shift.reshape((1, c, 1, 1))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.to_dtype(DType::F32)?.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

/// One SegFormer-style head.
struct Decoder {
    linear_c: Vec<Lin>,
    fuse: Pointwise,
    bn: BatchNorm,
    classifier: Pointwise,
}

impl Decoder {
    fn load(st: &Store, p: &str, dims: &[usize], width: usize, out: usize, eps: f64, dtype: DType) -> Result<Decoder> {
        let linear_c = dims
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let base = st.first_of(&names(p, &[&format!("linear_c{}.proj.weight", i + 1), &format!("linear_c{}.weight", i + 1)])).context("no linear_c")?;
                Lin::load(st, base.trim_end_matches(".weight"), c, width, true, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        let n = dims.len();
        let fuse = Pointwise::load(
            st,
            &names(p, &["linear_fuse.weight", "linear_fuse.conv.weight", "linear_fuse.0.weight"]),
            Some(&names(p, &["linear_fuse.bias", "linear_fuse.conv.bias", "linear_fuse.0.bias"])),
            n * width,
            width,
            dtype,
        )?;
        let bn = BatchNorm::load(st, &names(p, &["bn", "linear_fuse.bn", "linear_fuse.1", "batch_norm", "norm"]), width, eps)?;
        let classifier = Pointwise::load(
            st,
            &names(p, &["classifier.weight", "linear_pred.weight", "pred.weight", "head.weight"]),
            Some(&names(p, &["classifier.bias", "linear_pred.bias", "pred.bias", "head.bias"])),
            width,
            out,
            dtype,
        )?;
        Ok(Decoder { linear_c, fuse, bn, classifier })
    }

    /// The four stage maps to (1, out, H/4, W/4) f32 logits.
    fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let (_, _, h1, w1) = feats[0].dims4()?;
        let mut levels = Vec::with_capacity(feats.len());
        for (f, lin) in feats.iter().zip(&self.linear_c) {
            let (_, c, h, w) = f.dims4()?;
            let t = lin.forward(&f.reshape((c, h * w))?.t()?)?; // (hw, width)
            let m = t.t()?.reshape((1, (), h, w))?.to_dtype(DType::F32)?;
            let m = if (h, w) == (h1, w1) { m } else { m.upsample_bilinear2d(h1, w1, false)? };
            levels.push(m);
        }
        levels.reverse(); // deepest first
        let x = Tensor::cat(&levels, 1)?;
        let x = self.bn.forward(&self.fuse.forward(&x)?)?.relu()?;
        Ok(self.classifier.forward(&x)?.to_dtype(DType::F32)?)
    }
}

/// A 3x3 "same" convolution at stride 1, run in strips of output rows so that the im2col
/// buffer a backend builds stays bounded at full resolution.
struct Conv3 {
    weight: Tensor,
    bias: Option<Tensor>,
    dilation: usize,
}

/// Elements one strip's im2col buffer may hold.
const IM2COL_BUDGET: usize = 1 << 27;

impl Conv3 {
    fn load(st: &Store, name: &str, cin: usize, cout: usize, dilation: usize, dtype: DType) -> Result<Conv3> {
        Ok(Conv3 {
            weight: st.get(&format!("{name}.weight"), &[cout, cin, 3, 3], dtype)?,
            bias: st.get_opt(&[format!("{name}.bias")], &[cout], dtype)?,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv3_strips(x, &self.weight, self.bias.as_ref(), self.dilation, IM2COL_BUDGET)
    }
}

/// `x` (1, Cin, H, W) convolved with a (Cout, Cin, 3, 3) kernel at `dilation`, zero padding
/// `dilation`, computed in strips of rows of at most `budget` im2col elements.
fn conv3_strips(x: &Tensor, w: &Tensor, b: Option<&Tensor>, dilation: usize, budget: usize) -> Result<Tensor> {
    let (_, cin, h, wd) = x.dims4()?;
    let x = x.to_dtype(w.dtype())?;
    let pad = dilation;
    let xp = x.pad_with_zeros(2, pad, pad)?.pad_with_zeros(3, pad, pad)?;
    let rows = (budget / (wd * cin * 9).max(1)).clamp(1, h);
    let mut outs = Vec::with_capacity(h.div_ceil(rows));
    let mut y0 = 0;
    while y0 < h {
        let r = rows.min(h - y0);
        let strip = xp.narrow(2, y0, r + 2 * pad)?;
        outs.push(strip.conv2d(w, 0, 1, dilation, 1)?);
        y0 += r;
    }
    let y = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 2)? };
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, (), 1, 1))?)?,
        None => y,
    })
}

/// Group norm in f32, a group at a time (bounded f32 memory), with an optional residual added
/// after the affine step and an optional ReLU; the result in `dtype`.
struct GroupNorm {
    gamma: Tensor,
    beta: Tensor,
    groups: usize,
    eps: f64,
}

impl GroupNorm {
    fn load(st: &Store, prefixes: &[String], c: usize, groups: usize, eps: f64) -> Result<GroupNorm> {
        let p = st.first_of(&prefixes.iter().map(|p| format!("{p}.weight")).collect::<Vec<_>>()).context("no group norm")?;
        let p = p.trim_end_matches(".weight");
        ensure!(c.is_multiple_of(groups), "{c} channels do not split into {groups} groups");
        Ok(GroupNorm { gamma: st.get(&format!("{p}.weight"), &[c], DType::F32)?, beta: st.get(&format!("{p}.bias"), &[c], DType::F32)?, groups, eps })
    }

    fn forward(&self, x: &Tensor, residual: Option<&Tensor>, relu: bool, dtype: DType) -> Result<Tensor> {
        group_norm(x, &self.gamma, &self.beta, self.groups, self.eps, residual, relu, dtype)
    }
}

/// Mean over all elements, as means of rows then their mean (keeps f32 sums short).
fn mean_all(t: &Tensor, row: usize) -> Result<Tensor> {
    Ok(t.reshape(((), row))?.mean(1)?.mean_keepdim(0)?.reshape((1, 1, 1, 1))?)
}

#[allow(clippy::too_many_arguments)]
fn group_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, groups: usize, eps: f64, residual: Option<&Tensor>, relu: bool, dtype: DType) -> Result<Tensor> {
    let (_, c, _, w) = x.dims4()?;
    let cg = c / groups;
    let mut outs = Vec::with_capacity(groups);
    for g in 0..groups {
        let xs = x.narrow(1, g * cg, cg)?.to_dtype(DType::F32)?;
        let mean = mean_all(&xs, w)?;
        let xc = xs.broadcast_sub(&mean)?;
        let var = mean_all(&xc.sqr()?, w)?;
        let y = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
        let y = y.broadcast_mul(&gamma.narrow(0, g * cg, cg)?.reshape((1, cg, 1, 1))?)?.broadcast_add(&beta.narrow(0, g * cg, cg)?.reshape((1, cg, 1, 1))?)?;
        let y = match residual {
            Some(r) => (y + r.narrow(1, g * cg, cg)?.to_dtype(DType::F32)?)?,
            None => y,
        };
        let y = if relu { y.relu()? } else { y };
        outs.push(y.to_dtype(dtype)?);
    }
    Ok(Tensor::cat(&outs, 1)?)
}

struct ResBlock {
    conv1: Conv3,
    gn1: GroupNorm,
    conv2: Conv3,
    gn2: GroupNorm,
}

struct Refiner {
    stem: Conv3,
    stem_gn: GroupNorm,
    blocks: Vec<ResBlock>,
    out: Pointwise,
    scale: f32,
    dtype: DType,
}

impl Refiner {
    fn load(st: &Store, p: &str, c: &GreenKeyConfig, dtype: DType) -> Result<Refiner> {
        let ch = c.refiner_channels;
        let (g, eps) = (c.refiner_groups, c.group_norm_eps);
        let stem_name = st.first_of(&names(p, &["stem.0.weight", "stem.conv.weight", "stem.weight", "conv_in.weight"])).context("no refiner stem")?;
        let stem = Conv3::load(st, stem_name.trim_end_matches(".weight"), 7, ch, 1, dtype)?;
        let stem_gn = GroupNorm::load(st, &names(p, &["stem.1", "stem.gn", "stem.norm", "stem_gn", "gn_in"]), ch, g, eps)?;
        let blocks = c
            .refiner_dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let cands = names(p, &[&format!("res{}", i + 1), &format!("blocks.{i}"), &format!("layers.{i}")]);
                let base = st.first_of(&cands.iter().map(|b| format!("{b}.conv1.weight")).collect::<Vec<_>>()).context("no refiner block")?;
                let b = base.trim_end_matches(".conv1.weight");
                Ok(ResBlock {
                    conv1: Conv3::load(st, &format!("{b}.conv1"), ch, ch, d, dtype)?,
                    gn1: GroupNorm::load(st, &[format!("{b}.gn1"), format!("{b}.norm1")], ch, g, eps)?,
                    conv2: Conv3::load(st, &format!("{b}.conv2"), ch, ch, d, dtype)?,
                    gn2: GroupNorm::load(st, &[format!("{b}.gn2"), format!("{b}.norm2")], ch, g, eps)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let out = Pointwise::load(
            st,
            &names(p, &["final.weight", "out.weight", "head.weight", "final_conv.weight", "conv_out.weight"]),
            Some(&names(p, &["final.bias", "out.bias", "head.bias", "final_conv.bias", "conv_out.bias"])),
            ch,
            4,
            dtype,
        )?;
        Ok(Refiner { stem, stem_gn, blocks, out, scale: c.refiner_scale, dtype })
    }

    /// (1, 7, H, W) to (1, 4, H, W) f32 logit corrections (already times the scale).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = self.dtype;
        let mut h = self.stem_gn.forward(&self.stem.forward(&x.to_dtype(dt)?)?, None, true, dt)?;
        for b in &self.blocks {
            let y = b.gn1.forward(&b.conv1.forward(&h)?, None, true, dt)?;
            h = b.gn2.forward(&b.conv2.forward(&y)?, Some(&h), true, dt)?;
        }
        Ok((self.out.forward(&h)?.to_dtype(DType::F32)? * f64::from(self.scale))?)
    }
}

/// The keyer's raw outputs at its working size.
pub struct KeyTensors {
    /// (1, 1, S, S), 0-1.
    pub alpha: Tensor,
    /// (1, 3, S, S), straight sRGB 0-1.
    pub fg: Tensor,
}

/// A keyed picture at its own size.
pub struct KeyOutput {
    pub width: usize,
    pub height: usize,
    /// Linear coverage, 0-1, row-major.
    pub alpha: Vec<f32>,
    /// Straight (not premultiplied) sRGB foreground colour, 0-1, RGB interleaved.
    pub fg: Vec<f32>,
}

/// The keyer, resident.
pub struct GreenKeyer {
    config: GreenKeyConfig,
    encoder: Hiera,
    alpha: Decoder,
    fg: Decoder,
    refiner: Refiner,
    device: Device,
}

/// The compute dtype for `device`: f16 on a GPU (the reference runs in half precision), f32 on
/// the CPU.
pub fn default_dtype(device: &Device) -> DType {
    if device.is_cpu() { DType::F32 } else { DType::F16 }
}

/// The checkpoint at `path`: the file itself, or in a directory CorridorKey's own name, else
/// the first `.safetensors` file (names with "corridorkey" first).
pub fn find_checkpoint(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    let own = path.join(CHECKPOINT);
    if own.is_file() {
        return Some(own);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort_by_key(|p| (!p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().contains("corridorkey")), p.clone()));
    files.into_iter().next()
}

/// The prefix before `suffix` of the first name ending in it, preferring names starting with
/// one of `hints`.
fn prefix_of(names: &[String], suffix: &str, hints: &[&str]) -> Option<String> {
    let found: Vec<&String> = names.iter().filter(|n| n.ends_with(suffix)).collect();
    found.iter().find(|n| hints.iter().any(|h| n.starts_with(h))).or(found.first()).map(|n| n[..n.len() - suffix.len()].to_string())
}

impl GreenKeyer {
    /// Load the checkpoint at `path` (a file, or a directory holding one) onto `device`,
    /// computing in `dtype` (see [`default_dtype`]).
    pub fn load(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<GreenKeyer> {
        let path = path.as_ref();
        let file = find_checkpoint(path).with_context(|| format!("no CorridorKey checkpoint (.safetensors) at {}", path.display()))?;
        let st = Store::open(&file, device)?;
        Self::from_store(&st, GreenKeyConfig::default(), dtype, device)
    }

    pub(crate) fn from_store(st: &Store, mut config: GreenKeyConfig, dtype: DType, device: &Device) -> Result<GreenKeyer> {
        let all = st.names();
        let random = st.is_random();
        let enc = if random { Some("encoder.model.".to_string()) } else { prefix_of(&all, "patch_embed.proj.weight", &["encoder"]) };
        let enc = enc.context("no Hiera backbone (`…patch_embed.proj.weight`) in the weights")?;
        // The two heads: every prefix with a `linear_c1`, told apart by their output channels.
        let mut heads: Vec<String> =
            all.iter().filter_map(|n| n.strip_suffix("linear_c1.proj.weight").or_else(|| n.strip_suffix("linear_c1.weight"))).map(str::to_string).collect();
        heads.dedup();
        if random {
            heads = vec!["alpha_decoder.".into(), "fg_decoder.".into()];
        }
        if let Some(Some(s)) = heads.first().map(|h| st.shape(&format!("{h}linear_c1.proj.weight")).or_else(|| st.shape(&format!("{h}linear_c1.weight")))) {
            config.decoder_dim = s[0];
        }
        let out_channels = |h: &str| -> Option<usize> {
            ["classifier.weight", "linear_pred.weight", "pred.weight", "head.weight"].iter().find_map(|s| st.shape(&format!("{h}{s}"))).map(|s| s[0])
        };
        let (alpha_p, fg_p) = if random {
            (heads[0].clone(), heads[1].clone())
        } else {
            let a = heads.iter().find(|h| out_channels(h) == Some(1)).cloned().context("no alpha head (1 output channel) in the weights")?;
            let f = heads.iter().find(|h| out_channels(h) == Some(3)).cloned().context("no colour head (3 output channels) in the weights")?;
            (a, f)
        };
        let refiner_p = if random {
            "refiner.".to_string()
        } else {
            all.iter()
                .find(|n| st.shape(n).is_some_and(|s| s.len() == 4 && s[1] == 7 && s[2] == 3 && s[3] == 3))
                .and_then(|n| n.find("stem").map(|i| n[..i].to_string()).or_else(|| n.split_once('.').map(|(h, _)| format!("{h}."))))
                .unwrap_or_else(|| "refiner.".into())
        };
        if let Some(s) = st.shape(&format!("{refiner_p}stem.0.weight")).or_else(|| st.shape(&format!("{refiner_p}stem.conv.weight"))) {
            config.refiner_channels = s[0];
        }
        let encoder = Hiera::load(st, &enc, config.hiera.clone(), dtype)?;
        config.hiera = encoder.config().clone();
        let dims = config.hiera.stage_dims();
        let w = config.decoder_dim;
        let eps = config.batch_norm_eps;
        let alpha = Decoder::load(st, &alpha_p, &dims, w, 1, eps, dtype).context("alpha head")?;
        let fg = Decoder::load(st, &fg_p, &dims, w, 3, eps, dtype).context("colour head")?;
        let refiner = Refiner::load(st, &refiner_p, &config, dtype).context("refiner")?;
        Ok(GreenKeyer { config, encoder, alpha, fg, refiner, device: device.clone() })
    }

    pub fn config(&self) -> &GreenKeyConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The four backbone maps for a (1, 4, S, S) model input.
    pub fn features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        self.encoder.forward(x)
    }

    /// (1, 4, S, S) model input (normalised RGB + hint) to alpha and colour at S x S.
    /// `refiner_strength` scales the refiner's corrections (1 = as trained, 0 = coarse only).
    pub fn forward(&self, x: &Tensor, refiner_strength: f32) -> Result<KeyTensors> {
        let s = self.config.size();
        let x = x.to_device(&self.device)?.to_dtype(DType::F32)?;
        let feats = self.encoder.forward(&x)?;
        let a_logits = self.alpha.forward(&feats)?.upsample_bilinear2d(s, s, false)?;
        let f_logits = self.fg.forward(&feats)?.upsample_bilinear2d(s, s, false)?;
        drop(feats);
        if refiner_strength == 0.0 {
            return Ok(KeyTensors { alpha: candle_nn::ops::sigmoid(&a_logits)?, fg: candle_nn::ops::sigmoid(&f_logits)? });
        }
        let a0 = candle_nn::ops::sigmoid(&a_logits)?;
        let f0 = candle_nn::ops::sigmoid(&f_logits)?;
        let rin = Tensor::cat(&[&x.narrow(1, 0, 3)?, &a0, &f0], 1)?;
        drop((a0, f0));
        let corr = (self.refiner.forward(&rin)? * f64::from(refiner_strength))?;
        let alpha = candle_nn::ops::sigmoid(&(a_logits + corr.narrow(1, 0, 1)?)?)?;
        let fg = candle_nn::ops::sigmoid(&(f_logits + corr.narrow(1, 1, 3)?)?)?;
        Ok(KeyTensors { alpha, fg })
    }

    /// Key an RGBA8 picture (sRGB; its alpha ignored) with a hint (0-1, high = subject), both
    /// `width x height`.
    pub fn key(&self, rgba: &[u8], hint: &[f32], width: usize, height: usize, refiner_strength: f32) -> Result<KeyOutput> {
        let s = self.config.size();
        let x = prepare(rgba, hint, width, height, s)?;
        let x = Tensor::from_vec(x, (1, 4, s, s), &self.device)?;
        let out = self.forward(&x, refiner_strength)?;
        let alpha = out.alpha.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let fg = out.fg.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let back = |p: &[f32]| -> Vec<f32> { lanczos3(p, s, s, width, height).into_iter().map(|v| v.clamp(0.0, 1.0)).collect() };
        let alpha = back(&alpha);
        let planes: Vec<Vec<f32>> = (0..3).map(|c| back(&fg[c * s * s..(c + 1) * s * s])).collect();
        let mut rgb = Vec::with_capacity(width * height * 3);
        for ((r, g), b) in planes[0].iter().zip(&planes[1]).zip(&planes[2]) {
            rgb.extend([*r, *g, *b]);
        }
        Ok(KeyOutput { width, height, alpha, fg: rgb })
    }
}

/// The model input for an RGBA8 sRGB picture and a hint: four planes at `size` x `size`
/// (bilinear, antialiased when shrinking, as torchvision's default), RGB normalised, the hint
/// as is.
pub fn prepare(rgba: &[u8], hint: &[f32], width: usize, height: usize, size: usize) -> Result<Vec<f32>> {
    let n = width * height;
    ensure!(n > 0 && rgba.len() == n * 4 && hint.len() == n, "picture and hint must be {width}x{height}");
    let mut out = Vec::with_capacity(4 * size * size);
    for c in 0..3 {
        let plane: Vec<f32> = rgba.chunks_exact(4).map(|p| f32::from(p[c]) / 255.0).collect();
        out.extend(bilinear_aa(&plane, width, height, size, size).into_iter().map(|v| (v - MEAN[c]) / STD[c]));
    }
    out.extend(bilinear_aa(hint, width, height, size, size).into_iter().map(|v| v.clamp(0.0, 1.0)));
    Ok(out)
}

/// Separable Lanczos-3 resampling of one plane (support widened when shrinking, as PIL does).
pub fn lanczos3(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    resample(src, sw, sh, dw, dh, Kernel::Lanczos3)
}

/// Bilinear resampling with antialiasing when shrinking: torchvision's default
/// (`antialias=True`, PIL's triangle filter); the same as plain half-pixel bilinear when growing.
pub fn bilinear_aa(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    resample(src, sw, sh, dw, dh, Kernel::Triangle)
}

#[derive(Clone, Copy)]
enum Kernel {
    Triangle,
    Lanczos3,
}

fn resample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, kernel: Kernel) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() < sw * sh {
        return vec![0.0; dw * dh];
    }
    if (sw, sh) == (dw, dh) {
        return src[..sw * sh].to_vec();
    }
    let horiz = taps(sw, dw, kernel);
    let vert = taps(sh, dh, kernel);
    let mut mid = vec![0f32; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for (x, (start, ws)) in horiz.iter().enumerate() {
            mid[y * dw + x] = ws.iter().enumerate().map(|(k, w)| w * row[start + k]).sum();
        }
    }
    let mut out = vec![0f32; dw * dh];
    for (y, (start, ws)) in vert.iter().enumerate() {
        for x in 0..dw {
            out[y * dw + x] = ws.iter().enumerate().map(|(k, w)| w * mid[(start + k) * dw + x]).sum();
        }
    }
    out
}

/// Per output index: the first source index and the normalised weights.
fn taps(src: usize, dst: usize, kernel: Kernel) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let radius = match kernel {
        Kernel::Triangle => 1.0,
        Kernel::Lanczos3 => 3.0,
    };
    let support = radius * scale.max(1.0);
    let filter_scale = scale.max(1.0);
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        if matches!(kernel, Kernel::Triangle) {
            (1.0 - x).max(0.0)
        } else if x < 1e-9 {
            1.0
        } else if x < 3.0 {
            let px = std::f64::consts::PI * x;
            3.0 * px.sin() * (px / 3.0).sin() / (px * px)
        } else {
            0.0
        }
    };
    (0..dst)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            let lo = ((centre - support).floor().max(0.0)) as usize;
            let hi = ((centre + support).ceil() as usize).min(src);
            let mut ws: Vec<f64> = (lo..hi).map(|j| filter((j as f64 + 0.5 - centre) / filter_scale)).collect();
            let sum: f64 = ws.iter().sum();
            if sum.abs() > 1e-12 {
                for w in &mut ws {
                    *w /= sum;
                }
            }
            (lo, ws.into_iter().map(|w| w as f32).collect())
        })
        .collect()
}

/// The keyer's output as straight RGBA8: colour from `fg`, alpha from `alpha`.
pub fn to_rgba8(out: &KeyOutput) -> Vec<u8> {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut px = Vec::with_capacity(out.width * out.height * 4);
    for (a, c) in out.alpha.iter().zip(out.fg.chunks_exact(3)) {
        px.extend([q(c[0]), q(c[1]), q(c[2]), q(*a)]);
    }
    px
}

#[cfg(test)]
#[path = "greenkey_tests.rs"]
mod tests;
