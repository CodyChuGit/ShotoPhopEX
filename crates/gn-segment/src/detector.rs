//! SAM 3.1 image/prompt transformer and normalized detector outputs.

use crate::linear::Linear;
use std::path::Path;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module};

use crate::{detector_heads::Heads, prompt::Loader, weights::split_qkv};

/// Raw detector outputs, before score thresholds or image-size resampling.
pub struct DetectorOut {
    /// Normalized xyxy coordinates, `[1,200,4]`; corners may extend outside the image.
    pub boxes: Tensor,
    /// Clamped classification logits, `[1,200]`.
    pub logits: Tensor,
    /// Instance mask logits, `[1,200,288,288]`.
    pub masks: Tensor,
    /// Presence logit, `[1,1]`; the current worker does not multiply scores by it.
    pub presence: Tensor,
}

/// Detector transformer and heads, consuming already-resized and assembled prompts.
pub struct Detector {
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
    queries: Tensor,
    references: Tensor,
    presence: Tensor,
    norm: LayerNorm,
    presence_norm: LayerNorm,
    ref_head: Mlp,
    bbox: Mlp,
    presence_head: Mlp,
    bias_x: Mlp,
    bias_y: Mlp,
    heads: Heads,
}

impl Detector {
    /// Load detector weights once using the checkpoint compute precision, without cuDNN.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_checkpoint(&crate::weights::Checkpoint::open(path)?, device)
    }

    /// Load this part from the shared checkpoint buffer.
    pub fn from_checkpoint(checkpoint: &crate::weights::Checkpoint, device: &Device) -> Result<Self> {
        let l = Loader { checkpoint, device };
        let p = "detector.transformer.decoder";
        Ok(Self {
            encoder: (0..6).map(|i| EncoderLayer::load(&l, &format!("detector.transformer.encoder.layers.{i}"))).collect::<Result<_>>()?,
            decoder: (0..6).map(|i| DecoderLayer::load(&l, &format!("{p}.layers.{i}"))).collect::<Result<_>>()?,
            queries: l.get(&format!("{p}.query_embed.weight"), &[200, 256])?.unsqueeze(0)?,
            references: l.get(&format!("{p}.reference_points.weight"), &[200, 4])?.unsqueeze(0)?,
            presence: l.get(&format!("{p}.presence_token.weight"), &[1, 256])?.unsqueeze(0)?,
            norm: l.norm(&format!("{p}.norm"), 256)?,
            presence_norm: l.norm(&format!("{p}.presence_token_out_norm"), 256)?,
            ref_head: Mlp::load(&l, &format!("{p}.ref_point_head"), &[512, 256, 256])?,
            bbox: Mlp::load(&l, &format!("{p}.bbox_embed"), &[256, 256, 256, 4])?,
            presence_head: Mlp::load(&l, &format!("{p}.presence_token_head"), &[256, 256, 256, 1])?,
            bias_x: Mlp::load(&l, &format!("{p}.boxRPB_embed_x"), &[2, 256, 8])?,
            bias_y: Mlp::load(&l, &format!("{p}.boxRPB_embed_y"), &[2, 256, 8])?,
            heads: Heads::load(&l)?,
        })
    }

    /// Run three descending-resolution FPN levels and image positions `[1,256,72,72]`.
    /// Prompt `[1,T,256]` includes resized text, geometry and CLS; validity `[1,T]`
    /// uses one for valid keys. TextEncoder already applies the language resizer.
    pub fn forward(&self, fpn: [&Tensor; 3], image_pos: &Tensor, prompt: &Tensor, prompt_valid: &Tensor) -> Result<DetectorOut> {
        Ok(self.run(fpn, image_pos, prompt, prompt_valid, &mut |_, _| Ok(()))?.0)
    }

    /// Also expose the six refined cxcywh references for isolated parity diagnostics.
    pub fn forward_with_references(&self, fpn: [&Tensor; 3], image_pos: &Tensor, prompt: &Tensor, prompt_valid: &Tensor) -> Result<(DetectorOut, Vec<Tensor>)> {
        self.run(fpn, image_pos, prompt, prompt_valid, &mut |_, _| Ok(()))
    }

    /// Observe decoder and segmentation boundaries when diagnosing precision changes.
    pub fn forward_with_observer(
        &self,
        fpn: [&Tensor; 3],
        image_pos: &Tensor,
        prompt: &Tensor,
        valid: &Tensor,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<DetectorOut> {
        Ok(self.run(fpn, image_pos, prompt, valid, &mut observe)?.0)
    }
    fn run(
        &self,
        fpn: [&Tensor; 3],
        image_pos: &Tensor,
        prompt: &Tensor,
        valid: &Tensor,
        observe: &mut impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<(DetectorOut, Vec<Tensor>)> {
        for (x, size) in fpn.iter().zip([288, 144, 72]) {
            ensure!(x.dims() == [1, 256, size, size], "FPN must be [1,256,{size},{size}]");
        }
        ensure!(image_pos.dims() == [1, 256, 72, 72], "image positions must be [1,256,72,72]");
        let (b, t, c) = prompt.dims3()?;
        ensure!(b == 1 && t > 0 && c == 256 && valid.dims() == [1, t], "prompt must be [1,T,256], validity [1,T], T>0");
        let prompt = prompt.to_dtype(self.queries.dtype())?;
        let valid = valid.ne(0.0)?.to_dtype(DType::U8)?;
        let pos = image_pos.to_dtype(self.queries.dtype())?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let mut memory = fpn[2].to_dtype(self.queries.dtype())?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        crate::profile::measure(self.queries.device(), "detector encoder", || {
            for layer in &self.encoder {
                memory = layer.forward(&memory, &pos, &prompt, &valid)?;
            }
            Ok(())
        })?;
        observe("observe_add_218", &memory)?;
        let mut x = Tensor::cat(&[&self.presence, &self.queries], 1)?;
        let mut refs = candle_nn::ops::sigmoid(&self.references)?;
        let mut references = Vec::with_capacity(6);
        crate::profile::measure(self.queries.device(), "detector decoder", || {
            for layer in &self.decoder {
                let query_pos = self.ref_head.forward(&sine_embed(&refs)?.to_dtype(self.queries.dtype())?)?;
                let qp = Tensor::cat(&[self.presence.zeros_like()?, query_pos], 1)?;
                let bias = box_bias(&refs, 72, 72, &self.bias_x, &self.bias_y)?;
                x = layer.forward(&x, &qp, &memory, &pos, &prompt, &valid, &bias)?;
                let query = self.norm.forward(&x.narrow(1, 1, 200)?)?;
                refs = candle_nn::ops::sigmoid(&(inverse_sigmoid(&refs)? + self.bbox.forward(&query)?)?)?;
                references.push(refs.clone());
            }
            Ok(())
        })?;
        let query = self.norm.forward(&x.narrow(1, 1, 200)?)?;
        observe("observe_layer_norm_47", &query)?;
        let presence = self.presence_head.forward(&self.presence_norm.forward(&x.narrow(1, 0, 1)?)?)?.squeeze(2)?;
        let (logits, masks) =
            crate::profile::measure(self.queries.device(), "detector heads", || self.heads.forward(&query, &memory, fpn, &prompt, &valid, observe))?;
        Ok((DetectorOut { boxes: xyxy(&refs)?, logits, masks, presence }, references))
    }
}

pub(crate) struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
impl Attention {
    pub(crate) fn load(l: &Loader<'_>, p: &str) -> Result<Self> {
        let (qw, kw, vw) = split_qkv(&l.get(&format!("{p}.in_proj_weight"), &[768, 256])?)?;
        let (qb, kb, vb) = split_qkv(&l.get(&format!("{p}.in_proj_bias"), &[768])?)?;
        Ok(Self {
            q: Linear::new(qw.contiguous()?, Some(qb))?,
            k: Linear::new(kw.contiguous()?, Some(kb))?,
            v: Linear::new(vw.contiguous()?, Some(vb))?,
            out: l.linear(&format!("{p}.out_proj"), 256, 256)?,
        })
    }
    pub(crate) fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, valid: Option<&Tensor>, bias: Option<&Tensor>) -> Result<Tensor> {
        let heads = |x: &Tensor, l: &Linear| -> Result<Tensor> { Ok(l.forward(x)?.reshape((1, x.dim(1)?, 8, 32))?.transpose(1, 2)?.contiguous()?) };
        let out = crate::attention::attention(&heads(q, &self.q)?, &heads(k, &self.k)?, &heads(v, &self.v)?, valid, false, bias, 128)?;
        Ok(self.out.forward(&out.transpose(1, 2)?.contiguous()?.reshape((1, q.dim(1)?, 256))?)?)
    }
}

pub(crate) struct Mlp(Vec<Linear>);
impl Mlp {
    pub(crate) fn load(l: &Loader<'_>, p: &str, dims: &[usize]) -> Result<Self> {
        Ok(Self(dims.windows(2).enumerate().map(|(i, d)| l.linear(&format!("{p}.layers.{i}"), d[0], d[1])).collect::<Result<_>>()?))
    }
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, layer) in self.0.iter().enumerate() {
            x = layer.forward(&x)?;
            if i + 1 < self.0.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

struct EncoderLayer {
    sa: Attention,
    ca: Attention,
    norms: [LayerNorm; 3],
    ff1: Linear,
    ff2: Linear,
}
impl EncoderLayer {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self> {
        Ok(Self {
            sa: Attention::load(l, &format!("{p}.self_attn"))?,
            ca: Attention::load(l, &format!("{p}.cross_attn_image"))?,
            norms: [l.norm(&format!("{p}.norm1"), 256)?, l.norm(&format!("{p}.norm2"), 256)?, l.norm(&format!("{p}.norm3"), 256)?],
            ff1: l.linear(&format!("{p}.linear1"), 256, 2048)?,
            ff2: l.linear(&format!("{p}.linear2"), 2048, 256)?,
        })
    }
    fn forward(&self, x: &Tensor, pos: &Tensor, prompt: &Tensor, valid: &Tensor) -> Result<Tensor> {
        let n = self.norms[0].forward(x)?;
        let qk = (&n + pos)?;
        let x = (x + self.sa.forward(&qk, &qk, &n, None, None)?)?;
        let x = (&x + self.ca.forward(&self.norms[1].forward(&x)?, prompt, prompt, Some(valid), None)?)?;
        Ok((&x + self.ff2.forward(&self.ff1.forward(&self.norms[2].forward(&x)?)?.relu()?)?)?)
    }
}

struct DecoderLayer {
    sa: Attention,
    ca: Attention,
    text: Attention,
    norms: [LayerNorm; 4],
    ff1: Linear,
    ff2: Linear,
}
impl DecoderLayer {
    fn load(l: &Loader<'_>, p: &str) -> Result<Self> {
        Ok(Self {
            sa: Attention::load(l, &format!("{p}.self_attn"))?,
            ca: Attention::load(l, &format!("{p}.cross_attn"))?,
            text: Attention::load(l, &format!("{p}.ca_text"))?,
            norms: [
                l.norm(&format!("{p}.norm2"), 256)?,
                l.norm(&format!("{p}.catext_norm"), 256)?,
                l.norm(&format!("{p}.norm1"), 256)?,
                l.norm(&format!("{p}.norm3"), 256)?,
            ],
            ff1: l.linear(&format!("{p}.linear1"), 256, 2048)?,
            ff2: l.linear(&format!("{p}.linear2"), 2048, 256)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, x: &Tensor, qp: &Tensor, memory: &Tensor, pos: &Tensor, prompt: &Tensor, valid: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let qk = (x + qp)?;
        let x = self.norms[0].forward(&(x + self.sa.forward(&qk, &qk, x, None, None)?)?)?;
        let x = self.norms[1].forward(&(&x + self.text.forward(&(&x + qp)?, prompt, prompt, Some(valid), None)?)?)?;
        let x = self.norms[2].forward(&(&x + self.ca.forward(&(&x + qp)?, &(memory + pos)?, memory, None, Some(bias))?)?)?;
        Ok(self.norms[3].forward(&(&x + self.ff2.forward(&self.ff1.forward(&x)?.relu()?)?)?)?)
    }
}

fn inverse_sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok((x / ((x.neg()? + 1.0)? + 1e-6)?)?.affine(1.0, 1e-6)?.log()?)
}
fn xyxy(x: &Tensor) -> Result<Tensor> {
    let center = x.narrow(2, 0, 2)?;
    let half = (x.narrow(2, 2, 2)? * 0.5)?;
    Ok(Tensor::cat(&[(&center - &half)?, (&center + &half)?], 2)?)
}

/// Reference coordinates use 128 alternating sine/cosine features per coordinate.
fn sine_embed(refs: &Tensor) -> Result<Tensor> {
    let device = refs.device();
    let freq: Vec<f32> = (0..64).map(|i| 10000_f32.powf((2 * i) as f32 / 128.0)).collect();
    let freq = Tensor::from_vec(freq, (1, 1, 64), device)?;
    let refs_f32 = refs.to_dtype(DType::F32)?;
    let mut coords = Vec::with_capacity(4);
    for c in 0..4 {
        let angle = (refs_f32.narrow(2, c, 1)? * std::f64::consts::TAU)?.broadcast_div(&freq)?;
        coords.push(Tensor::stack(&[angle.sin()?, angle.cos()?], 3)?.flatten_from(2)?);
    }
    Ok(Tensor::cat(&coords, 2)?.to_dtype(refs.dtype())?)
}

/// Signed-log distances to both edges are learned additive biases, not Q/K rotations.
fn box_bias(refs: &Tensor, h: usize, w: usize, mx: &Mlp, my: &Mlp) -> Result<Tensor> {
    let boxes = xyxy(refs)?.to_dtype(DType::F32)?;
    let axis = |size: usize, c: usize, mlp: &Mlp| -> Result<Tensor> {
        let coords = Tensor::arange(0_f32, size as f32, refs.device())?.affine(1.0 / size as f64, 0.0)?.reshape((1, 1, size, 1))?;
        let edges = Tensor::cat(&[boxes.narrow(2, c, 1)?, boxes.narrow(2, c + 2, 1)?], 2)?.unsqueeze(2)?;
        let d = coords.broadcast_sub(&edges)?.affine(8.0, 0.0)?;
        // d/(|d|+epsilon) would alter small distances; compare explicitly for exact sign.
        let sign = (d.gt(0.0)?.to_dtype(DType::F32)? - d.lt(0.0)?.to_dtype(DType::F32)?)?;
        let scaled = (sign * (d.abs()? + 1.0)?.log()?)?.affine(1.0 / 8_f64.ln(), 0.0)?;
        mlp.forward(&scaled.to_dtype(refs.dtype())?)
    };
    let x = axis(w, 0, mx)?;
    let y = axis(h, 1, my)?;
    let bias = y.unsqueeze(3)?.broadcast_add(&x.unsqueeze(2)?)?.reshape((1, 200, h * w, 8))?.permute((0, 3, 1, 2))?.contiguous()?;
    Ok(Tensor::cat(&[Tensor::zeros((1, 8, 1, h * w), refs.dtype(), refs.device())?, bias], 2)?)
}
