//! Two-way interactive attention and all four raw SAM mask hypotheses.

use crate::prompt::{Conv, Loader, PromptEncoder, channel_norm};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{LayerNorm, Linear, Module};
use std::path::Path;

/// Raw logits and scores before mask selection, object gating or image-size resampling.
pub struct InteractiveOutput {
    /// Four hypotheses in checkpoint token order, shaped `[1,4,288,288]`.
    pub masks: Tensor,
    /// Predicted IoUs for the four hypotheses, shaped `[1,4]`.
    pub iou: Tensor,
    /// Object-presence logit, shaped `[1,1]`.
    pub object_score: Tensor,
}

/// Interactive prompt encoder and mask decoder, independent of the image backbone.
pub struct InteractiveHeads {
    prompt: PromptEncoder,
    tokens: Tensor,
    blocks: Vec<Block>,
    final_attn: Attention,
    final_norm: LayerNorm,
    up0: Conv,
    up_norm: LayerNorm,
    up3: Conv,
    s0: Conv,
    s1: Conv,
    hyper: Vec<Mlp>,
    iou: Mlp,
    object: Mlp,
}
impl InteractiveHeads {
    /// Read raw tracker keys from an F16 checkpoint and compute with f32 parameters.
    /// Buffered loading avoids unsafe mapping and leaves unrelated model weights on the host.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).with_context(|| format!("read {}", path.as_ref().display()))?;
        let l = Loader { tensors: BufferedSafetensors::new(bytes)?, device };
        let p = "tracker.model.interactive_sam_mask_decoder";
        let tokens = Tensor::cat(
            &[
                l.get(&format!("{p}.obj_score_token.weight"), &[1, 256])?,
                l.get(&format!("{p}.iou_token.weight"), &[1, 256])?,
                l.get(&format!("{p}.mask_tokens.weight"), &[4, 256])?,
            ],
            0,
        )?
        .unsqueeze(0)?;
        Ok(Self {
            prompt: PromptEncoder::load(&l)?,
            tokens,
            blocks: (0..2).map(|i| Block::load(&l, &format!("{p}.transformer.layers.{i}"), i == 0)).collect::<Result<_>>()?,
            final_attn: Attention::load(&l, &format!("{p}.transformer.final_attn_token_to_image"), 128)?,
            final_norm: l.norm(&format!("{p}.transformer.norm_final_attn"), 256)?,
            up0: l.conv(&format!("{p}.output_upscaling.0"), 256, 64, 2, true)?,
            up_norm: l.norm(&format!("{p}.output_upscaling.1"), 64)?,
            up3: l.conv(&format!("{p}.output_upscaling.3"), 64, 32, 2, true)?,
            s0: l.conv(&format!("{p}.conv_s0"), 256, 32, 1, false)?,
            s1: l.conv(&format!("{p}.conv_s1"), 256, 64, 1, false)?,
            hyper: (0..4).map(|i| Mlp::load(&l, &format!("{p}.output_hypernetworks_mlps.{i}"), 32)).collect::<Result<_>>()?,
            iou: Mlp::load(&l, &format!("{p}.iou_prediction_head"), 4)?,
            object: Mlp::load(&l, &format!("{p}.pred_obj_score_head"), 1)?,
        })
    }
    /// Match heads.onnx with descending-resolution FPNs and already-packed point labels.
    /// FPN shapes are `[1,256,288,288]`, `[1,256,144,144]`, `[1,256,72,72]`;
    /// the last includes the backbone's no-memory embedding. Coordinates are in 1008-space.
    /// `None` replaces has_mask=0; feedback is `[1,1,1008,1008]` unthresholded logits.
    /// No padding is added here; use `pack_prompts` when starting with points and a box.
    pub fn forward(&self, fpn: [&Tensor; 3], points: &[[f32; 2]], labels: &[i32], mask_input: Option<&Tensor>) -> Result<InteractiveOutput> {
        for (x, size) in fpn.iter().zip([288, 144, 72]) {
            ensure!(x.dims() == [1, 256, size, size], "FPN must be [1,256,{size},{size}]");
        }
        let fpn = fpn.map(|x| x.to_dtype(DType::F32)).into_iter().collect::<candle_core::Result<Vec<_>>>()?;
        let sparse = self.prompt.sparse(points, labels)?;
        let tokens = Tensor::cat(&[&self.tokens, &sparse], 1)?;
        let mut queries = tokens.clone();
        let mut keys = (&fpn[2] + self.prompt.dense(mask_input)?)?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let pe = self.prompt.image_pe()?.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        for block in &self.blocks {
            (queries, keys) = block.forward(&queries, &keys, &tokens, &pe)?;
        }
        let attended = self.final_attn.forward(&(&queries + &tokens)?, &(&keys + &pe)?, &keys)?;
        queries = self.final_norm.forward(&(&queries + attended)?)?;
        let src = keys.transpose(1, 2)?.contiguous()?.reshape((1, 256, 72, 72))?;
        let up = channel_norm(&(self.up0.forward(&src)? + self.s1.forward(&fpn[1])?)?, &self.up_norm)?.gelu_erf()?;
        let up = (self.up3.forward(&up)? + self.s0.forward(&fpn[0])?)?.gelu_erf()?;
        let hyper = self.hyper.iter().enumerate().map(|(i, m)| m.forward(&queries.narrow(1, 2 + i, 1)?.squeeze(1)?)).collect::<Result<Vec<_>>>()?;
        Ok(InteractiveOutput {
            masks: Tensor::stack(&hyper, 1)?.matmul(&up.flatten_from(2)?)?.reshape((1, 4, 288, 288))?,
            iou: self.iou.forward(&queries.narrow(1, 1, 1)?.squeeze(1)?)?,
            object_score: self.object.forward(&queries.narrow(1, 0, 1)?.squeeze(1)?)?,
        })
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    width: usize,
}
impl Attention {
    fn load(l: &Loader<'_>, p: &str, width: usize) -> Result<Self> {
        Ok(Self {
            q: l.linear(&format!("{p}.q_proj"), 256, width)?,
            k: l.linear(&format!("{p}.k_proj"), 256, width)?,
            v: l.linear(&format!("{p}.v_proj"), 256, width)?,
            out: l.linear(&format!("{p}.out_proj"), width, 256)?,
            width,
        })
    }
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let heads = |x: &Tensor, l: &Linear| -> Result<Tensor> { Ok(l.forward(x)?.reshape((1, x.dim(1)?, 8, self.width / 8))?.transpose(1, 2)?.contiguous()?) };
        let (q, k, v) = (heads(q, &self.q)?, heads(k, &self.k)?, heads(v, &self.v)?);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / ((self.width / 8) as f64).sqrt()))?;
        let out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?.transpose(1, 2)?.contiguous()?;
        Ok(self.out.forward(&out.reshape((1, q.dim(2)?, self.width))?)?)
    }
}
struct Block {
    self_attn: Attention,
    to_image: Attention,
    to_token: Attention,
    norms: [LayerNorm; 4],
    lin1: Linear,
    lin2: Linear,
    first: bool,
}
impl Block {
    fn load(l: &Loader<'_>, p: &str, first: bool) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(l, &format!("{p}.self_attn"), 256)?,
            to_image: Attention::load(l, &format!("{p}.cross_attn_token_to_image"), 128)?,
            to_token: Attention::load(l, &format!("{p}.cross_attn_image_to_token"), 128)?,
            norms: [
                l.norm(&format!("{p}.norm1"), 256)?,
                l.norm(&format!("{p}.norm2"), 256)?,
                l.norm(&format!("{p}.norm3"), 256)?,
                l.norm(&format!("{p}.norm4"), 256)?,
            ],
            lin1: l.linear(&format!("{p}.mlp.lin1"), 256, 2048)?,
            lin2: l.linear(&format!("{p}.mlp.lin2"), 2048, 256)?,
            first,
        })
    }
    fn forward(&self, queries: &Tensor, keys: &Tensor, qp: &Tensor, kp: &Tensor) -> Result<(Tensor, Tensor)> {
        let queries = if self.first {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = (queries + qp)?;
            (queries + self.self_attn.forward(&q, &q, queries)?)?
        };
        let queries = self.norms[0].forward(&queries)?;
        let cross = self.to_image.forward(&(&queries + qp)?, &(keys + kp)?, keys)?;
        let queries = self.norms[1].forward(&(&queries + cross)?)?;
        let mlp = self.lin2.forward(&self.lin1.forward(&queries)?.relu()?)?;
        let queries = self.norms[2].forward(&(&queries + mlp)?)?;
        let cross = self.to_token.forward(&(keys + kp)?, &(&queries + qp)?, &queries)?;
        Ok((queries, self.norms[3].forward(&(keys + cross)?)?))
    }
}
struct Mlp {
    layers: [Linear; 3],
}
impl Mlp {
    fn load(l: &Loader<'_>, p: &str, o: usize) -> Result<Self> {
        Ok(Self {
            layers: [l.linear(&format!("{p}.layers.0"), 256, 256)?, l.linear(&format!("{p}.layers.1"), 256, 256)?, l.linear(&format!("{p}.layers.2"), 256, o)?],
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.layers[2].forward(&self.layers[1].forward(&self.layers[0].forward(x)?.relu()?)?.relu()?)?)
    }
}
