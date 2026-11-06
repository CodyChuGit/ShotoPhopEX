//! Private scoring and two-stage instance segmentation heads.

use crate::linear::Linear;
use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{LayerNorm, Module};

use crate::{
    detector::{Attention, Mlp},
    prompt::Loader,
};

pub(crate) struct Heads {
    prompt_mlp: Mlp,
    prompt_norm: LayerNorm,
    hs_proj: Linear,
    prompt_proj: Linear,
    cross: Attention,
    cross_norm: LayerNorm,
    stages: Vec<(Conv, SpatialGroupNorm)>,
    instance: Conv,
    mask: Mlp,
}
impl Heads {
    pub(crate) fn load(l: &Loader<'_>) -> Result<Self> {
        let s = "detector.dot_prod_scoring";
        let p = "detector.segmentation_head";
        Ok(Self {
            prompt_mlp: Mlp::load(l, &format!("{s}.prompt_mlp"), &[256, 2048, 256])?,
            prompt_norm: l.norm(&format!("{s}.prompt_mlp.out_norm"), 256)?,
            hs_proj: l.linear(&format!("{s}.hs_proj"), 256, 256)?,
            prompt_proj: l.linear(&format!("{s}.prompt_proj"), 256, 256)?,
            cross: Attention::load(l, &format!("{p}.cross_attend_prompt"))?,
            cross_norm: l.norm(&format!("{p}.cross_attn_norm"), 256)?,
            // Three levels execute only stages 0 and 1; semantic output is unused.
            stages: (0..2)
                .map(|i| {
                    Ok((
                        Conv::load(l, &format!("{p}.pixel_decoder.conv_layers.{i}"), 3)?,
                        SpatialGroupNorm::new(
                            l.get(&format!("{p}.pixel_decoder.norms.{i}.weight"), &[256])?,
                            l.get(&format!("{p}.pixel_decoder.norms.{i}.bias"), &[256])?,
                        ),
                    ))
                })
                .collect::<Result<_>>()?,
            instance: Conv::load(l, &format!("{p}.instance_seg_head"), 1)?,
            mask: Mlp::load(l, &format!("{p}.mask_predictor.mask_embed"), &[256, 256, 256, 256])?,
        })
    }
    pub(crate) fn forward(
        &self,
        query: &Tensor,
        memory: &Tensor,
        fpn: [&Tensor; 3],
        prompt: &Tensor,
        valid: &Tensor,
        observe: &mut impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<(Tensor, Tensor)> {
        let p = self.prompt_norm.forward(&(self.prompt_mlp.forward(prompt)? + prompt)?)?;
        let weight = valid.to_dtype(prompt.dtype())?.unsqueeze(2)?;
        let pooled = p.broadcast_mul(&weight)?.sum(1)?.broadcast_div(&weight.sum(1)?.clamp(1.0, f32::MAX)?)?;
        let logits =
            (self.hs_proj.forward(query)?.matmul(&self.prompt_proj.forward(&pooled)?.unsqueeze(2)?)? * (1.0 / 16.0))?.clamp(-12.0, 12.0)?.squeeze(2)?;
        let cross = self.cross.forward(&self.cross_norm.forward(memory)?, prompt, prompt, Some(valid), None)?;
        let mut pixels = (memory + cross)?.transpose(1, 2)?.contiguous()?.reshape((1, 256, 72, 72))?;
        for (i, ((conv, norm), feat)) in self.stages.iter().zip([fpn[1], fpn[0]]).enumerate() {
            let (_, _, h, w) = feat.dims4()?;
            pixels = conv.forward(&(pixels.upsample_nearest2d(h, w)? + feat.to_dtype(memory.dtype())?)?)?;
            observe(if i == 0 { "observe_conv2d" } else { "observe_conv2d_1" }, &pixels)?;
            pixels = norm.forward(&pixels)?;
            observe(if i == 0 { "observe_group_norm" } else { "observe_group_norm_1" }, &pixels)?;
            pixels = pixels.relu()?;
        }
        let pixels = self.instance.forward(&pixels)?.flatten_from(2)?;
        observe("observe_conv2d_2", &pixels.reshape((1, 256, 288, 288))?)?;
        let hyper = self.mask.forward(query)?;
        observe("observe_linear_211", &hyper)?;
        let masks = hyper.matmul(&pixels)?.reshape((1, 200, 288, 288))?;
        Ok((logits, masks))
    }
}

/// Native convolution keeps this head independent of cuDNN.
struct Conv {
    pointwise: Option<Linear>,
    fused_weight: Option<Tensor>,
    weight: Tensor,
    bias: Tensor,
    padding: usize,
}
impl Conv {
    fn load(l: &Loader<'_>, p: &str, k: usize) -> Result<Self> {
        let weight = l.get(&format!("{p}.weight"), &[256, 256, k, k])?;
        let bias = l.get(&format!("{p}.bias"), &[256])?;
        let pointwise = if l.device.is_cuda() && k == 1 { Some(Linear::new(weight.reshape((256, 256))?, Some(bias.clone()))?) } else { None };
        let fused_weight = if l.device.is_cuda() && k > 1 { Some(crate::ops::fuse_conv_bias(&weight, &bias)?) } else { None };
        Ok(Self { weight, bias, pointwise, fused_weight, padding: k / 2 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(pointwise) = &self.pointwise {
            return Ok(pointwise.forward_channels(x)?);
        }
        if let Some(weight) = &self.fused_weight {
            return Ok(crate::ops::bias_channels(x)?.conv2d(weight, self.padding, 1, 1, 1)?);
        }
        Ok(x.conv2d(&self.weight, self.padding, 1, 1, 1)?.broadcast_add(&self.bias.reshape((1, 256, 1, 1))?)?)
    }
}

/// Reduce spatial groups in blocks so f32 accumulation does not drift over millions of pixels.
struct SpatialGroupNorm {
    weight: Tensor,
    bias: Tensor,
}
impl SpatialGroupNorm {
    fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let n = c / 8 * h * w;
        let groups = x.to_dtype(DType::F32)?.reshape((b, 8, n))?;
        let mean = spatial_mean(&groups)?;
        let centered = groups.broadcast_sub(&mean)?;
        let variance = spatial_mean(&centered.sqr()?)?;
        let normalized = centered.broadcast_div(&(variance + 1e-5)?.sqrt()?)?.reshape((b, c, h, w))?;
        Ok(normalized.to_dtype(x.dtype())?.broadcast_mul(&self.weight.reshape((1, c, 1, 1))?)?.broadcast_add(&self.bias.reshape((1, c, 1, 1))?)?)
    }
}
fn spatial_mean(x: &Tensor) -> Result<Tensor> {
    let (b, g, n) = x.dims3()?;
    let chunks = x.reshape((b, g, n / 1024, 1024))?;
    Ok(chunks.mean_keepdim(3)?.mean_keepdim(2)?.reshape((b, g, 1))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn spatial_group_statistics_retain_small_variation() -> Result<()> {
        // One 32-channel 288-square group. Its alternating binary-exact values
        // have mean 1.125 and population variance 0.25, even at this reduction size.
        let n = 32 * 288 * 288;
        let values: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.625 } else { 1.625 }).collect();
        let x = Tensor::from_vec(values, (1, 1, n), &Device::Cpu)?;
        let mean = spatial_mean(&x)?;
        let variance = spatial_mean(&x.broadcast_sub(&mean)?.sqr()?)?;
        assert!((mean.flatten_all()?.to_vec1::<f32>()?[0] - 1.125).abs() < 1e-6);
        assert!((variance.flatten_all()?.to_vec1::<f32>()?[0] - 0.25).abs() < 1e-6);
        Ok(())
    }
}
