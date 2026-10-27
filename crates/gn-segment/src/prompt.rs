//! Pixel-centered Fourier prompts and mask feedback for the interactive SAM heads.

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{LayerNorm, Linear, Module};

/// Pack already-scaled 1008-space points and a box exactly as the existing worker does.
/// Empty points retain a placeholder even when a box follows; no box adds another placeholder.
pub fn pack_prompts(points: &[([f32; 2], i32)], bbox: Option<[f32; 4]>) -> (Vec<[f32; 2]>, Vec<i32>) {
    let (mut coords, mut labels): (Vec<_>, Vec<_>) = points.iter().copied().unzip();
    if points.is_empty() {
        coords.push([0.0; 2]);
        labels.push(-1);
    }
    if let Some([x0, y0, x1, y1]) = bbox {
        coords.extend([[x0, y0], [x1, y1]]);
        labels.extend([2, 3]);
    } else {
        coords.push([0.0; 2]);
        labels.push(-1);
    }
    (coords, labels)
}

pub(crate) struct Loader<'a> {
    pub(crate) tensors: BufferedSafetensors,
    pub(crate) device: &'a Device,
}
impl Loader<'_> {
    pub(crate) fn get(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.tensors.load(key, self.device)?;
        ensure!(t.dims() == shape, "{key}: expected {shape:?}, got {:?}", t.dims());
        ensure!(t.dtype() == DType::F16, "{key}: expected checkpoint F16");
        Ok(t.to_dtype(DType::F32)?)
    }
    pub(crate) fn linear(&self, p: &str, i: usize, o: usize) -> Result<Linear> {
        Ok(Linear::new(self.get(&format!("{p}.weight"), &[o, i])?, Some(self.get(&format!("{p}.bias"), &[o])?)))
    }
    pub(crate) fn norm(&self, p: &str, c: usize) -> Result<LayerNorm> {
        Ok(LayerNorm::new(self.get(&format!("{p}.weight"), &[c])?, self.get(&format!("{p}.bias"), &[c])?, 1e-5))
    }
    pub(crate) fn conv(&self, p: &str, i: usize, o: usize, k: usize, transpose: bool) -> Result<Conv> {
        let shape = if transpose { [i, o, k, k] } else { [o, i, k, k] };
        Ok(Conv { weight: self.get(&format!("{p}.weight"), &shape)?, bias: self.get(&format!("{p}.bias"), &[o])?, transpose })
    }
}

/// Nonoverlapping stride-two transpose convolutions use GEMM and pixel rearrangement.
/// This avoids backend-specific convolution requirements, including cuDNN.
pub(crate) struct Conv {
    weight: Tensor,
    bias: Tensor,
    transpose: bool,
}
impl Conv {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = x.dims4()?;
        let o = self.bias.elem_count();
        let y = if self.transpose {
            let x = x.permute((0, 2, 3, 1))?.contiguous()?.flatten_to(2)?;
            x.matmul(&self.weight.flatten_from(1)?)?.reshape((b, h, w, o, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((b, o, h * 2, w * 2))?
        } else {
            let k = self.weight.dim(2)?;
            x.conv2d(&self.weight, 0, k, 1, 1)?
        };
        Ok(y.broadcast_add(&self.bias.reshape((1, o, 1, 1))?)?)
    }
}

pub(crate) fn channel_norm(x: &Tensor, norm: &LayerNorm) -> Result<Tensor> {
    Ok(norm.forward(&x.permute((0, 2, 3, 1))?.contiguous()?)?.permute((0, 3, 1, 2))?.contiguous()?)
}

pub(crate) struct PromptEncoder {
    matrix: Tensor,
    types: Vec<Tensor>,
    absent: Tensor,
    no_mask: Tensor,
    conv0: Conv,
    norm1: LayerNorm,
    conv3: Conv,
    norm4: LayerNorm,
    conv6: Conv,
}
impl PromptEncoder {
    pub(crate) fn load(l: &Loader<'_>) -> Result<Self> {
        let p = "tracker.model.interactive_sam_prompt_encoder";
        Ok(Self {
            matrix: l.get(&format!("{p}.pe_layer.positional_encoding_gaussian_matrix"), &[2, 128])?,
            types: (0..4).map(|i| l.get(&format!("{p}.point_embeddings.{i}.weight"), &[1, 256])).collect::<Result<_>>()?,
            absent: l.get(&format!("{p}.not_a_point_embed.weight"), &[1, 256])?,
            no_mask: l.get(&format!("{p}.no_mask_embed.weight"), &[1, 256])?,
            conv0: l.conv(&format!("{p}.mask_downscaling.0"), 1, 4, 2, false)?,
            norm1: l.norm(&format!("{p}.mask_downscaling.1"), 4)?,
            conv3: l.conv(&format!("{p}.mask_downscaling.3"), 4, 16, 2, false)?,
            norm4: l.norm(&format!("{p}.mask_downscaling.4"), 16)?,
            conv6: l.conv(&format!("{p}.mask_downscaling.6"), 16, 256, 1, false)?,
        })
    }
    fn fourier(&self, xy: &Tensor) -> Result<Tensor> {
        let phase = ((xy.affine(2.0, -1.0)? * std::f64::consts::TAU)?.matmul(&self.matrix))?;
        Ok(Tensor::cat(&[phase.sin()?, phase.cos()?], 1)?)
    }
    pub(crate) fn sparse(&self, points: &[[f32; 2]], labels: &[i32]) -> Result<Tensor> {
        ensure!(!points.is_empty() && points.len() == labels.len(), "points and labels must have equal nonzero lengths");
        ensure!(points.iter().flatten().all(|x| x.is_finite()), "point coordinates must be finite");
        ensure!(labels.iter().all(|l| (-1..=3).contains(l)), "labels must be -1, 0, 1, 2 or 3");
        let xy: Vec<f32> = points.iter().flatten().map(|v| (v + 0.5) / 1008.0).collect();
        let pe = self.fourier(&Tensor::from_vec(xy, (points.len(), 2), self.matrix.device())?)?;
        let rows = labels
            .iter()
            .enumerate()
            .map(|(i, &label)| -> Result<Tensor> {
                if label == -1 { Ok(self.absent.clone()) } else { Ok(pe.narrow(0, i, 1)?.broadcast_add(&self.types[label as usize])?) }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&rows, 0)?.unsqueeze(0)?)
    }
    pub(crate) fn dense(&self, mask: Option<&Tensor>) -> Result<Tensor> {
        if let Some(mask) = mask {
            ensure!(mask.dims() == [1, 1, 1008, 1008], "mask input must be [1,1,1008,1008] logits");
            let mask = resize_mask(&mask.to_dtype(DType::F32)?)?;

            let x = channel_norm(&self.conv0.forward(&mask)?, &self.norm1)?.gelu_erf()?;
            let x = channel_norm(&self.conv3.forward(&x)?, &self.norm4)?.gelu_erf()?;
            self.conv6.forward(&x)
        } else {
            Ok(self.no_mask.reshape((1, 256, 1, 1))?.broadcast_as((1, 256, 72, 72))?)
        }
    }
    pub(crate) fn image_pe(&self) -> Result<Tensor> {
        let xy: Vec<f32> = (0..72).flat_map(|y| (0..72).flat_map(move |x| [(x as f32 + 0.5) / 72.0, (y as f32 + 0.5) / 72.0])).collect();
        Ok(self.fourier(&Tensor::from_vec(xy, (72 * 72, 2), self.matrix.device())?)?.t()?.reshape((1, 256, 72, 72))?)
    }
}

/// Apply the same normalized triangle taps as ops::resize_bilinear_antialias on-device.
/// Separable matrix products keep feedback in f32 on CPU or CUDA without a host round trip.
fn resize_mask(mask: &Tensor) -> Result<Tensor> {
    let mut weights = vec![0_f32; 1008 * 288];
    let scale = 1008_f32 / 288.0;
    for d in 0..288 {
        let center = scale * (d as f32 + 0.5);
        let start = ((center - scale + 0.5) as isize).max(0) as usize;
        let end = ((center + scale + 0.5) as usize).min(1008);
        let taps: Vec<_> = (start..end).map(|s| (s, (1.0 - ((s as f32 - center + 0.5) * scale.recip()).abs()).max(0.0))).collect();
        let total: f32 = taps.iter().map(|(_, w)| w).sum();
        for (s, w) in taps {
            weights[s * 288 + d] = w / total;
        }
    }
    let weights = Tensor::from_vec(weights, (1008, 288), mask.device())?;
    let horizontal = mask.reshape((1008, 1008))?.matmul(&weights)?;
    Ok(weights.t()?.matmul(&horizontal)?.reshape((1, 1, 288, 288))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_resize_matches_cpu_reference() -> Result<()> {
        let values: Vec<f32> = (0..1008).flat_map(|y| (0..1008).map(move |x| (0.017 * y as f32 + 0.023 * x as f32).sin())).collect();
        let mask = Tensor::from_vec(values, (1, 1, 1008, 1008), &Device::Cpu)?;
        let actual = resize_mask(&mask)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = crate::ops::resize_bilinear_antialias(&mask, 288, 288)?.flatten_all()?.to_vec1::<f32>()?;
        let error = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
        assert!(error < 5e-7, "resize max absolute error {error}");
        Ok(())
    }
}
