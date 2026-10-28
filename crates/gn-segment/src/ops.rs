//! CPU parity references and device implementations for SAM sampling and convolution.

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

fn cpu_f32(x: &Tensor) -> Result<()> {
    ensure!(x.dtype() == DType::F32 && matches!(x.device(), Device::Cpu), "expected a CPU f32 tensor");
    Ok(())
}

fn element_count(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |count, &dim| count.checked_mul(dim).ok_or_else(|| anyhow::anyhow!("output dimensions overflow")))
}

/// Pool feature-pixel boxes into 7 by 7 bins with torchvision's unaligned adaptive ROIAlign.
/// The single feature batch is shared by all boxes; output is `[boxes, C, 7, 7]`.
pub fn roi_align(features: &Tensor, boxes: &[[f32; 4]]) -> Result<Tensor> {
    if features.device().is_cuda() {
        return roi_align_device(features, boxes);
    }
    cpu_f32(features)?;
    let (n, c, h, w) = features.dims4()?;
    ensure!(n == 1 && c > 0 && h > 0 && w > 0, "ROIAlign requires nonempty (1,C,H,W) features");
    let src = features.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0_f32; element_count(&[boxes.len(), c, 49])?];
    for (r, &[x0, y0, x1, y1]) in boxes.iter().enumerate() {
        ensure!([x0, y0, x1, y1].iter().all(|v| v.is_finite()), "ROI coordinates must be finite");
        let rw = (x1 - x0).max(1.0);
        let rh = (y1 - y0).max(1.0);
        ensure!(rw.is_finite() && rh.is_finite(), "ROI extent overflows f32");
        let bw = rw / 7.0;
        let bh = rh / 7.0;
        let gx = bw.ceil() as usize;
        let gy = bh.ceil() as usize;
        ensure!(gx.checked_mul(gy).is_some(), "ROI sampling grid overflows");
        for channel in 0..c {
            let plane = &src[channel * h * w..(channel + 1) * h * w];
            for py in 0..7 {
                for px in 0..7 {
                    let mut sum = 0.0;
                    for iy in 0..gy {
                        let y = y0 + py as f32 * bh + (iy as f32 + 0.5) * bh / gy as f32;
                        for ix in 0..gx {
                            let x = x0 + px as f32 * bw + (ix as f32 + 0.5) * bw / gx as f32;
                            sum += bilinear_sample(plane, h, w, y, x);
                        }
                    }
                    out[(r * c + channel) * 49 + py * 7 + px] = sum / (gx * gy) as f32;
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (boxes.len(), c, 7, 7), &Device::Cpu)?)
}

fn bilinear_sample(src: &[f32], h: usize, w: usize, y: f32, x: f32) -> f32 {
    if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
        return 0.0;
    }
    let y = y.max(0.0).min((h - 1) as f32);
    let x = x.max(0.0).min((w - 1) as f32);
    let yl = y as usize;
    let xl = x as usize;
    let yh = (yl + 1).min(h - 1);
    let xh = (xl + 1).min(w - 1);
    let ly = y - yl as f32;
    let lx = x - xl as f32;
    (1.0 - ly) * (1.0 - lx) * src[yl * w + xl] + (1.0 - ly) * lx * src[yl * w + xh] + ly * (1.0 - lx) * src[yh * w + xl] + ly * lx * src[yh * w + xh]
}

/// Resize NCHW with PyTorch's antialiased bilinear filter and half-pixel centers.
/// Downsampling widens the triangle; both directions renormalize the truncated edge taps.
pub fn resize_bilinear_antialias(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    cpu_f32(x)?;
    let (n, c, h, w) = x.dims4()?;
    ensure!(n > 0 && c > 0 && h > 0 && w > 0 && out_h > 0 && out_w > 0, "resize dimensions must be positive");
    let src = x.flatten_all()?.to_vec1::<f32>()?;
    let horizontal_len = element_count(&[n, c, h, out_w])?;
    let output_len = element_count(&[n, c, out_h, out_w])?;
    let xs = resize_taps(w, out_w);
    let ys = resize_taps(h, out_h);
    let mut horizontal = vec![0.0; horizontal_len];
    for plane in 0..n * c {
        for y in 0..h {
            for (dx, taps) in xs.iter().enumerate() {
                let mut sum = 0.0;
                for &(sx, weight) in taps {
                    sum += src[(plane * h + y) * w + sx] * weight;
                }
                horizontal[(plane * h + y) * out_w + dx] = sum;
            }
        }
    }
    let mut out = vec![0.0; output_len];
    for plane in 0..n * c {
        for (dy, taps) in ys.iter().enumerate() {
            for dx in 0..out_w {
                let mut sum = 0.0;
                for &(sy, weight) in taps {
                    sum += horizontal[(plane * h + sy) * out_w + dx] * weight;
                }
                out[(plane * out_h + dy) * out_w + dx] = sum;
            }
        }
    }
    Ok(Tensor::from_vec(out, (n, c, out_h, out_w), &Device::Cpu)?)
}

fn resize_taps(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input as f32 / output as f32;
    let support = scale.max(1.0);
    let inv_scale = 1.0 / support;
    (0..output)
        .map(|d| {
            let center = scale * (d as f32 + 0.5);
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = (((center + support + 0.5) as isize).max(0) as usize).min(input);
            let mut taps: Vec<_> = (start..end).map(|s| (s, (1.0 - ((s as f32 - center + 0.5) * inv_scale).abs()).max(0.0))).collect();
            let total: f32 = taps.iter().map(|&(_, weight)| weight).sum();
            for (_, weight) in &mut taps {
                *weight /= total;
            }
            taps
        })
        .collect()
}

/// Apply affine LayerNorm over channels independently at each NCHW pixel.
/// Population variance and epsilon inside the square root match PyTorch LayerNorm.
pub fn channel_layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    for tensor in [x, weight, bias] {
        cpu_f32(tensor)?;
    }
    let (n, c, h, w) = x.dims4()?;
    ensure!(c > 0 && h > 0 && w > 0, "LayerNorm dimensions must be positive");
    ensure!(weight.dims() == [c] && bias.dims() == [c], "LayerNorm affine tensors must have shape [C]");
    ensure!(eps.is_finite() && eps >= 0.0, "LayerNorm epsilon must be finite and nonnegative");
    let src = x.flatten_all()?.to_vec1::<f32>()?;
    let weight = weight.to_vec1::<f32>()?;
    let bias = bias.to_vec1::<f32>()?;
    let mut out = vec![0.0; src.len()];
    for batch in 0..n {
        for pixel in 0..h * w {
            let index = |ch| (batch * c + ch) * h * w + pixel;
            let mean = (0..c).map(|ch| f64::from(src[index(ch)])).sum::<f64>() / c as f64;
            let var = (0..c).map(|ch| (f64::from(src[index(ch)]) - mean).powi(2)).sum::<f64>() / c as f64;
            let inv_std = (var + eps).sqrt().recip();
            for ch in 0..c {
                out[index(ch)] = (((f64::from(src[index(ch)]) - mean) * inv_std) as f32) * weight[ch] + bias[ch];
            }
        }
    }
    Ok(Tensor::from_vec(out, (n, c, h, w), &Device::Cpu)?)
}

/// Pool with bin weights on the feature device, avoiding a feature-map download.
pub(crate) fn roi_align_device(features: &Tensor, boxes: &[[f32; 4]]) -> Result<Tensor> {
    let (n, c, h, w) = features.dims4()?;
    ensure!(n == 1 && c > 0 && h > 0 && w > 0, "ROIAlign requires nonempty (1,C,H,W) features");
    let flat = features.to_dtype(DType::F32)?.reshape((c, h * w))?;
    let mut pooled = Vec::with_capacity(boxes.len());
    for &[x0, y0, x1, y1] in boxes {
        ensure!([x0, y0, x1, y1].iter().all(|v| v.is_finite()), "ROI coordinates must be finite");
        let bw = (x1 - x0).max(1.0) / 7.0;
        let bh = (y1 - y0).max(1.0) / 7.0;
        ensure!(bw.is_finite() && bh.is_finite(), "ROI extent overflows f32");
        let gx = bw.ceil() as usize;
        let gy = bh.ceil() as usize;
        let count = gx.checked_mul(gy).ok_or_else(|| anyhow::anyhow!("ROI sampling grid overflows"))?;
        let mut weights = vec![0_f32; element_count(&[h, w, 49])?];
        for py in 0..7 {
            for px in 0..7 {
                for iy in 0..gy {
                    let y = y0 + py as f32 * bh + (iy as f32 + 0.5) * bh / gy as f32;
                    for ix in 0..gx {
                        let x = x0 + px as f32 * bw + (ix as f32 + 0.5) * bw / gx as f32;
                        if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
                            continue;
                        }
                        let y = y.max(0.0).min((h - 1) as f32);
                        let x = x.max(0.0).min((w - 1) as f32);
                        let (yl, xl) = (y as usize, x as usize);
                        let (yh, xh) = ((yl + 1).min(h - 1), (xl + 1).min(w - 1));
                        let (ly, lx) = (y - yl as f32, x - xl as f32);
                        for (sy, sx, v) in [(yl, xl, (1.0 - ly) * (1.0 - lx)), (yl, xh, (1.0 - ly) * lx), (yh, xl, ly * (1.0 - lx)), (yh, xh, ly * lx)] {
                            weights[(sy * w + sx) * 49 + py * 7 + px] += v / count as f32;
                        }
                    }
                }
            }
        }
        let weights = Tensor::from_vec(weights, (h * w, 49), features.device())?;
        pooled.push(flat.matmul(&weights)?.reshape((1, c, 7, 7))?);
    }
    if pooled.is_empty() {
        return Ok(Tensor::zeros((0, c, 7, 7), features.dtype(), features.device())?);
    }
    Ok(Tensor::cat(&pooled, 0)?.to_dtype(features.dtype())?)
}

/// Append a constant channel kernel so convolution accumulates bias before rounding.
pub(crate) fn fuse_conv_bias(weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (o, _, kh, kw) = weight.dims4()?;
    let extra = bias.reshape((o, 1, 1, 1))?.pad_with_zeros(2, kh / 2, kh - 1 - kh / 2)?.pad_with_zeros(3, kw / 2, kw - 1 - kw / 2)?.pad_with_zeros(1, 0, 7)?;
    Ok(Tensor::cat(&[weight, &extra], 1)?)
}
pub(crate) fn bias_channels(x: &Tensor) -> Result<Tensor> {
    let (b, _, h, w) = x.dims4()?;
    Ok(Tensor::cat(&[x, &Tensor::ones((b, 8, h, w), x.dtype(), x.device())?], 1)?)
}

#[cfg(test)]
mod device_tests {
    use super::*;
    #[test]
    fn roi_bin_matrix_matches_adaptive_reference() -> Result<()> {
        let data: Vec<f32> = (0..3 * 13 * 16).map(|i| (i as f32 * 0.017).sin()).collect();
        let features = Tensor::from_vec(data, (1, 3, 13, 16), &Device::Cpu)?;
        let boxes = [[0.0, 0.0, 16.0, 13.0], [-2.0, -1.0, 7.3, 9.2], [4.2, 5.1, 4.3, 5.2], [15.0, 12.0, 18.0, 15.0]];
        let expected = roi_align(&features, &boxes)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = roi_align_device(&features, &boxes)?.flatten_all()?.to_vec1::<f32>()?;
        let error = expected.iter().zip(actual).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
        assert!(error < 1e-5, "ROI bin matrix error {error}");
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(0)?;
            let actual = roi_align_device(&features.to_device(&device)?, &boxes)?.flatten_all()?.to_vec1::<f32>()?;
            let error = expected.iter().zip(actual).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
            assert!(error < 1e-5, "CUDA ROI bin matrix error {error}");
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_conv_tests {
    use super::*;
    #[test]
    fn fused_convolution_bias_preserves_padding() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let x = Tensor::from_vec((0..4 * 7 * 9).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect::<Vec<_>>(), (1, 4, 7, 9), &device)?;
        let weight = Tensor::from_vec((0..3 * 4 * 3 * 3).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect::<Vec<_>>(), (3, 4, 3, 3), &device)?;
        let bias = Tensor::from_slice(&[0.125_f32, -0.25, 0.5], 3, &device)?;
        let expected = x
            .conv2d(&weight, 1, 1, 1, 1)?
            .broadcast_add(&bias.reshape((1, 3, 1, 1))?)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let x = x.to_dtype(DType::F16)?;
        let weight = fuse_conv_bias(&weight.to_dtype(DType::F16)?, &bias.to_dtype(DType::F16)?)?;
        let actual = bias_channels(&x)?.conv2d(&weight, 1, 1, 1, 1)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
