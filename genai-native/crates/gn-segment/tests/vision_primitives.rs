use anyhow::Result;
use candle_core::{Device, Tensor};
use gn_segment::{
    attention::attention,
    position::{Rope2d, image_position},
    vision::{transpose_conv2, window_partition, window_unpartition},
};

#[test]
fn windows_preserve_order_and_round_trip_padding() -> Result<()> {
    for (b, h, w, c, size) in [(2, 5, 7, 3, 4), (1, 6, 6, 2, 3), (1, 1, 2, 1, 4)] {
        let values: Vec<f32> = (0..b * h * w * c).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&values, (b, h, w, c), &Device::Cpu)?;
        let (windows, padded) = window_partition(&x, size)?;
        let restored = window_unpartition(&windows, size, padded, (h, w))?;
        assert_eq!(values, restored.flatten_all()?.to_vec1::<f32>()?);
        let flat = windows.flatten_all()?.to_vec1::<f32>()?;
        for batch in 0..b {
            for y in 0..padded.0 {
                for x in 0..padded.1 {
                    for ch in 0..c {
                        let win = (batch * (padded.0 / size) + y / size) * (padded.1 / size) + x / size;
                        let index = ((win * size + y % size) * size + x % size) * c + ch;
                        let expected = if y < h && x < w { values[((batch * h + y) * w + x) * c + ch] } else { 0.0 };
                        assert_eq!(flat[index], expected);
                    }
                }
            }
        }
    }
    assert!(window_partition(&Tensor::zeros((1, 2, 2, 1), candle_core::DType::F32, &Device::Cpu)?, 0).is_err());
    Ok(())
}

#[test]
fn axial_rope_matches_formula_and_interleaved_rotation() -> Result<()> {
    for scale in [1.0, 24.0 / 72.0] {
        let rope = Rope2d::new(3, 2, 64, scale, &Device::Cpu)?;
        let cos = rope.cos.to_vec2::<f32>()?;
        let sin = rope.sin.to_vec2::<f32>()?;
        for t in 0..6 {
            for pair in 0..32 {
                let coord = if pair < 16 { t % 3 } else { t / 3 };
                let angle = (coord as f32 * scale) as f64 * 10000_f64.powf(-2.0 * (pair % 16) as f64 / 32.0);
                assert!((cos[t][pair] - angle.cos() as f32).abs() < 1e-7);
                assert!((sin[t][pair] - angle.sin() as f32).abs() < 1e-7);
            }
        }
        let x = Tensor::from_vec((0..384).map(|i| i as f32 / 100.0).collect(), (1, 1, 6, 64), &Device::Cpu)?;
        let rotated = rope.apply(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let src = x.flatten_all()?.to_vec1::<f32>()?;
        for t in 0..6 {
            for p in 0..32 {
                let i = t * 64 + 2 * p;
                assert!((rotated[i] - (src[i] * cos[t][p] - src[i + 1] * sin[t][p])).abs() < 1e-6);
                assert!((rotated[i + 1] - (src[i] * sin[t][p] + src[i + 1] * cos[t][p])).abs() < 1e-6);
            }
        }
    }
    Ok(())
}

#[test]
fn detr_positions_use_y_then_x_and_pixel_endpoints() -> Result<()> {
    let pos = image_position(3, 4, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    for ch in 0..256 {
        for y in 0..3 {
            for x in 0..4 {
                let (index, size) = if ch < 128 { (y, 3) } else { (x, 4) };
                let phase = index as f64 / (size as f64 - 1.0 + 1e-6) * std::f64::consts::TAU / 10000_f64.powf((2 * ((ch % 128) / 2)) as f64 / 128.0);
                let expected = if ch % 2 == 0 { phase.sin() } else { phase.cos() };
                assert!((pos[(ch * 3 + y) * 4 + x] as f64 - expected).abs() < 7e-7);
            }
        }
    }
    assert_eq!(image_position(1, 1, &Device::Cpu)?.dims(), [1, 256, 1, 1]);
    Ok(())
}

#[test]
fn chunked_attention_matches_naive_masks_bias_and_all_masked_rows() -> Result<()> {
    let (b, h, nq, nk, d) = (2, 2, 3, 4, 2);
    let q: Vec<f32> = (0..b * h * nq * d).map(|i| (i as f32 - 7.0) / 13.0).collect();
    let k: Vec<f32> = (0..b * h * nk * d).map(|i| (i as f32 % 7.0 - 3.0) / 5.0).collect();
    let v: Vec<f32> = (0..b * h * nk * d).map(|i| (i as f32 % 11.0) / 7.0).collect();
    let qt = Tensor::from_slice(&q, (b, h, nq, d), &Device::Cpu)?;
    let kt = Tensor::from_slice(&k, (b, h, nk, d), &Device::Cpu)?;
    let vt = Tensor::from_slice(&v, (b, h, nk, d), &Device::Cpu)?;
    // First batch has no valid keys, second has a masked interior key.
    let mask = [0u8, 0, 0, 0, 1, 0, 1, 1];
    let mt = Tensor::from_slice(&mask, (b, nk), &Device::Cpu)?;
    let bias: Vec<f32> = (0..h * nq * nk).map(|i| i as f32 / 17.0 - 0.5).collect();
    let bt = Tensor::from_slice(&bias, (1, h, nq, nk), &Device::Cpu)?;
    for padding in [false, true] {
        for causal in [false, true] {
            for add_bias in [false, true] {
                let mut expected = vec![0.0; b * h * nq * d];
                for batch in 0..b {
                    for head in 0..h {
                        for row in 0..nq {
                            let mut scores = vec![f64::NEG_INFINITY; nk];
                            for key in 0..nk {
                                if (padding && mask[batch * nk + key] == 0) || (causal && key > row) {
                                    continue;
                                }
                                let dot = (0..d)
                                    .map(|i| q[((batch * h + head) * nq + row) * d + i] as f64 * k[((batch * h + head) * nk + key) * d + i] as f64)
                                    .sum::<f64>();
                                scores[key] = dot / (d as f64).sqrt() + if add_bias { bias[(head * nq + row) * nk + key] as f64 } else { 0.0 };
                            }
                            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            if !max.is_finite() {
                                continue;
                            }
                            let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                            let sum: f64 = exp.iter().sum();
                            for i in 0..d {
                                expected[((batch * h + head) * nq + row) * d + i] =
                                    (0..nk).map(|key| exp[key] / sum * v[((batch * h + head) * nk + key) * d + i] as f64).sum::<f64>() as f32;
                            }
                        }
                    }
                }
                for chunk in [1, 2, 99] {
                    let actual = attention(&qt, &kt, &vt, if padding { Some(&mt) } else { None }, causal, if add_bias { Some(&bt) } else { None }, chunk)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!((a - e).abs() < 3e-7, "{a} vs {e}, padding={padding} causal={causal} bias={add_bias}");
                    }
                }
            }
        }
    }
    assert!(attention(&qt, &kt, &vt, None, false, None, 0).is_err());
    Ok(())
}

#[test]
fn transpose_conv_is_nonoverlapping_and_uses_input_output_kernel_order() -> Result<()> {
    let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (1, 2, 1, 2), &Device::Cpu)?;
    let weights = Tensor::from_slice(&[1f32, 2., 3., 4., 10., 20., 30., 40.], (2, 1, 2, 2), &Device::Cpu)?;
    let bias = Tensor::from_slice(&[0.5f32], 1, &Device::Cpu)?;
    let out = transpose_conv2(&x, &weights, &bias)?;
    assert_eq!(out.dims(), [1, 1, 2, 4]);
    assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, vec![31.5, 62.5, 42.5, 84.5, 93.5, 124.5, 126.5, 168.5]);
    Ok(())
}

#[test]
fn attention_hand_checked_uniform_causal_and_additive_weights() -> Result<()> {
    let device = Device::Cpu;
    let q = Tensor::zeros((1, 1, 2, 2), candle_core::DType::F32, &device)?;
    let v = Tensor::from_slice(&[1f32, 2., 3., 4.], (1, 1, 2, 2), &device)?;
    let uniform = attention(&q, &q, &v, None, false, None, 1)?;
    assert_eq!(uniform.flatten_all()?.to_vec1::<f32>()?, [2., 3., 2., 3.]);
    let causal = attention(&q, &q, &v, None, true, None, 1)?;
    assert_eq!(causal.flatten_all()?.to_vec1::<f32>()?, [1., 2., 2., 3.]);
    let bias = Tensor::from_slice(&[0f32, 3f32.ln()], (1, 2), &device)?;
    let weighted = attention(&q, &q, &v, None, false, Some(&bias), 1)?;
    for (a, expected) in weighted.flatten_all()?.to_vec1::<f32>()?.iter().zip([2.5, 3.5, 2.5, 3.5]) {
        assert!((a - expected).abs() < 3e-7);
    }
    let bias = Tensor::full(f32::NEG_INFINITY, (2, 2), &device)?;
    let masked = attention(&q, &q, &v, None, false, Some(&bias), 1)?;
    assert_eq!(masked.flatten_all()?.to_vec1::<f32>()?, [0.; 4]);
    Ok(())
}
