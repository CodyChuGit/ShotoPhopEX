use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use gn_segment::{mask_decoder::InteractiveHeads, prompt::pack_prompts};
use std::path::{Path, PathBuf};

#[test]
fn worker_point_box_padding() {
    assert_eq!(pack_prompts(&[], None), (vec![[0.0; 2]; 2], vec![-1, -1]));
    assert_eq!(pack_prompts(&[([2.0, 3.0], 1)], None), (vec![[2.0, 3.0], [0.0; 2]], vec![1, -1]));
    assert_eq!(pack_prompts(&[([2.0, 3.0], 0)], None).1, vec![0, -1]);
    assert_eq!(pack_prompts(&[], Some([10.0, 20.0, 30.0, 40.0])), (vec![[0.0; 2], [10.0, 20.0], [30.0, 40.0]], vec![-1, 2, 3]));
    assert_eq!(pack_prompts(&[([2.0, 3.0], 1), ([5.0, 6.0], 0)], Some([10.0, 20.0, 30.0, 40.0])).1, vec![1, 0, 2, 3]);
}

fn compare(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|v| v.is_finite()));
    let abs = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
    let rel = actual.iter().zip(expected).map(|(a, b)| (a - b).abs() / b.abs().max(1e-6)).fold(0_f32, f32::max);
    let scale = expected.iter().map(|x| x.abs()).fold(0_f32, f32::max).max(1e-6);
    eprintln!("{name}: max_abs={abs:.8e}, max_relative(floor 1e-6)={rel:.8e}, relative_inf={:.8e}", abs / scale);
    assert!(abs < 1e-3, "{name}: maximum absolute error {abs}");
}

#[test]
fn interactive_heads_match_comfyui_f32() -> Result<()> {
    let path = std::env::var_os("GN_SAM31_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"));
    if !path.exists() {
        eprintln!("skipping interactive parity: checkpoint absent at {}", path.display());
        return Ok(());
    }
    let device = Device::Cpu;
    let model = InteractiveHeads::load(path, &device)?;
    let fixtures = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/interactive.safetensors"), &device)?;
    let get = |key: &str| fixtures.get(key).with_context(|| format!("missing fixture {key}"));
    // Exact fixture formula, evaluated in f32 with this grouping:
    // fpn_k[c,y,x] = 0.5*sin(((0.013*c + 0.071*y) + 0.031*x) + k).
    // k=2 is already the post-no-memory feature at the interactive heads' input.
    let mut fpn = Vec::new();
    for (k, size) in [288, 144, 72].into_iter().enumerate() {
        let values: Vec<f32> = (0..256)
            .flat_map(|c| {
                (0..size).flat_map(move |y| (0..size).map(move |x| 0.5 * (((0.013 * c as f32 + 0.071 * y as f32) + 0.031 * x as f32) + k as f32).sin()))
            })
            .collect();
        fpn.push(Tensor::from_vec(values, (1, 256, size, size), &device)?);
    }
    // Feedback is unthresholded 1008-space logits, not probabilities.
    let values: Vec<f32> = (0..1008).flat_map(|y| (0..1008).map(move |x| (0.017 * y as f32 + 0.023 * x as f32).sin())).collect();
    let mask = Tensor::from_vec(values, (1, 1, 1008, 1008), &device)?;
    for case in 0..3 {
        let coords = get(&format!("{case}.points"))?.flatten_all()?.to_vec1::<f32>()?;
        let points: Vec<[f32; 2]> = coords.as_chunks::<2>().0.to_vec();
        let labels: Vec<i32> = get(&format!("{case}.labels"))?.flatten_all()?.to_vec1()?;
        let output = model.forward([&fpn[0], &fpn[1], &fpn[2]], &points, &labels, if case == 2 { Some(&mask) } else { None })?;
        assert_eq!(output.masks.dims(), [1, 4, 288, 288]);
        assert_eq!(output.iou.dims(), [1, 4]);
        assert_eq!(output.object_score.dims(), [1, 1]);
        let expected = get(&format!("{case}.samples"))?.to_vec2::<f32>()?;
        let stats = get(&format!("{case}.stats"))?.to_vec2::<f32>()?;
        let masks = output.masks.reshape((4, 288 * 288))?.to_vec2::<f32>()?;
        for plane in 0..4 {
            let values = &masks[plane];
            let sampled: Vec<f32> = (0..4096).map(|i| values[i * (288 * 288 - 1) / 4095]).collect();
            compare(&format!("case {case} mask {plane}"), &sampled, &expected[plane]);
            let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
            let std = (values.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
            let actual =
                [mean as f32, std as f32, values.iter().copied().fold(f32::INFINITY, f32::min), values.iter().copied().fold(f32::NEG_INFINITY, f32::max)];
            compare(&format!("case {case} mask {plane} stats"), &actual, &stats[plane]);
        }
        compare(&format!("case {case} IoUs"), &output.iou.flatten_all()?.to_vec1::<f32>()?, &get(&format!("{case}.iou"))?.flatten_all()?.to_vec1::<f32>()?);
        compare(
            &format!("case {case} object"),
            &output.object_score.flatten_all()?.to_vec1::<f32>()?,
            &get(&format!("{case}.object"))?.flatten_all()?.to_vec1::<f32>()?,
        );
    }
    assert!(model.forward([&fpn[0], &fpn[1], &fpn[2]], &[[0.0; 2]], &[4], None).is_err());
    assert!(model.forward([&fpn[0], &fpn[1], &fpn[2]], &[], &[], None).is_err());
    Ok(())
}
