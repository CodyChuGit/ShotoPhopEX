use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use gn_segment::text::TextEncoder;

fn checkpoint() -> PathBuf {
    std::env::var_os("GN_SAM31_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"))
}

#[test]
fn clip_and_resizer_match_comfyui_f32() -> Result<()> {
    let path = checkpoint();
    if !path.exists() {
        eprintln!("skipping text parity: checkpoint absent at {}", path.display());
        return Ok(());
    }
    let device = Device::Cpu;
    let encoder = TextEncoder::load(path, &device)?;
    let fixtures = candle_core::safetensors::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text.safetensors"), &device)?;
    let get = |key: &str| fixtures.get(key).with_context(|| format!("missing fixture {key}"));
    for (index, prompt) in ["the cup", "a red umbrella", "person", "full context"].iter().enumerate() {
        let ids: [u32; 32] = get(&format!("{index}.ids"))?.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?.try_into().ok().context("ids shape")?;
        let mask: [bool; 32] = get(&format!("{index}.mask"))?
            .flatten_all()?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|v| v != 0)
            .collect::<Vec<_>>()
            .try_into()
            .ok()
            .context("mask shape")?;
        assert_eq!(ids[0], 49406);
        let valid = mask.iter().filter(|&&v| v).count();
        assert_eq!(valid, [4, 5, 3, 32][index]);
        assert_eq!(ids[valid - 1], 49407);
        assert!(mask[..valid].iter().all(|&v| v));
        assert!(ids[valid..].iter().all(|&id| id == 0));
        let (actual, returned_mask) = encoder.encode(&ids, &mask)?;
        assert_eq!(returned_mask, mask);
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(actual.dims(), [1, 32, 256]);
        let expected = get(&format!("{index}.features"))?;
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert!(actual.iter().all(|v| v.is_finite()));
        let max_abs = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0_f32, f32::max);
        let max_relative = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs() / b.abs().max(1e-6)).fold(0_f32, f32::max);
        let scale = expected.iter().map(|v| v.abs()).fold(0_f32, f32::max);
        eprintln!("{prompt}: max_abs={max_abs:.8e}, max_relative (floor 1e-6)={max_relative:.8e}, relative_inf={:.8e}", max_abs / scale);
        assert!(max_abs < 1e-3, "{prompt}: max absolute error {max_abs}");
    }
    assert!(encoder.encode(&[49408; 32], &[true; 32]).is_err());
    Ok(())
}
