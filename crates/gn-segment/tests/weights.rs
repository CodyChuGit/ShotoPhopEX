use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use gn_segment::weights::{expected_shapes, split_qkv, validate};
use std::{fs, path::Path};

#[test]
fn qkv_splits_weights_and_biases_in_order() -> Result<()> {
    for shape in [vec![6], vec![6, 2]] {
        let data: Vec<f32> = (0..shape.iter().product::<usize>()).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data.clone(), shape, &Device::Cpu)?.to_dtype(DType::F16)?;
        let (q, k, v) = split_qkv(&x)?;
        for (i, part) in [q, k, v].into_iter().enumerate() {
            assert_eq!(part.dtype(), DType::F16);
            assert_eq!(part.dim(0)?, 2);
            let actual = part.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let width = data.len() / 3;
            assert_eq!(actual, data[i * width..(i + 1) * width]);
        }
    }
    for shape in [vec![], vec![0], vec![5], vec![3, 2, 2]] {
        assert!(split_qkv(&Tensor::zeros(shape, DType::F32, &Device::Cpu)?).is_err());
    }
    Ok(())
}

#[test]
fn validates_real_checkpoint_when_present() -> Result<()> {
    let path = Path::new(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors");
    if !path.exists() {
        return Ok(());
    }
    let report = validate(path)?;
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    assert!(report.mismatches.is_empty(), "{:?}", report.mismatches);
    assert_eq!(report.checked, expected_shapes().len());
    Ok(())
}

#[test]
fn reports_missing_and_mismatched_shapes_without_tensor_data() -> Result<()> {
    let shapes = expected_shapes();
    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for (key, shape) in &shapes {
        let end = offset + 2 * shape.iter().product::<usize>();
        header.insert(key.clone(), serde_json::json!({"dtype": "F16", "shape": shape, "data_offsets": [offset, end]}));
        offset = end;
    }
    let path = std::env::temp_dir().join(format!("gn-segment-header-{}.safetensors", std::process::id()));
    let write_header = |header: &serde_json::Map<String, serde_json::Value>| -> Result<()> {
        let mut json = serde_json::to_vec(header)?;
        json.resize(json.len().next_multiple_of(8), b' ');
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        // No tensor data is written, so successful validation proves the read is header-only.
        fs::write(&path, bytes)?;
        Ok(())
    };
    write_header(&header)?;
    let clean = validate(&path)?;
    assert_eq!(clean.checked, shapes.len());
    assert!(clean.missing.is_empty() && clean.mismatches.is_empty());
    let missing = "detector.backbone.vision_backbone.trunk.pos_embed";
    let wrong = "tracker.model.interactivity_no_mem_embed";
    header.remove(missing);
    header[wrong]["shape"] = serde_json::json!([1, 1, 128]);
    write_header(&header)?;
    let report = validate(&path)?;
    assert_eq!(report.missing, [missing]);
    assert_eq!(report.mismatches.len(), 1);
    assert_eq!(report.mismatches[0].key, wrong);
    assert_eq!(report.mismatches[0].expected, [1, 1, 256]);
    assert_eq!(report.mismatches[0].actual, [1, 1, 128]);
    fs::write(&path, 100_000_001_u64.to_le_bytes())?;
    assert!(validate(&path).is_err());
    fs::write(&path, 16_u64.to_le_bytes())?;
    assert!(validate(&path).is_err());
    fs::write(&path, [0_u8; 3])?;
    assert!(validate(&path).is_err());
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn shape_table_covers_complete_selected_modules() {
    let shapes = expected_shapes();
    let groups = [
        ("detector.backbone.vision_backbone.trunk.", 388),
        ("detector.backbone.vision_backbone.convs.", 18),
        ("detector.backbone.vision_backbone.interactive_convs.", 18),
        ("detector.backbone.language_backbone.", 294),
        ("detector.geometry_encoder.", 76),
        ("detector.transformer.", 283),
        ("detector.dot_prod_scoring.", 10),
        ("detector.segmentation_head.", 28),
        ("tracker.model.interactive_sam_prompt_encoder.", 17),
        ("tracker.model.interactive_sam_mask_decoder.", 131),
        ("tracker.model.interactivity_no_mem_embed", 1),
    ];
    assert_eq!(shapes.len(), 1264);
    for (prefix, count) in groups {
        assert_eq!(shapes.keys().filter(|key| key.starts_with(prefix)).count(), count, "{prefix}");
    }
}
