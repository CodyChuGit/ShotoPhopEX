//! How long MOSS-Image's files take to reach the GPU, and that the INT8 bytes are the ones a plain
//! read of the file gives (both ignored by default, they need the installed files):
//! `MOSS_ROOT=<folder> cargo test -p qwen-image-rs --features cuda,convrot --test
//! moss_load_bench -- --ignored --nocapture`.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};
use qwen_image_rs::comfy_int8;
use qwen_image_rs::moss::dit::{MossConfig, MossDit};
use qwen_image_rs::moss::text::{Qwen35Text, TextConfig};
use qwen_image_rs::moss::weights::Weights;

fn files() -> Option<[PathBuf; 2]> {
    let root = PathBuf::from(std::env::var("MOSS_ROOT").unwrap_or_default());
    let files = [root.join("text_encoders/qwen3.5_9b_int8_convrot.safetensors"), root.join("diffusion_models/moss_image_sft_int8_convrot.safetensors")];
    files.iter().all(|f| f.is_file()).then_some(files)
}

#[test]
#[ignore]
fn moss_loads() {
    let Some([text, dit]) = files() else { return };
    let dev = Device::new_cuda(0).unwrap();
    for round in 0..3 {
        let t0 = Instant::now();
        let w = Weights::open(&text, &dev, DType::BF16, true).unwrap();
        let te = Qwen35Text::load(&TextConfig::qwen35_9b(), &w).unwrap();
        dev.synchronize().unwrap();
        let te_s = t0.elapsed().as_secs_f64();
        drop((te, w));
        let t1 = Instant::now();
        let w = Weights::open(&dit, &dev, DType::BF16, true).unwrap();
        let d = MossDit::load(&MossConfig::default(), &w).unwrap();
        dev.synchronize().unwrap();
        let dit_s = t1.elapsed().as_secs_f64();
        drop((d, w));
        println!("[{round}] text encoder {te_s:5.2} s   DiT {dit_s:5.2} s   total {:5.2} s", te_s + dit_s);
    }
}

/// Every 256-group INT8 module of both files, through [`Weights`], against the file read with
/// `seek` + `read_exact` and [`comfy_int8::convert`] (what the loader did before the mapping).
#[test]
#[ignore]
fn the_mapped_int8_bytes_are_the_files() {
    let Some(files) = files() else { return };
    for path in files {
        let w = Weights::open(&path, &Device::Cpu, DType::F32, false).unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).unwrap();
        let mut header = vec![0; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut header).unwrap();
        let start = 8 + header.len() as u64;
        let meta: safetensors::tensor::Metadata = serde_json::from_slice(&header).unwrap();
        let mut read = |name: &str| {
            let (a, b) = meta.info(name).unwrap().data_offsets;
            file.seek(SeekFrom::Start(start + a as u64)).unwrap();
            let mut bytes = vec![0; b - a];
            file.read_exact(&mut bytes).unwrap();
            bytes
        };
        let modules: Vec<String> = meta.tensors().keys().filter_map(|n| n.strip_suffix(".comfy_quant").map(str::to_owned)).collect();
        let mut checked = 0;
        for m in modules {
            let descriptor = read(&format!("{m}.comfy_quant"));
            if serde_json::from_slice::<serde_json::Value>(&descriptor).unwrap()["convrot_groupsize"] != 256 {
                continue;
            }
            let shape = meta.info(&format!("{m}.weight")).unwrap().shape.clone();
            let scale: Vec<f32> = read(&format!("{m}.weight_scale")).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            let want = comfy_int8::convert(&descriptor, &shape, read(&format!("{m}.weight")), &[shape[0], 1], scale).unwrap();
            let got = w.prequantized(&m).unwrap();
            assert!(got.weight_i8 == want.weight_i8 && got.col_scale == want.col_scale, "{m}");
            checked += 1;
        }
        println!("{}: {checked} INT8 modules identical", path.display());
        assert!(checked > 0);
    }
}
