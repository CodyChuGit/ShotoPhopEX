//! The hand-written safetensors readers in qir that open model files from the Models folder
//! (BYOM checkpoints and LoRAs anyone can download): the mapped reader, the ComfyUI INT8
//! reader, the ConvRot cache header and the LoRA loader. The input is written to a file first,
//! since they all read from a path.
#![no_main]

use libfuzzer_sys::fuzz_target;
use qwen_image_rs::{comfy_int8, convrot_cache, layered_files, lora};

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("qir-fuzz-{}.safetensors", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    if let Ok(f) = layered_files::TensorFile::open(&path) {
        for name in f.names() {
            let _ = f.tensor(&name, &candle_core::Device::Cpu);
        }
    }
    if let Ok(mut f) = comfy_int8::Int8File::open(&path) {
        let modules: Vec<String> = f.modules().keys().cloned().collect();
        for m in modules {
            let _ = f.load(&m);
        }
    }
    let _ = convrot_cache::read_header(&path);
    let _ = lora::LoraFile::load(&path, &candle_core::Device::Cpu, candle_core::DType::F32);
});
