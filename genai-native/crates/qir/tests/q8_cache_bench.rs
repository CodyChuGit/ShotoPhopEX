//! How long the chat head (`lm_head.weight`, Q8_0 on the GPU) takes to load, quantized and from
//! its cache entry (a measurement, ignored by default): `QIR_Q8_BENCH=<text encoder
//! .safetensors> cargo test -p qwen-image-rs --features cuda --test q8_cache_bench -- --ignored
//! --nocapture` (default: ComfyUI's `qwen3vl_8b_bf16.safetensors`). Writes the entry where the
//! worker keeps it.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};

#[test]
#[ignore]
fn chat_head_loads() {
    let file = std::env::var_os("QIR_Q8_BENCH")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(&std::env::var_os("COMFY_MODELS").unwrap_or_default()).join("text_encoders/qwen3vl_8b_bf16.safetensors"));
    if !file.is_file() {
        return;
    }
    let (files, shape, dev) = (vec![file], (151_936, 4096), Device::new_cuda(0).unwrap());
    let mut blocks = Vec::new();
    for (label, cache) in
        [("quantized (QIR_Q8_CACHE=0)", "0"), ("quantized (QIR_Q8_CACHE=0)", "0"), ("cache (miss or hit)", "1"), ("cache hit", "1"), ("cache hit", "1")]
    {
        // The only test of its binary: nothing else reads the environment meanwhile.
        std::env::set_var("QIR_Q8_CACHE", cache);
        let t0 = Instant::now();
        let q = qwen_image_rs::q8::cached_q8_0(&files, "lm_head.weight", shape, DType::BF16, &dev).unwrap();
        dev.synchronize().unwrap();
        println!("{label:<28} {:5.2} s", t0.elapsed().as_secs_f64());
        blocks.push(q.data().unwrap().into_owned());
    }
    assert!(blocks.windows(2).all(|w| w[0] == w[1]), "the same blocks every way");
}
