//! How fast a big safetensors file reaches the GPU, by route (a measurement, ignored by default):
//! `QIR_LOAD_BENCH=<file.safetensors> cargo test -p qwen-image-rs --release --features cuda
//! --test load_bench -- --ignored --nocapture`.

use std::time::Instant;

use candle_core::{DType, Device};
use qwen_image_rs::layered_files::TensorFile;

fn names(path: &std::path::Path) -> Vec<String> {
    let file = TensorFile::open(path).unwrap();
    file.names()
}

fn report(label: &str, t0: Instant, bytes: u64) {
    let s = t0.elapsed().as_secs_f64();
    println!("{label:<44} {s:6.2} s  {:5.2} GB/s", bytes as f64 / 1e9 / s);
}

#[test]
#[ignore]
fn upload_routes() {
    let Some(path) = std::env::var_os("QIR_LOAD_BENCH").map(std::path::PathBuf::from) else { return };
    let dev = Device::new_cuda(0).unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();
    let names = names(&path);
    for round in 0..2 {
        // A: candle's mapped VarBuilder (a CPU copy of each tensor, then a pageable upload).
        let t0 = Instant::now();
        {
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&path], DType::BF16, &dev).unwrap() };
            let held: Vec<_> = names.iter().map(|n| vb.get_unchecked(n).unwrap()).collect();
            dev.synchronize().unwrap();
            drop(held);
        }
        report(&format!("[{round}] A candle mmap VarBuilder"), t0, bytes);
        // B: straight from the mapping to the device, tensor by tensor.
        let t0 = Instant::now();
        {
            let file = TensorFile::open(&path).unwrap();
            let held: Vec<_> = names.iter().map(|n| file.tensor(n, &dev).unwrap()).collect();
            dev.synchronize().unwrap();
            drop(held);
        }
        report(&format!("[{round}] B TensorFile from the mapping"), t0, bytes);
        // C: B with the mapping faulted in first by several threads.
        let t0 = Instant::now();
        {
            let file = TensorFile::open(&path).unwrap();
            file.prefault_parallel(8);
            let faulted = t0.elapsed().as_secs_f64();
            let held: Vec<_> = names.iter().map(|n| file.tensor(n, &dev).unwrap()).collect();
            dev.synchronize().unwrap();
            drop(held);
            println!("    (prefault {faulted:.2} s)");
        }
        report(&format!("[{round}] C TensorFile, prefaulted on 8 threads"), t0, bytes);
    }
}
