use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

const CUTILE_FEATURE: &str = "CARGO_FEATURE_CUTILE";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_sources = vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
        "src/mmvq_gguf.cu",
        "src/mmq_gguf/mmq_quantize.cu",
        "src/mmq_gguf/mmq_instance_q4_0.cu",
        "src/mmq_gguf/mmq_instance_q4_1.cu",
        "src/mmq_gguf/mmq_instance_q5_0.cu",
        "src/mmq_gguf/mmq_instance_q5_1.cu",
        "src/mmq_gguf/mmq_instance_q8_0.cu",
        "src/mmq_gguf/mmq_instance_q2_k.cu",
        "src/mmq_gguf/mmq_instance_q3_k.cu",
        "src/mmq_gguf/mmq_instance_q4_k.cu",
        "src/mmq_gguf/mmq_instance_q5_k.cu",
        "src/mmq_gguf/mmq_instance_q6_k.cu",
    ];
    if env::var_os(CUTILE_FEATURE).is_some() {
        moe_sources.push("src/moe/moe_align.cu");
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(moe_sources)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    // ShotoPhopEX: cudaforge compiles this static library for one architecture only
    // (CUDA_COMPUTE_CAP), with no PTX, so a release built for sm_86 had no GGUF kernels an
    // RTX 50 card (sm_120) can load: the lean Q8_0 text encoder failed there with
    // CUDA_ERROR_NO_BINARY_FOR_GPU. Add a cubin for every other architecture of
    // QIR_CUDA_ARCHS (the list qir's own kernels are built for) and the newest one's PTX.
    println!("cargo::rerun-if-env-changed=QIR_CUDA_ARCHS");
    if let Ok(list) = env::var("QIR_CUDA_ARCHS") {
        let base = env::var("CUDA_COMPUTE_CAP").ok().and_then(|c| c.trim().parse::<u32>().ok());
        let mut archs: Vec<u32> = list.split([',', ';', ' ']).filter_map(|a| a.trim().parse().ok()).collect();
        archs.sort_unstable();
        archs.dedup();
        for cap in archs.iter().filter(|c| Some(**c) != base) {
            moe_builder = moe_builder.arg(&format!("-gencode=arch=compute_{cap},code=sm_{cap}"));
        }
        if let Some(newest) = archs.iter().max() {
            moe_builder = moe_builder.arg(&format!("-gencode=arch=compute_{newest},code=compute_{newest}"));
        }
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
