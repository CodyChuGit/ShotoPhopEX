//! Builds the CUDA kernels behind the `convrot`, `fusednorm`, `sage` and `sage2` features.
//!
//! Needs nvcc (CUDA 12.8+ for sm_120; 13.x works) and, for `convrot`, a fresh CUTLASS: set
//! `CUTLASS_DIR` to its `include/` dir (the flash-attn-vendored CUTLASS has a `matrix.h` bug
//! CUDA 13.3 rejects). `CUDA_COMPUTE_CAP` picks the architecture (`120` for the RTX 5090, `89`
//! for Ada; default 89). On Windows the host compiler is MSVC 2022 found through `vcvars64.bat`
//! (put the VS Installer folder on PATH first so vswhere resolves), and CUDA 13's headers need
//! the conforming preprocessor, C++17 and the dynamic CRT, which this script passes; the same
//! three flags must reach candle-kernels' own nvcc calls through
//! `NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor -Xcompiler /MD -std=c++17`, and cudarc 0.19
//! wants `CUDARC_CUDA_VERSION=13030` for a 13.4 toolkit. The recipe is in PhotoshopEX's
//! `docs/comfy/dev-environment-windows.md`.

#[cfg(any(feature = "convrot", feature = "fusednorm", feature = "sage", feature = "sage2"))]
fn nvcc() -> cc::Build {
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
    let mut b = cc::Build::new();
    b.cuda(true).flag("-std=c++17").flag("--expt-relaxed-constexpr").flag(format!("-arch=sm_{cap}"));
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // CUDA 13's CCCL under MSVC: the standard-conforming preprocessor; and the CRT the Rust
        // target links (static with `+crt-static`, dynamic otherwise) so every object agrees.
        let crt_static = std::env::var("CARGO_CFG_TARGET_FEATURE").map(|f| f.split(',').any(|x| x == "crt-static")).unwrap_or(false);
        b.flag("-Xcompiler").flag("/Zc:preprocessor").flag("-Xcompiler").flag(if crt_static { "/MT" } else { "/MD" });
    }
    b
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/convrot/int8_gemm.cu");
    println!("cargo:rerun-if-changed=kernels/convrot/quant_ops.cu");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    #[cfg(feature = "convrot")]
    {
        let cutlass = std::env::var("CUTLASS_DIR").unwrap_or_else(|_| {
            format!(
                "{}/dev_tmp/cutlass/include",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        nvcc()
            .flag("--expt-extended-lambda") // EVT epilogue visitors use device lambdas
            .include(&cutlass)
            .file("kernels/convrot/int8_gemm.cu")
            .file("kernels/convrot/quant_ops.cu")
            .compile("convrot_int8");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "fusednorm")]
    {
        // Fused LayerNorm+AdaLN kernel (no external deps).
        println!("cargo:rerun-if-changed=kernels/fusednorm/fused_norm.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/vae_norm.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/vae_nhwc.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/head_rmsnorm.cuh");
        nvcc()
            .file("kernels/fusednorm/fused_norm.cu")
            .file("kernels/fusednorm/vae_norm.cu")
            .file("kernels/fusednorm/vae_nhwc.cu")
            .compile("fused_norm");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "sage")]
    {
        // Vendored SageAttention INT8-QK / FP16-PV kernel (thu-ml), torch-free,
        // plus the BSHD interleaved-RoPE kernel used only by the sage path.
        println!("cargo:rerun-if-changed=kernels/sage/sage_ffi.cu");
        println!("cargo:rerun-if-changed=kernels/sage/rope_bshd.cu");
        println!("cargo:rerun-if-changed=kernels/sage/rope_pair.cuh");
        nvcc()
            .flag("--expt-extended-lambda")
            .flag("-diag-suppress=177") // unused CHECK_ macros in vendored utils
            .include("kernels/sage")
            .file("kernels/sage/sage_ffi.cu")
            .file("kernels/sage/rope_bshd.cu")
            .compile("sage_attn");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "sage2")]
    {
        // SageAttention2 sm89 (INT8-QK per-thread + FP8-PV): its own TU so the
        // vendored sm80/sm89 kernels' PACK_SIZE_* macros never meet.
        println!("cargo:rerun-if-changed=kernels/sage/sage2_ffi.cu");
        println!("cargo:rerun-if-changed=kernels/fusednorm/head_rmsnorm.cuh");
        println!("cargo:rerun-if-changed=kernels/sage/vendor/qattn/qk_int_sv_f8_sm89.cuh");
        nvcc()
            .flag("--expt-extended-lambda")
            .flag("-diag-suppress=177")
            .include("kernels/sage")
            .file("kernels/sage/sage2_ffi.cu")
            .compile("sage2_attn");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    #[cfg(feature = "cudnn")]
    {
        // cudarc links `-lcudnn` (candle enables its `dynamic-linking`). cuDNN is
        // not a system package on the host: it comes from the NVIDIA pip wheel
        // `nvidia-cudnn-cu13` unpacked in user space (scripts/setup-host.sh), so
        // point the linker at it and bake an rpath so the binary runs without
        // LD_LIBRARY_PATH. Override the dir with QIR_CUDNN_LIB. (Windows: the
        // import library lives in NVIDIA's cuDNN zip, not in the wheel.)
        println!("cargo:rerun-if-env-changed=QIR_CUDNN_LIB");
        let lib = std::env::var("QIR_CUDNN_LIB").unwrap_or_else(|_| {
            format!(
                "{}/dev_tmp/cudnn/py/nvidia/cudnn/lib",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        println!("cargo:rustc-link-search=native={lib}");
        if !cfg!(target_os = "windows") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        }
    }
}
