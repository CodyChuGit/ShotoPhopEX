# Pure-Rust / Rust-first inference runtimes for 12–20B diffusion transformers on an RTX 5090 (state as of 2026-10-16)

Scope: candle (HuggingFace), burn + CubeCL/CubeK (Tracel), diffusion-rs (EricLBuehler), mistral.rs's diffusion path, ort (pykeio), plus every Rust project found that actually runs Qwen-Image / Qwen-Image-Edit / FLUX-class DiTs. Everything older than mid-2025 is flagged "(pre-mid-2025)". Numbers are quoted with their source; where a source is a single self-reported README I say so.

Quick orientation table (details and citations in the sections below):

| Runtime | Latest version / date | Licence | Runs a Qwen-Image / FLUX-class DiT today? | bf16 | fp8 matmul | NVFP4 | Flash attn on sm_120 | Runtime LoRA | Build needs nvcc? |
|---|---|---|---|---|---|---|---|---|---|
| candle (`candle-core`) | 0.11.0, 2026-06-26 | MIT OR Apache-2.0 | Yes, via third-party projects (qwen-image-rs, qwen3-image21, mold, SceneWorks) and candle-transformers FLUX example | Yes | No (F8E4M3 dtype exists; CUDA kernels unreachable, issue #3886 open) | Not upstream; SceneWorks fork via cuBLASLt W4A4, LLM only | `candle-flash-attn` (FA2) compiles with `CUDA_COMPUTE_CAP`; FA4 bindings experimental, SM120 passed kernel tests | Not upstream for diffusion; `candle-lora` is training-oriented, "not compatible with peft" | Yes for kernels (`candle-kernels` compiles .cu at build; flash-attn ~9 min build) |
| burn + CubeCL/CubeK | burn 0.22.0, 2026-10-06; cubecl 0.11.0-pre.3, 2026-08-25 | MIT OR Apache-2.0 | No DiT port found | Yes | Not stated | Not stated (packed `e2m1x2` store only) | Flash attention kernel exists in CubeK but burn issue #4325 "flash attention fails on most backends" is OPEN, labelled blocker (2026-09-24) | LoRA/QLoRA in 0.22 (training-side) | No (LLVM→PTX, prebuilt LLVM fetched, PTX executed via cudarc) |
| diffusion-rs (EricLBuehler) | crate 0.1.0 only; repo 84 commits, no releases | MIT | FLUX.1 Dev/Schnell only | Yes | No | No | Not mentioned | "Upcoming" | candle-based (yes) |
| mistral.rs diffusion | mistralrs 0.8.1 (2026-04-02); GitHub v0.9.4 (24 Sep 2026) | MIT | FLUX.1 schnell/dev only | Yes (native precision only; ISQ refused for diffusion) | FP8 for LLMs only (0.9.3) | NVFP4 for LLMs only (0.9.3, cuTile) | FAv3 for LLMs; sm_120 + CUDA 13 wheels shipped in 0.9.4 | LoRA kernels for LLMs (0.9.1); not for diffusion | Yes |
| ort (pykeio) | 2.0.0-rc.13, 2026-07-28 (secondary source) | not verified | No Rust FLUX/Qwen-Image example found | via ORT | via ORT | no | ORT cuDNN flash attn has known sm_120 gaps | no | No (prebuilt ORT binaries; CUDA 13 only) |
| FLAME (inference-flame) | path dep, 319 commits, no releases | MIT | Yes: FLUX.2 Klein 4B/9B, Qwen-Image-Edit-2511, Qwen-Image-2512, many others | Yes | weight-only FP8 (Ideogram-4) | MXFP4 dequant to bf16 only | cuDNN fused SDPA {64,96,128} | lycoris-rs: LoCon/LoHa/LoKr merged into weights | Yes (CUDA 12+, cuDNN 9) |
| stable-diffusion.cpp via `diffusion-rs` (newfla) | sd.cpp news 2026-10-16 | MIT (bindings) | Yes: FLUX.1/.2/.3, Krea2, Qwen Image / 2.1 / Edit series, Z-Image | Yes | not listed | no | "Flash Attention for memory usage optimization" | Yes, "same as stable-diffusion-webui" | C++ (CMake + nvcc for CUDA) |

---

## Key Question 1: Which Rust runtime can run a 20B DiT at bf16/fp8 on a 5090 today, and at what fraction of ComfyUI's speed?

### Takeaway
No framework ships a Qwen-Image-Edit-2511 or Krea 2 port out of the box; the only Rust code that runs these models today is application-level code written on top of candle (qwen-image-rs, qwen3-image21, mold, SceneWorks) or on a private framework (FLAME). The one measured head-to-head is candle-based qwen-image-rs on an RTX 4090: with custom INT8/SageAttention CUDA kernels it is 0.148 s/step vs ComfyUI's best INT8 path 0.156 s/step and ComfyUI stock bf16 0.419 s/step; plain candle bf16 + FlashAttention-2 is 0.62 s/step, i.e. about 1.5x slower than ComfyUI stock bf16 on the same host. No one has published a 5090 (sm_120) number for any of these Rust ports.

### Cited Findings
- qwen-image-rs (umag): "Rust + CUDA inference for Qwen-Image-2.1 on the RTX 4090 — candle spine + custom INT8/fused kernels … No weights." Hardware: "Single NVIDIA RTX 4090 (24 GB, Ada, sm89) under WSL2, CUDA 13.3"; "Other GPUs are untested. The INT8/FP8 kernels and vendored SageAttention/CUTLASS configs target sm80+/sm89." — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen-image-rs timings, 1024², 40 steps, guidance 1: denoise "0.62 → 0.148 s/step (~4.2×)"; VAE decode "1.15 → 0.13 s"; steady-state batch "6.07 s per image"; warm single generate "18.3–18.7 s"; cold "27.0 s"; peak VRAM "20.2 GB (resident batch), 12.9 GB (generate)". Benchmarks "measured 2026-10-02 at HEAD 65e5e7f". — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen-image-rs same-host comparison table: qwen-image-rs HEAD 0.148 s/step, 6.07 s/image, 19.6 GB; "ComfyUI INT8 ConvRot + sage + --fast" 0.156 s/step, 6.94 s, 21.4 GB; "ComfyUI official template (SDPA)" 0.182 s/step, 8.25 s, 16.3 GB; "ComfyUI stock bf16" 0.419 s/step, 19.54 s, 21.8 GB. Also "TensorRT INT8 SmoothQuant: 0.216 s/forward … TensorRT bf16: 0.428. torch.compile bf16: 0.428. Diffusers eager bf16: 0.506." — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen-image-rs bf16 baseline is "bf16 with FlashAttention-2 (flash-attn feature)" at 0.62 s/step — i.e. stock candle bf16 was slower than ComfyUI stock bf16 (0.419) on the same 4090. — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen-image-rs kernel breakdown per 148 ms step: "INT8 GEMMs (CUTLASS, dequant in epilogue): 95.6 ms (65%). SageAttention2: 18.7 ms (12%)"; VAE decode "cuDNN implicit-GEMM conv is 110 ms of ~132 ms". Quality vs diffusers bf16 oracle: "DiT single forward cosine: 0.999911", "VAE decode PSNR: 56.4 dB". — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen-image-rs blockers: "CUDA graphs blocked: candle 0.11 has no stream-capture hook"; repo stats "1 star, 0 forks"; 192 commits; licence Apache-2.0; weights "Research build only" under Qwen Research License. — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen3-image21 (paulwang667): candle-core/candle-nn engine for Qwen-Image-2.1 incl. image-conditioned editing; tested on "NVIDIA L20 (46 GB)". T2I 1024², 20 steps denoise-only: F32 63.2 s, Q8_0 GGUF 29.8 s, Q4_K_M GGUF 30.0 s (KV cache on). Image-conditioned, 1 reference, 20 steps: "Full --precision bf16: 52 s default / 24 s --flash-attn"; later note "BF16 flash 20 s". Peak memory "Full BF16 + --flash-attn: 17.6 GiB". "GGUF transformers and the VAE always compute in F32." No ComfyUI/PyTorch speed comparison. Licence MIT; 36 commits, 1 star. — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21)
- qwen3-image21 notes it "loads ComfyUI-style GGUFs, and the VAE decoder accepts the ComfyUI-repackaged layout. The VAE encoder requires the official layout." Flash attention: "FlashAttention v2, CUDA only, requires sm80+"; build "takes about 9 minutes on a 44-core machine" with "CUDA_COMPUTE_CAP=89". — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21)
- mold (utensils/mold): "Mold is a Rust workspace built on candle"; models "FLUX.1, Flux.2, Stable Diffusion, Z-Image, Qwen-Image, Qwen Image 2.1, Wuerstchen, LTX Video, Wan, MiniMax H3, Hunyuan3D"; example ids "flux-dev:q4", "flux2-klein:bf16", "qwen-image-edit-2511:q4"; tiers "BF16, INT8, FP8, GGUF and 6-step turbo tiers"; only benchmark is internal: "its CUDA fast path is 2.5x faster at 1024² and 6x at 2K than v0.32". No ComfyUI comparison, no Blackwell/5090 mention. MIT, 54 stars, 1,535 commits, 32 open issues; crate `mold-ai-inference` 0.21.0 published 2026-08-05 (first version 2026-03-20). — [utensils/mold](https://github.com/utensils/mold); [crates.io API](https://crates.io/api/v1/crates/mold-ai-inference)
- SceneWorks/inference PR #1006 "Qwen-Image 2.1 native Candle/CUDA provider qwen_image_2_1", merged 2026-09-22: parity vs torch oracle "Pipeline image: 3.1e-5 × peak"; "Not run — no CUDA hardware is reachable from this lane"; no timings; "No on-the-fly Q4/Q8"; DiT uses "chunked block-causal SDPA". — [SceneWorks/inference #1006](https://github.com/SceneWorks/inference/pull/1006)
- FLAME (inference-flame): "Pure Rust diffusion model inference using flame-core … No Python, no diffusers, no ONNX"; models include "FLUX.1-dev … Klein 4B/9B, AsymFLUX.2 Klein 9B … QwenImage-2512, Qwen-Image-Edit-2511"; tested on "24 GB RTX 3090 Ti". Timings: "AsymFLUX.2 Klein 9B: 1024²/38 steps in ~115s"; "Qwen-Image-Edit-2511: 50 steps, true-CFG 4.0, 1024², ~23 min (offloaded)"; Klein 9B Base 1024², 50 steps: "Denoise 193s PyTorch vs 247s Flame; total 322s vs 295s" (README claims "10% faster per-step than PyTorch" but its own table shows 4.94 s/step Flame vs 3.86 PyTorch). MIT; 8 stars, 319 commits; flame-core is a Cargo path dependency (not on crates.io). — [CodeAlexx/inference-flame](https://github.com/CodeAlexx/inference-flame)
- mistral.rs: "The main supported model is FLUX" (FLUX.1-schnell, FLUX.1-dev); "Diffusion models do not support" ISQ; "roughly 12B parameters"; `Flux` (resident, "Fastest") vs `FluxOffloaded` (SDK only); no performance numbers; Qwen-Image, FLUX.2, Krea not mentioned. — [mistral.rs image generation guide](https://ericlbuehler.github.io/mistral.rs/guides/models/use-image-generation/)
- diffusion-rs (EricLBuehler): support matrix lists only "FLUX.1 Dev/Schnell" (DDUF + quantized DDUF); quantization "bitsandbytes format (fp4, nf4, and int8)" and "GGUF (2-8 bit quantization)"; no benchmarks; 130 stars, 84 commits, 6 open issues, no releases; MIT. — [EricLBuehler/diffusion-rs](https://github.com/EricLBuehler/diffusion-rs); crate [diffusion_rs_core 0.1.0](https://crates.io/crates/diffusion_rs_core/0.1.0)
- Generic candle-vs-PyTorch on a small DiT (RTX 4070 Ti SUPER, fp32): "0.18 s vs 0.20 s for 2 images, 10 steps" (candle slightly faster), but at batch 100/50 steps "PyTorch was faster (22.5 s vs 35.0 s)". — [j-rmccrery/dit-from-scratch](https://github.com/j-rmccrery/dit-from-scratch)
- Candle issue #3052 "Candle vs. PyTorch performance" (Sep–Nov 2025): candle matched PyTorch on plain matmul but "when addition operations were included, the performance dropped significantly" (no op fusion). — [candle #3052](https://github.com/huggingface/candle/issues/3052)
- Reference PyTorch/ComfyUI baseline: FLUX.1 [dev] 28 steps 1024² "5.8 s to 14.5 s per image depending on the GPU" (April 2026, ComfyUI). — [gigagpu.com](https://gigagpu.com/image-gen-benchmark-update-april-2026/)
- Burn: no diffusion/image model mentioned in the 0.20.0 or 0.22.0 release notes; no Rust DiT port on burn was found by any search. — [Burn 0.20.0 blog](https://tracel.ai/blog/release-0.20.0); [Burn v0.22.0 release](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)

### Inferences
- Stock candle bf16 on a 20B DiT lands at roughly 0.65–0.7x of ComfyUI stock bf16 speed (0.62 vs 0.419 s/step on a 4090), mainly because candle has no op fusion, no CUDA-graph capture and no SageAttention-class attention; parity with ComfyUI required ~5 custom CUDA kernels (INT8 CUTLASS GEMM, SageAttention2, fused norms/SwiGLU) in qwen-image-rs.
- A 5090 should run the bf16 Qwen-Image-2.1 DiT (≈40 GB bf16 total pipeline, ~14 GB DiT bf16 per qwen3-image21's "14 GB (BF16)" disk size, with 17.6 GiB peak measured for bf16+flash) resident in 32 GB, but Qwen-Image-Edit-2511's extra reference-image tokens push attention memory up; chunked SDPA or FA2 is mandatory. This is an inference from the L20 numbers, not a 5090 measurement.
- The efficient path on a 5090 would be INT8/FP8 weights with Blackwell kernels, which no Rust framework provides upstream today; qwen-image-rs's CUTLASS configs are sm89-only and would need re-tuning for sm_120.

### Gaps
- No published timing of any Rust Qwen-Image / FLUX / Krea 2 port on an RTX 5090 (sm_120) was found.
- No Rust port of Krea 2 or Qwen-Image-Edit-2511 on candle with timings was found (mold lists `qwen-image-edit-2511:q4` but publishes no numbers; FLAME's 2511 run is offloaded on a 3090 Ti, ~23 min).
- diffusion-rs / mistral.rs publish no FLUX throughput numbers at all.
- mold's README fetch via lib.rs returned 403; details come from the GitHub README and crates.io API only.

---

## Key Question 2: CUDA/Blackwell (sm_120) support and build story per runtime

### Takeaway
candle reaches sm_120 only through PTX forward-compatibility: issue #2966 "Add support for sm_120" (opened 2025-05-23) is still open, a contributor reported it working on sm_121 (GB10) from main with CUDA 13.0 in Dec 2025, and a separate open issue (#3358, Feb 2026) shows a 5090 failing device init with `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` under candle 0.9.2 / CUDA 13.1 / driver 580. cudarc 0.19.10 has `cuda-12080` and `cuda-13000` features and `dynamic-loading`. Burn/CubeCL compiles kernels through LLVM→PTX at runtime (no nvcc) and its 0.20 matmul engine explicitly targets "Blackwell-based hardware like the RTX 5090". mistral.rs ships `cuda128.sm120` and `cuda130` wheels since 0.9.4.

### Cited Findings
- candle-core: latest 0.11.0 published "2026-06-26T21:14:33Z"; prior 0.10.2 (2026-04-01), 0.10.0 (2026-03-31), 0.9.2 (2026-01-24); licence "MIT OR Apache-2.0"; features `cuda: ["cudarc","dep:candle-kernels","candle-ug?/cuda"]`, `cudnn: ["cuda","cudarc/cudnn"]`, `nccl`, `metal`, `mkl`, `accelerate`, `ug`. — [crates.io API candle-core](https://crates.io/api/v1/crates/candle-core)
- candle GitHub has "There aren't any releases here" (versions are crates.io-only). — [candle releases](https://github.com/huggingface/candle/releases)
- candle #2966 "Add support for sm_120", opened 2025-05-23, OPEN. Comment 2025-12-05 (DrJesseGlass): sm_120/sm_121 "working", confirmed on "NVIDIA GB10 (sm_121) with CUDA Toolkit 13.0 on aarch64, built from main", build log "compute_cap: Some(121)", "kernels compiled through PTX forward compatibility"; Qwen3-0.6B BF16 CUDA "64.15 generation tok/s", Q4_K_M "128.90 generation tok/s". No release version named; sm_120 specifically not confirmed in-thread. — [candle #2966](https://github.com/huggingface/candle/issues/2966)
- candle #3358 "forward compatibility was attempted on non supported HW" on RTX 5090, opened 2026-02-03, OPEN, zero comments: `DriverError(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)` from `CudaDevice::new`, candle-core 0.9.2, container `nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04`, driver 580.126.09 (running mistral.rs). — [candle #3358](https://github.com/huggingface/candle/issues/3358)
- candle #3199 "CUDA integer reductions use undefined initialization (±INFINITY) causing incorrect max/min on Blackwell (sm_100)" closed 2025-11-20. — [candle issue search](https://github.com/huggingface/candle/issues?q=is%3Aissue%205090%20OR%20sm_120%20OR%20blackwell)
- cudarc 0.19.10 published 2026-09-24 (0.19.9 2026-08-11, 0.19.8 2026-06-19); features include `cuda-12080`, `cuda-13000`, `dynamic-loading`, `cublaslt: ["driver"]`, `cudnn: ["cudnn-09021"]`; licence MIT OR Apache-2.0. — [crates.io API cudarc](https://crates.io/api/v1/crates/cudarc)
- candle-based Qwen-Image ports build kernels with nvcc at build time: qwen3-image21's flash-attn build "takes about 9 minutes on a 44-core machine" with `CUDA_COMPUTE_CAP=89`; qwen-image-rs requires `CUTLASS_DIR`, cuDNN 9, and `cargo build --release --features convrot,sage,fusednorm,sage2,cudnn`. — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21); [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- Burn 0.20.0 (2026-01-15): matmul engine "combines TMA with warp specialization, specifically targeting Blackwell-based hardware like the RTX 5090", "also benefit NVIDIA's Ada and Hopper"; adds "Tensor Memory Accelerator (TMA) and inlined PTX for manual MMA instructions". Matmul (1,4096,4096): CubeCL 0.20 639µs vs LibTorch 627µs; (1,6144,6144): 1.99ms vs 2.00ms; batched vec@mat (4096,1,4096): 638µs vs 0.19.1's 58.82ms vs LibTorch 628µs. — [Burn 0.20.0 blog](https://tracel.ai/blog/release-0.20.0)
- Burn 0.22.0 published 2026-10-06 (crates.io "2026-10-06T18:46:00Z"; pre-releases from 2026-07-29; 0.21.0 2026-05-07); licence MIT OR Apache-2.0. Release notes: "CubeCL also adds LLVM-based CUDA and AMD GPU compilation paths"; "perf(cube): update cubecl for llvm TF32 support on CUDA (#5979)"; "Candle has been removed in this release, and NdArray and LibTorch are being deprecated"; "Migrate to pliron" IR; backend generics removed. Blackwell not mentioned. — [crates.io API burn](https://crates.io/api/v1/crates/burn); [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)
- CubeCL: "currently in alpha … expect breaking changes between minor versions"; runtimes "NVIDIA via CUDA (LLVM → PTX)", HIP, wgpu, CPU; "For the LLVM targets, the build fetches a prebuilt LLVM, so you don't need to install it yourself"; CUDA runtime "executes PTX through cudarc"; tensor cores "not available with the WGSL compiler"; "Cluster variables … available on NVIDIA Hopper and newer with the cuda-cpp compiler"; CubeK "split out from CubeCL" with "Ready-made kernels for matrix multiplication, attention, and reductions"; MIT OR Apache-2.0; cubecl newest 0.11.0-pre.3 (2026-08-25). — [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl); [docs.rs cubecl](https://docs.rs/crate/cubecl/latest)
- mistral.rs v0.9.4 (24 Sep 2026) assets include "mistralrs-0.9.4+cuda128.sm120-cp310-abi3-manylinux_2_35_x86_64.whl" and "cuda130.sm100" wheels. v0.9.3 (07 Sep): "fix(cuda): restore GB10 builds with cudaforge 0.1.6". — [mistral.rs releases](https://github.com/EricLBuehler/mistral.rs/releases); [v0.9.3](https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.3)
- mistralrs crate: 0.8.1 (2026-04-02), 0.7.0 (2026-01-28); MIT. The GitHub releases (0.9.x) are ahead of crates.io. — [crates.io API mistralrs](https://crates.io/api/v1/crates/mistralrs)
- ort (pykeio): "latest version is 2.0.0-rc.13, released 28 July 2026 … moving to ONNX Runtime 1.28 and shipping only CUDA 13 binaries" (secondary summary of the releases page). — [ort releases via search](https://github.com/pykeio/ort/releases); [ai-tldr summary](https://ai-tldr.dev/tools/ort-rust/)
- ONNX Runtime on sm_120: "GroupQueryAttention throws 'Failed to build cuDNN graph for Flash Attention' on Blackwell (sm_120) with head_size 256 - no fallback" (issue #33154). — [onnxruntime #33154](https://github.com/microsoft/onnxruntime/issues/33154)
- Context from the ecosystem: llama.cpp #24399 reports an intermittent sm_120-specific crash in `mul_mat_q<Q8_0,128>` built with `CMAKE_CUDA_ARCHITECTURES=120`. — [llama.cpp #24399](https://github.com/ggml-org/llama.cpp/issues/24399)

### Inferences
- candle on a 5090 is "works by PTX JIT": `candle-kernels` is compiled by nvcc at build time for whatever `CUDA_COMPUTE_CAP` is detected, so a 5090 build should emit sm_120 SASS when the local CUDA toolkit is ≥12.8, but there is no sm_120 CI and the two Blackwell issues (#2966, #3358) are both open with no maintainer response.
- Burn/CubeCL is the only runtime whose kernel generator explicitly optimises for the 5090 (TMA + warp specialisation), and it avoids nvcc entirely; but it has no diffusion models and its attention kernel is broken (see KQ4).
- Binary size: none of the sources state binary sizes. Candle's `candle-kernels` embeds PTX for every kernel; flash-attn adds a large cubin bundle (9-minute build suggests many instantiations). This is unquantified.

### Gaps
- No source states whether `candle-kernels` emits native sm_120 SASS or only `compute_90` PTX for a 5090 build; the #2966 comment describes forward-compatible PTX on sm_121.
- No binary-size figures for any runtime were found.
- ort's licence and exact rc.13 release note text were not fetched directly (crates.io HTML is JS-only; the releases page was summarised by a secondary site).

---

## Key Question 3: FP8 / NVFP4 matmuls on Blackwell (cuBLASLt scaled matmul, block-scaled FP4)

### Takeaway
Upstream candle has an `F8E4M3` dtype but its CUDA kernels are unreachable because of a kernel-name mismatch (issue #3886, open since 2026-08-14) and there is no fp8 matmul; the only Rust NVFP4 matmul found is a SceneWorks fork crate (`candle-quant-kernels`) that calls cuBLASLt W4A4 FP4 GEMM, refuses anything below sm_120, and is wired only into LLM projections (perplexity +7.37%, measured on an RTX Pro 6000). mistral.rs added FP8 and NVFP4 checkpoint loading with cuTile Blackwell kernels in v0.9.3 (Sep 2026) for LLMs only. Burn/CubeCL state no FP8/FP4 compute support. cudarc exposes `cublaslt`.

### Cited Findings
- candle #3886 "F8E4M3 CUDA kernels are unreachable: DType::as_str() yields f8e4m3, the kernels are named f8_e4m3", opened 2026-08-14, OPEN, linked PRs #3911 and #3895 both open. Kernels declared in "affine.cu, binary.cu, cast.cu, fill.cu, indexing.cu, and unary.cu"; casts gated `__CUDA_ARCH__ >= 800`, affine/indexing `>= 890`; "Casts, copies, affine, binary ops, comparisons, and indexing" broken on CUDA; "Metal: Declares no F8E4M3 kernels"; fp8 matmul not mentioned at all. — [candle #3886](https://github.com/huggingface/candle/issues/3886)
- candle PR #3801 (ROCm backend) independently reports "the CUDA F8E4M3 dispatch looks up kernel names that don't exist in candle-kernels"; downstream fork inference.rs #313 fixes the same naming bug. — [candle #3801](https://github.com/huggingface/candle/pull/3801); [inference.rs #313](https://github.com/christopherthompson81/inference.rs/pull/313)
- SceneWorks/inference PR #1022 "NVFP4 as a shared Projection variant" (merged 2026-09-23): "quantizes the bf16 projection weights at load, on the device"; prefill and decode "through the existing cuBLASLt W4A4 FP4 GEMM"; codec + cuBLASLt wrapper moved to "crates/kernels/candle-quant-kernels"; load "fails unless there is a CUDA device, a cuBLASLt handle, compute capability of at least sm_120, and a working fused quantizer"; Qwen3.8-27B on RTX Pro 6000: "4.500 bits/param, versus 16.0 for bf16", device memory "22.95 GB versus 56.38 GB", perplexity "7.8366 (bf16) to 8.4140 (NVFP4), about 7.37% worse … outside the 2% gate, so NVFP4 stays opt-in"; "LLMs only … does not cover diffusion models"; refused for "projection shapes where N % 16 ≠ 0". — [SceneWorks/inference #1022](https://github.com/SceneWorks/inference/pull/1022)
- SceneWorks PR #1027 adds a "fused NVFP4 decode GEMV inside Projection::Nvfp4" with cuBLASLt as fallback. — [SceneWorks/inference #1027](https://github.com/SceneWorks/inference/pull/1027)
- mistral.rs v0.9.3: "perf: first-class FP8 and concurrent DFlash decode" (#2390); "feat(core): fp8 tensor-core gemv/gemm, cutile gdn prefill" (#2412); "feat(cutile): add fp8 fused moe and a load-time kernel autotuner" (#2413); "feat(fp8): support ModelOpt and compressed-tensors checkpoints with cutile kernels" (#2416); "feat(quant): add NVFP4 checkpoint loading and cuTile Blackwell acceleration" (#2418); "bump cutile to v0.3.0". No performance numbers; FLUX/diffusion not mentioned. — [mistral.rs v0.9.3](https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.3)
- mistral.rs README: diffusion models "can run, but quantization and adapters are not yet supported for them". — [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- Burn 0.20.0: FP8/FP4 "not stated"; only a `QuantStore` example naming packed "e2m1x2"; `PackedNative(usize)` / `PackedU32(usize)` stores. Burn 0.22.0: "feat(quant): support UE4M3 scale params end to end on CPU (#5253)", "two-level quantization scales (#5262)"; no FP8/FP4 compute named. CubeCL README: dtypes FP8/FP4 "Not mentioned". — [Burn 0.20.0 blog](https://tracel.ai/blog/release-0.20.0); [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0); [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)
- FLAME: "FP8: Ideogram-4 'FP8 weight-only per-output-row scale'"; "MXFP4: Lens GPT-OSS experts dequantized to BF16"; no FP8 tensor-core matmul claimed. — [CodeAlexx/inference-flame](https://github.com/CodeAlexx/inference-flame)
- qwen-image-rs uses INT8 (W8A8 "ConvRot") CUTLASS GEMMs and SageAttention2 ("INT8 Q·K, FP8 P·V"); "W4A4 (INT4) GEMMs are listed as a future lever, not implemented". — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- Hardware facts: NVFP4 block-scaled tensor cores "need sm_100/sm_103 or sm_120/sm_121"; cuBLASLt has "no F16 × E4M3 kernel at M=16" (mixed-dtype gap reported in another project). — [kadirnar/kernel-agent #233](https://github.com/kadirnar/kernel-agent/issues/233); [kekzl/imp #2563](https://github.com/kekzl/imp/issues/2563)
- A CUTLASS CuTeDSL PR for SM120 FP8 flash attention notes "FP8 MMA segfaults on SM120 in the CuTe DSL path and relies on inline PTX as a workaround". — [NVIDIA/cutlass #3030](https://github.com/NVIDIA/cutlass/pull/3030)

### Inferences
- Loading ComfyUI `fp8_e4m3fn` / "fp8mixed" safetensors into upstream candle today means upcasting to bf16 at load (the dtype exists in `safetensors` but no CUDA ops run on it); ComfyUI NVFP4 repacks (block-scaled, with scale tensors) have no loader anywhere in candle/burn/diffusion-rs upstream. The SceneWorks crate quantises bf16 at load rather than reading pre-quantised NVFP4 files.
- Any PhotoCraft FP8/NVFP4 path would have to be hand-written against cuBLASLt via cudarc's `cublaslt` feature (as SceneWorks did, ~1 PR) or imported from mistral.rs's cuTile work, and would be the first such thing for a DiT in Rust.

### Gaps
- No evidence of anyone running an FP8 or NVFP4 diffusion transformer in any Rust runtime.
- No source gives FP8/NVFP4 matmul throughput from Rust on a 5090.
- Whether mistral.rs's cuTile kernels (cuTile is NVIDIA's tile DSL; the release lists `cudaforge`) require nvcc or a Python toolchain at build time was not determined.

---

## Key Question 4: Flash / fused attention availability for Blackwell in each runtime

### Takeaway
candle's `candle-flash-attn` wraps FlashAttention-2 (sm80+) and is what the Qwen-Image ports use; it compiles for a given `CUDA_COMPUTE_CAP` and its FA4 successor (`candle-flash-attn-v4`, experimental, unreleased) explicitly packages SM120 and passed 24 native kernel tests on an RTX Pro 6000, FP16/BF16 only, no FP8. Burn's CubeK attention kernel is unfinished: issue #4325 "cubecl: flash attention fails on most backends" (Vulkan wrong values, CUDA assertion) is open and labelled `blocker` as of 2026-09-24. mistral.rs has FAv3 for LLMs only. FLAME uses cuDNN fused SDPA restricted to head dims {64, 96, 128}.

### Cited Findings
- candle-flash-attn-v4 (michaelfeil): "Experimental native FlashAttention-4 bindings for Rust and Candle, using upstream kernels"; "Experimental, not yet a release"; bundles target "SM80, SM86, SM89, SM90, and SM120"; "Datacenter Blackwell SM100/110 … not packaged yet"; "RTX Pro 6000 (SM120) passed all 24 native kernel reference cases. Its model-level qualification is still in progress"; dtypes FP16/BF16, "FP8 … not exposed"; head dims 64 and 128 only; build needs "Linux, CUDA, Python with CUDA-compatible PyTorch, and a C++17 compiler", pinned "CUTLASS DSL 4.7.1, and TVM FFI 0.1.14.post1"; wrapper MIT OR Apache-2.0, upstream BSD-3-Clause. — [michaelfeil/candle-flash-attn-v4](https://github.com/michaelfeil/candle-flash-attn-v4)
- candle-flash-attn (FA2) in practice: qwen3-image21 "FlashAttention v2, CUDA only, requires sm80+ (A100, RTX 30xx, L20 and newer)", opt-in, "compiling takes about 9 minutes"; qwen-image-rs bf16 baseline uses "FlashAttention-2 (flash-attn feature)" and replaces it with vendored "SageAttention2 (INT8 Q·K, FP8 P·V)" targeting sm89. — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21); [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- candle #2753 "Bug Error compiling with flash attn" (Jan 2025, pre-mid-2025) is a build-script bug, not architecture-related. — [candle #2753](https://github.com/huggingface/candle/issues/2753)
- Upstream FA2 on sm_120: "building flash-attn 2.x with FLASH_ATTN_CUDA_ARCHS=120 made nvcc segfault on backward-pass kernels, with CUDA 12.8 and 12.9 both failing" (forward-only builds are what candle-flash-attn needs). — [Dao-AILab/flash-attention #2361](https://github.com/Dao-AILab/flash-attention/issues/2361)
- Burn #4325 "cubecl: flash attention fails on most backends": opened 2026-01-14 by wingertge; OPEN; labels `blocker`, `bug` (added 2026-09-24); test `attention::test_attention_no_mask` "fails on Vulkan with incorrect values" and "on CUDA with an assertion failure in cubek-attention: unit_tile.layout.num_cols % line_size == 0"; nathanielsimard 2026-01-14: kernel "work in progress and should be ready for burn 0.21"; April 2026 commits "skip the test rather than fix the kernel". — [burn #4325](https://github.com/tracel-ai/burn/issues/4325)
- Burn 0.20.0 blog: engine "generates optimal Flash Attention kernels" per backend; "no benchmark numbers for it". Burn 0.22.0 notes list only "perf(flex): parallelize attention across (batch, head) pairs (#5612)" and "fix(flex): support GQA/MQA in attention"; "No flash attention kernel is named." — [Burn 0.20.0 blog](https://tracel.ai/blog/release-0.20.0); [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)
- cubek-attention crate latest "0.3.0-pre.3 from August 25, 2026, still a pre-release"; a Burn fork issue notes the autotuner can pick a slower FlashUnit over a faster fallback. — [lib.rs cubek-attention](https://lib.rs/crates/cubek-attention); [Mikyx-1/burn #376](https://github.com/Mikyx-1/burn/issues/376)
- Burn feature request #4931 (2026-05-08) for paged flash attention with block tables — open feature request. — [burn #4931](https://github.com/tracel-ai/burn/issues/4931)
- mistral.rs v0.9.2 (20 Aug 2026): "fix(fav3): misc fixes relating to FAv3" (LLM path). — [mistral.rs releases](https://github.com/EricLBuehler/mistral.rs/releases)
- FLAME: Boogu "head_dim 120 is zero-padded to 128 to use flame's fused cuDNN SDPA"; Ideogram-4 "head_dim 256 is outside the cuDNN/WMMA flash set {64, 96, 128}, so it uses a materialized SDPA fallback with Q-tiling". — [CodeAlexx/inference-flame](https://github.com/CodeAlexx/inference-flame)
- SceneWorks candle Qwen-Image provider uses "chunked block-causal SDPA, budgeted by ATTN_SCORES_BUDGET" (no flash kernel). — [SceneWorks/inference #1006](https://github.com/SceneWorks/inference/pull/1006)
- stable-diffusion.cpp lists "Flash Attention for memory usage optimization" (ggml). — [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md)

### Inferences
- On a 5090 with candle, the realistic attention options are: (a) FA2 via `candle-flash-attn` built with `CUDA_COMPUTE_CAP=120` (unverified on sm_120 by any source found; upstream FA2 forward kernels are generally built for 12.0 by the community), (b) naive/chunked SDPA, or (c) vendoring SageAttention as qwen-image-rs did. FA4 bindings are promising for Blackwell but experimental and Linux/Python-at-build-time only.
- Burn's attention is not usable for production DiT inference until #4325 closes; this plus the absence of any DiT port makes burn a 2027 option, not an October 2026 one.

### Gaps
- No report of `candle-flash-attn` (FA2) being built and run on an sm_120 consumer card was found.
- Qwen-Image's head dim (128) fits FA2/FA4; whether Krea 2 / FLUX head dims (128) are covered by candle-flash-attn-v4's "64 and 128 only" is fine, but no one has benchmarked it.

---

## Key Question 5: Runtime LoRA merging/applying support

### Takeaway
No Rust framework offers diffusers/ComfyUI-style runtime LoRA application for DiTs out of the box: candle-lora is a training-oriented macro crate whose "weight naming is not compatible with peft yet"; diffusion-rs lists LoRA as "upcoming"; mistral.rs added LoRA kernels for LLMs (v0.9.1) but states adapters are not supported for diffusion. Application-level projects do it themselves: mold caches "derived LoRA deltas" per adapter revision, and FLAME's `lycoris-rs` merges LoCon/LoHa/LoKr into weights. stable-diffusion.cpp supports LoRA "same as stable-diffusion-webui".

### Cited Findings
- candle-lora (EricLBuehler): "Weight merging is supported to improve inference speed, and weights can be unmerged"; layers "Linear, Conv1d, Conv2d, and Embedding"; "weight naming is 'not compatible with peft yet'"; diffusers not mentioned; installed via `cargo add --git` (no crates.io pin); MIT; 232 commits, 176 stars, 6 open issues; supported model list is LLM/vision classifiers only (llama, mistral, falcon, bert, stable_lm, t5, dinov2, resnet, mpt, blip, starcoder). — [EricLBuehler/candle-lora](https://github.com/EricLBuehler/candle-lora)
- diffusion-rs: "Upcoming features: 🚧 LoRA support". — [EricLBuehler/diffusion-rs](https://github.com/EricLBuehler/diffusion-rs)
- mistral.rs v0.9.1 (14 Aug 2026): "feat(lora): add true lora kernels and dynamic adapter support" — LLM path; README: for diffusion "quantization and adapters are not yet supported". — [mistral.rs releases](https://github.com/EricLBuehler/mistral.rs/releases); [mistral.rs README](https://github.com/EricLBuehler/mistral.rs)
- mold: "LoRA reuse tracks adapter file revisions as well as strengths; replacing an adapter invalidates cached weights"; "FLUX and SD3 retain at most 256 MiB of derived LoRA deltas per engine". — [utensils/mold](https://github.com/utensils/mold)
- FLAME: "lycoris-rs: LoCon, LoHa, LoKr, and Full adapters, merged into weights. 'DoRA loud-skips.'"; AsymFLUX.2 "58 LoRA pairs fused". — [CodeAlexx/inference-flame](https://github.com/CodeAlexx/inference-flame)
- Burn 0.22.0 release notes include LoRA and QLoRA support (training-side module API). — [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)
- SceneWorks PR #1146 "Qwen-Image 2.1 candle LoRA/LoKr trainer" and #1145 "every adapter format resolves on both backends" exist (titles only; bodies not fetched). — [SceneWorks/inference #1146](https://github.com/SceneWorks/inference/pull/1146); [#1145](https://github.com/SceneWorks/inference/pull/1145)
- qwen-image-rs, qwen3-image21 and vincenthz/qwen-imager: LoRA "Not mentioned" / "Not supported". — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs); [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21); [vincenthz/qwen-imager](https://github.com/vincenthz/qwen-imager)
- Offline alternative: `tpt_lora_merge` crate does CPU merging "base + scale * (B @ A)" into safetensors. — [docs.rs tpt-lora-merge](https://docs.rs/tpt-lora-merge/latest/tpt_lora_merge/)
- stable-diffusion.cpp: LoRA supported, "same as stable-diffusion-webui". — [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md)

### Inferences
- Runtime LoRA for a candle DiT is a small amount of code (W += scale·B·A on load, or an extra low-rank matmul per linear) but every project writes it afresh; the hard part is key-name mapping from ComfyUI/diffusers LoRA files (`diffusion_model.transformer_blocks.N...` vs diffusers names), which nobody in the Rust ecosystem has published as a reusable crate for Qwen-Image or FLUX.

### Gaps
- No Rust crate found that parses ComfyUI/Kohya-format DiT LoRA safetensors for Qwen-Image/FLUX generically.

---

## Key Question 6: Model file formats accepted (safetensors, GGUF) and whether ComfyUI-repacked fp8mixed / nvfp4 safetensors load

### Takeaway
All candle-based ports read sharded diffusers-layout safetensors and (qwen3-image21, mold, umag) GGUF; qwen3-image21 explicitly accepts ComfyUI-style GGUFs and the ComfyUI-repackaged VAE decoder layout. No Rust runtime was found that loads ComfyUI fp8 (`fp8_e4m3fn`/fp8mixed) or NVFP4 safetensors as fp8/fp4 compute; candle cannot even cast F8E4M3 on CUDA today (#3886). diffusion-rs reads DDUF (bnb fp4/nf4/int8, GGUF 2–8 bit).

### Cited Findings
- candle safetensors loading: `MmapedSafetensors` memory-mapped loading and `save`. — [docs.rs candle_core::safetensors](https://docs.rs/candle-core/latest/candle_core/safetensors/index.html)
- qwen3-image21: "Weights in: sharded safetensors (F32 or BF16 compute) and GGUF (Q8_0, Q4_K_M tested)" from `unsloth/Qwen-Image-2.1-GGUF`; "loads ComfyUI-style GGUFs, and the VAE decoder accepts the ComfyUI-repackaged layout. The VAE encoder requires the official layout"; "GGUF transformers and the VAE always compute in F32"; Q8_0 cosine 0.99986, Q4_K_M 0.99537 vs full precision. — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21)
- qwen-image-rs: "loader/ handles safetensors and GGUF"; text encoder "Q8_0 GGUF (~8 GB)"; DiT "prequantized INT8 ConvRot cache (~7.1 GB)" generated locally. — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- mold: tiers "BF16, INT8, FP8, GGUF"; "mold quantize can create 'qualified FP8/GGUF tiers'"; "fp8-scaled Wan checkpoints remain CUDA-only"; ComfyUI weight compatibility "Not mentioned". — [utensils/mold](https://github.com/utensils/mold)
- vincenthz/qwen-imager (Metal only): "Weights: Diffusers layout, with safetensors files"; also MLX affine 4/8-bit packs. — [vincenthz/qwen-imager](https://github.com/vincenthz/qwen-imager)
- diffusion-rs: "strong support for DDUF models"; example `FLUX.1-dev-Q4-bnb.dduf`; "bitsandbytes format (fp4, nf4, and int8)", "GGUF (2-8 bit quantization)". — [EricLBuehler/diffusion-rs](https://github.com/EricLBuehler/diffusion-rs)
- mistral.rs: diffusion models "Load them at native precision instead of passing --quant or --isq". — [mistral.rs image generation guide](https://ericlbuehler.github.io/mistral.rs/guides/models/use-image-generation/)
- candle F8E4M3 on CUDA: "Casts, copies, affine, binary ops, comparisons, and indexing" broken (#3886). — [candle #3886](https://github.com/huggingface/candle/issues/3886)
- Burn 0.22.0: "fix(store): respect PyTorch tensor strides (#5392)", "write safetensors files atomically (#5489)", "load torch.save(model) full-model pickles (#5766)"; records migrate to burnpack. — [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)
- stable-diffusion.cpp: "Pytorch checkpoint (.ckpt/.pth/.pt)", "Safetensors", "GGUF"; guides "Quantization and GGUF" and "INT8 convrot safetensors". — [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md)

### Inferences
- A ComfyUI `fp8_e4m3fn` Qwen-Image-Edit-2511 safetensors could be read by candle's safetensors loader as raw F8E4M3 tensors but must be upcast (on CPU, or with a fixed kernel) to bf16 before any CUDA op; "fp8mixed" (per-tensor dtype mix) is loadable the same way. ComfyUI NVFP4 repacks (packed uint8 + block scales + per-tensor scale) have no loader anywhere; writing one is ~tens of lines but the matmul to consume it does not exist outside the SceneWorks fork.

### Gaps
- No source documents the exact tensor layout of ComfyUI's NVFP4 repacks or a Rust reader for them.

---

## Key Question 7: What porting a new architecture costs (lines of Rust, who has done it)

### Takeaway
Nobody publishes line counts, but the shape of the effort is visible: a from-scratch candle Qwen-Image-2.1 pipeline (Qwen2.5-VL/Qwen3-VL text tower, 60-block MMDiT, 16-ch VAE, flow-matching scheduler) was done at least four times independently in 2026 (umag 192 commits, paulwang667 36 commits, vincenthz 37 commits, SceneWorks one 17-commit PR), each validated against a frozen diffusers oracle to ~1e-5; reaching ComfyUI-parity speed then required a second project-sized effort of hand-written CUDA kernels (CUTLASS INT8 GEMM, SageAttention2, fused norms) on top of candle.

### Cited Findings
- SceneWorks PR #1006: Qwen-Image 2.1 provider, "17 commits", parity "Transformer stages: 1.5e-5 × peak … VAE encoder: 3.6e-7 × peak … Text encoder: 2.3e-7 absolute"; "Lines/files changed: Not stated". — [SceneWorks/inference #1006](https://github.com/SceneWorks/inference/pull/1006)
- qwen-image-rs: "192 commits"; components: candle spine + "custom CUDA kernels cover INT8 GEMMs, SageAttention, and fused norms, SwiGLU, and VAE ops"; "Two correctness bugs were found and fixed (stale shared memory in vendored SageAttention; ignored view offsets at B=1)". — [umag/qwen-image-rs](https://github.com/umag/qwen-image-rs)
- qwen3-image21: 36 commits; each stage "match the upstream diffusers pipeline to ~1e-5"; golden tensors from `tools/ref_dump_i2i.py` (needs PyTorch/diffusers/transformers). — [paulwang667/qwen3-image21](https://github.com/paulwang667/qwen3-image21)
- vincenthz/qwen-imager: 37 commits; "a Qwen3-VL encoder, a single-stream DiT denoiser, an RGBA VAE, and a fixed flow-matching Euler schedule"; library crate + CLI + GPUI desktop app; Apache-2.0; M2 Max 1024², 4 steps bf16 "9.5 s / 49 s / 29.6 GB" (per step / total / peak). — [vincenthz/qwen-imager](https://github.com/vincenthz/qwen-imager)
- dit-from-scratch: "Rust/Candle inference port on CUDA, matching PyTorch to 3e-6". — [j-rmccrery/dit-from-scratch](https://github.com/j-rmccrery/dit-from-scratch)
- FLAME ported ~20 image/video architectures (FLUX.1, FLUX.2 Klein, Qwen-Image-2512/Edit-2511, Z-Image, Chroma, SD3.5, SDXL, LTX-2.3, Wan …) in 319 commits on a private framework; known gaps listed per model (e.g. "sensenova_u1_edit currently produces tiled artifacts", "Motif VAE decode runs via a Python bridge"). — [CodeAlexx/inference-flame](https://github.com/CodeAlexx/inference-flame)
- mold: 1,535 commits for a full product covering FLUX.1/.2, Qwen-Image/Edit-2511, Z-Image, video. — [utensils/mold](https://github.com/utensils/mold)
- Ecosystem churn risk: Burn 0.22 "Candle has been removed in this release, and NdArray and LibTorch are being deprecated", backend generics removed — porting to burn now means targeting a moving API. — [Burn v0.22.0](https://github.com/tracel-ai/burn/releases/tag/v0.22.0)
- Alternative "Rust-first" wrapper: stable-diffusion.cpp already supports "FLUX.1-dev/schnell, FLUX.2-dev/klein, FLUX.3 Image Klein … Qwen Image, Qwen Image 2.1 … Krea2 … Qwen Image Edit series … Z-Image" with news "2026/10/16, Day-0 support for FLUX.3 Image Klein"; Rust bindings `diffusion-rs` by newfla (MIT, 55 stars, 202 commits, CUDA "working on Windows and Linux", preset-based `PresetBuilder`/`gen_img` API). — [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md); [newfla/diffusion-rs](https://github.com/newfla/diffusion-rs)

### Inferences
- Porting Qwen-Image-Edit-2511 to candle: the DiT/VAE/scheduler are shared with 2.1 ports (only the edit conditioning path and Qwen2.5-VL vs Qwen3-VL text tower differ), so a port that reuses one of the Apache-2.0/MIT candle projects is weeks, not months; Krea 2 (FLUX.1-dev architecture fine-tune) can reuse candle-transformers' existing FLUX module. Getting ComfyUI-class speed on the 5090 is the larger unknown: it needs sm_120-tuned INT8/FP8 GEMMs and a Blackwell attention kernel that no Rust project has yet written.
- stable-diffusion.cpp via newfla's bindings is the only path with Krea 2 + Qwen-Image-Edit + FLUX.2 + LoRA + GGUF today, at the cost of a C++/CMake/nvcc build and ggml's (not Rust's) crash/safety semantics.

### Gaps
- No line-of-code counts for any port; no one documents engineer-time.
- No Rust FLUX.1/Krea 2 timing on any NVIDIA GPU was found (candle-transformers' FLUX example exists but no benchmark surfaced).
- stable-diffusion.cpp performance vs ComfyUI on a 5090 was out of scope and not researched.

---

## Maintenance / licence snapshot (all verified 2026-10-16 unless noted)

- candle-core 0.11.0 (2026-06-26), MIT OR Apache-2.0; no GitHub releases; Blackwell issues #2966 (2025-05) and #3358 (2026-02) open; FP8 issue #3886 (2026-08) open. — [crates.io API](https://crates.io/api/v1/crates/candle-core); [issues](https://github.com/huggingface/candle/issues?q=is%3Aissue%205090%20OR%20sm_120%20OR%20blackwell)
- burn 0.22.0 (2026-10-06), MIT OR Apache-2.0; cubecl 0.11.0-pre.3 (2026-08-25), alpha; cubek-attention 0.3.0-pre.3 (2026-08-25); flash-attention blocker #4325 open. — [crates.io API burn](https://crates.io/api/v1/crates/burn); [docs.rs cubecl](https://docs.rs/crate/cubecl/latest); [lib.rs cubek-attention](https://lib.rs/crates/cubek-attention); [burn #4325](https://github.com/tracel-ai/burn/issues/4325)
- diffusion-rs (EricLBuehler) MIT; crate 0.1.0 only; 84 commits, 130 stars, 6 open issues; issue #36 (Jan 2025, pre-mid-2025) F16/BF16 dtype mismatch loading FLUX. — [repo](https://github.com/EricLBuehler/diffusion-rs); [#36](https://github.com/EricLBuehler/diffusion-rs/issues/36)
- mistral.rs MIT; crates.io 0.8.1 (2026-04-02); GitHub v0.9.4 (24 Sep 2026) with sm_120/CUDA 13 wheels. — [crates.io API](https://crates.io/api/v1/crates/mistralrs); [releases](https://github.com/EricLBuehler/mistral.rs/releases)
- cudarc 0.19.10 (2026-09-24), MIT OR Apache-2.0, features cuda-12080/cuda-13000/dynamic-loading/cublaslt/cudnn. — [crates.io API](https://crates.io/api/v1/crates/cudarc)
- ort 2.0.0-rc.13 (2026-07-28, ONNX Runtime 1.28, CUDA 13 only) — secondary source; licence not verified. — [ai-tldr](https://ai-tldr.dev/tools/ort-rust/); [releases](https://github.com/pykeio/ort/releases)
- mold-ai-inference 0.21.0 (2026-08-05), MIT, 2,070 downloads. — [crates.io API](https://crates.io/api/v1/crates/mold-ai-inference)
- candle-flash-attn-v4: unreleased, MIT OR Apache-2.0 wrapper over BSD-3 FA4. — [repo](https://github.com/michaelfeil/candle-flash-attn-v4)
- stable-diffusion.cpp news 2026-10-16 (FLUX.3 Image Klein day-0); newfla/diffusion-rs bindings MIT. — [README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md); [newfla/diffusion-rs](https://github.com/newfla/diffusion-rs)

Low-quality sources deliberately excluded: rust4ai.com "Candle vs Burn" (May 2026, no methodology) and markaicode.com (2025) comparison tables.
