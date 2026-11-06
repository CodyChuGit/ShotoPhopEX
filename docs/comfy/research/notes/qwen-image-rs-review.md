# qwen-image-rs review (umag/qwen-image-rs @ 60d100c, Apache-2.0)

Read-only review of `C:\Users\5090\Documents\_eval\qwen-image-rs` on 2026-10-18, to inform the
`genai-native` fork (plan §3.2 layers, §7 Phase 2). Nothing was built or run. All numbers in §8
are the authors' own claims. Paths below are relative to the repo root; `file:line` cites the
checkout.

## 1. What it is and how it is built

- Single crate `qwen-image-rs` v0.0.0, edition 2021, `rust-version = "1.98"`, stable toolchain
  (`rust-toolchain.toml:5`). `Cargo.toml:6` says `MIT OR Apache-2.0`; `LICENSE`/`NOTICE`/README
  say Apache-2.0 only. Treat it as Apache-2.0 (the narrower claim) until the author clarifies.
- Tensor spine: **candle 0.11.0** (`candle-core`, `candle-nn`, `candle-transformers`; lock
  `Cargo.lock:291-359`), optional `candle-flash-attn 0.11`. CUDA via candle's cuda backend
  (`cudarc 0.19.9`, `Cargo.lock:581`); the `cudnn` feature pulls `cudarc` directly, default
  features off, only its `cudnn` module (`Cargo.toml:28`). A second `cudarc 0.17.8` sits in the
  lock from `candle-flash-attn`.
- Custom kernels are **not** cudarc PTX/JIT: they are raw `extern "C"` launchers linked as static
  libs. `build.rs` compiles each `.cu` group with the `cc` crate (`cc::Build::new().cuda(true)`,
  `build.rs:19-28, 40-48, 60-70, 82-91`) at build time, i.e. **nvcc is required on the build
  machine**; no prebuilt fatbins/PTX are shipped. Arch flag: a single
  `-arch=sm_{CUDA_COMPUTE_CAP}` defaulting to `89` (`build.rs:18,24`); no `-gencode` list, no PTX
  fallback. Flags: `-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda` (EVT device
  lambdas), `-diag-suppress=177` for the sage TUs. Each group links `-lcudart` (`build.rs:29`).
- **CUTLASS is neither vendored nor a submodule**: `CUTLASS_DIR` must point at a fresh clone's
  `include/` (`build.rs:12-17`, default `$HOME/dev_tmp/cutlass/include`). HANDOVER notes the
  flash-attn-vendored CUTLASS has a `matrix.h` bug CUDA 13.3 rejects (`HANDOVER.md:20-21`); the
  EVT path is documented as "CUTLASS 4.8 SM80 Epilogue Visitor Tree" (`HANDOVER.md:696`).
- cuDNN 9 comes from the NVIDIA pip wheel unpacked in user space; `build.rs:103-110` adds a
  link-search and a **`-Wl,-rpath`** link arg (Linux-only syntax).
- Four feature groups, all gated and additive: `cuda` (candle backend), `convrot` (CUTLASS INT8
  GEMM + quant kernels), `sage` (vendored SageAttention v1 + BSHD RoPE), `sage2` (SA2 sm89, own
  TU so the two vendored kernels' `PACK_SIZE_*` macros never meet, `build.rs:75-77`),
  `fusednorm` (norm/residual/VAE kernels), `cudnn`, `flash-attn`. Default build is CPU-only and
  compiles on macOS (`scripts/check.sh`). Recommended: `convrot,sage,fusednorm,sage2,cudnn`
  (`README.md:225`).
- **SageAttention source**: `kernels/sage/vendor/` is thu-ml/SageAttention, Apache-2.0 headers
  intact (`vendor/qattn/qk_int_sv_f16_sm80.cuh:2-14`, `qk_int_sv_f8_sm89.cuh:2-14`, from commit
  `d1a57a5` per `qk_int_sv_f8_sm89.cuh:20` and `NOTICE:13-20`). Sub-headers carry FlashInfer,
  vLLM and NVIDIA Apache-2.0 notices (`mma.cuh:3-5`, `reduction_utils.cuh:3-6`). Torch bindings
  stripped; one marked LOCAL PATCH (`attn_utils.cuh:130-136`, `kNoFill -> kFillZero`).
  `vendor/wgmma.cuh` (Hopper) is present but included by nothing (grep: no includer).
- Host workflow is Mac-edits-then-`git archive`-to-WSL via the author's private `swamp` tooling
  (`scripts/host.sh`, `agent-constraints/implementation-conventions.md:9-10`); irrelevant to us.

## 2. Architecture map

| Piece | File | Notes |
|---|---|---|
| DiT (QwenImage21Transformer2DModel) | `src/model/dit.rs` (1317 lines) | 32 single-stream blocks, 32x128, 3-axis interleaved RoPE `[16,56,56]` θ=1e4 (`dit.rs:246-250, 1025-1075`), block-causal attention (text causal prefix, image full) expressed as two attention calls (`dit.rs:728-744`), AdaLN as `(2, INNER)` modulation rows (row 0 image at t, row 1 text at t=0; `dit.rs:268-307`), ZeroCenterRMSNorm `weight+1` (`dit.rs:363-381`), per-head q/k RMSNorm, SwiGLU MLP ratio 3, `norm_out` scale-only + `proj_out`. Every linear is a `QLinear {Full, Quant(Q8_0), Convrot}` chosen by a prefix policy table (`dit.rs:15-100`). Text-to-image only: "no condition images, no KV cache" (`dit.rs:5-7`). |
| Text encoder (Qwen3-VL 8B, text tower only) | `src/model/text_encoder.rs` | Standard Qwen3 decoder: 36 layers, GQA 32/8, head 128, q/k RMSNorm, SwiGLU, RoPE θ=5e6 (`text_encoder.rs:64-79`). Returns **pre-final-norm** hidden states of all positions (`:8-12`); caller drops the 14 system-prefix tokens (`main.rs:598-616`). Naive S² attention with a dense causal mask (`:161-190`). **No vision tower, no DeepStack, no mRoPE** (`:3-6`): edit-with-reference is impossible as is. Two loaders: bf16 safetensors (`load`) and Q8_0 GGUF written by `prequantize-text` (`load_gguf`, `:281-352`). Chat template in `prompt::t2i_template` (`:378-390`). |
| VAE (AutoencoderKLQwenImage21, decoder only) | `src/model/vae.rs` (~970 lines) | All 2D convs (2.1's CausalConv3d subclasses Conv2d, `vae.rs:4-9`), channel RMSNorm (`gamma`), residual/attention/mid blocks, DupUp3D upsampler, `post_quant_conv` + `decoder.*` keys (`vae.rs:479-790`). NCHW or NHWC end to end (`ActLayout`, `:26-59`); cuDNN convs via own bridge; tiled decode with feather blend (`decode_tiled`, `:881`), `decode_auto` picks whole-image when the VRAM estimate fits (`:835`). `unpack_latents` does the 2.1 transpose-reshape and `z*std+mean` (`:950`). **Encoder absent** (no img2img/inpaint). |
| Scheduler | `src/model/scheduler.rs` | FlowMatchEuler with dynamic shift `mu = seq*m + b` (base/max 256/8192, 0.5/0.9), exponential time shift, stretch to terminal 0.02, Euler `x += dσ·v` (`scheduler.rs:41-69`). Pure, f64, unit-tested. |
| Sampling loop | `src/main.rs` `generate` (`:533-753`), `batch` (`:759`), `denoise` (`:1021`) | Tokenize → TE → drop prefix → DiT loop (`main.rs:673-690`) → VAE. CFG = two DiT forwards combined in f32 (`guided_noise_pred`, `:468-492`). Latents kept f32, cast to bf16 per step. Seeds via `dev.set_seed` + `Tensor::randn` per lane (`:667-672`). The loop is CLI-embedded, not a library API. |
| Weight loading | `src/loader/mod.rs` (90 lines) + candle `VarBuilder` | `WeightSet::resolve(dir)` lists `*.safetensors`/`*.gguf` directly under a dir, GGUF wins if present (`loader/mod.rs:47-78`). Actual loading is `VarBuilder::from_mmaped_safetensors` on the **official HF diffusers snapshot layout** (`processor/ text_encoder/ transformer/ vae/`, `docs/WEIGHTS.md:18-23`), sharded files fine, **diffusers tensor names** (`transformer_blocks.N.attn.to_q`, `img_mlp.gate_layer`, `txt_in.in_layer`, `time_text_embed.timestep_embedder.linear_1`, `modulation.1`, `norm_out.linear`, `proj_out`, `dit.rs:17-50`; `model.language_model.layers.N.self_attn.q_proj`, `text_encoder.rs:261-268`). Configs are **hard-coded defaults**, `config.json` is read only for VAE `latents_mean/std` (`main.rs:695-701`). **No ComfyUI repack naming, no key remap layer.** GGUF: only their own Q8_0 text-encoder file (`prequantize-text`, `main.rs:1891`) via candle's `quantized_var_builder`; no city96/Unsloth GGUF DiT support despite `loader/mod.rs:5-6` saying so. Own prequant format: `<prefix>.weight_i8` (U8) + `<prefix>.col_scale` (f32) in a safetensors file with `__metadata__` `qir.policy`/`qir.source` (`src/convrot_cache.rs:1-38`, `dit.rs:102-127`). |
| CUDA graphs | — | Not implemented: "candle 0.11 has no stream-capture hook" and per-step allocations move pointers (`README.md:293`, `HANDOVER.md:1064-1067`). |
| Memory | `src/device.rs` | `free_vram` (cuMemGetInfo), `retain_pool` raises the default mempool release threshold (`device.rs:87-119`), `pool_slack`. No arena, no offload; residency is "load the three models once" (`batch --resident`). |

Kernel bridges (candle `CustomOpN` over raw device pointers): `src/convrot.rs`, `src/sage.rs`,
`src/sage2.rs`, `src/fusednorm.rs`, `src/vae_fused.rs`, `src/cudnn_conv.rs`, `src/rope.rs`.
`src/layout.rs` is the view-offset guard every bridge must use (the B=1 off-prompt bug,
`HANDOVER.md:1148-1165`). `src/gemm_tiles.rs` is the CPU-testable tile/launch-plan table.

## 3. The kernels

| Kernel (file) | Fuses | dtype / arch assumptions |
|---|---|---|
| `int8_gemm_s32`, `int8_gemm_dequant_cfg_launch` (`kernels/convrot/int8_gemm.cu`) | CUTLASS 2.x `GemmUniversalAdapter` + Sm80 EVT: int8 x int8 → i32, epilogue `acc·s_row[m]·s_col[n]` → bf16 or f16 (`:107-161`). 5 tile configs `Cfg0..4` (`:235-240`), launch plans split M=4096+txt into head+tail launches (`src/gemm_tiles.rs:96-111`). | `ArchTag = Sm80`, `InstructionShape<16,8,32>` int8 `mma.sync`, `OpMultiplyAddSaturate` (`:81-83, 158`). A row-major, B (N,K) row-major, 16-B aligned, N%8. Workspace via `cudaMalloc` per call (`:218-227`, usually 0 bytes). |
| `quantize_rows_i8_k`, `quantize_rows_fused_k` (`quant_ops.cu:9-60`) | per-row amax → scale → int8 | bf16 in, generic. |
| `rotate_quantize_rows_k<CPW, Loader>` (`quant_ops.cu:159-208`) | in-register 256-wide Regular Hadamard (4 radix-4 stages via shfl, `:76-106`) + row amax + int8; `SwiGluRow` loader also does `silu(g)·p` in f32 (`:132-152`) | bf16, K%256, K ≤ 16384, 16-B aligned; warp shuffles only; generic sm_70+. |
| SageAttention v1 (`kernels/sage/sage_ffi.cu` + `vendor/qattn/qk_int_sv_f16_sm80.cuh`) | INT8 Q·K (per-block scales, CTA_Q 128 / CTA_K 64 / WARP 32x64), FP16 P·V, f32 accum, bf16 out; `RopeQuantInt8Kernel` fuses interleaved RoPE into the quant (`sage_ffi.cu:96-140`); BSHD strides. | `mma.sync m16n8k32 s8`, `m16n8k16 f16`, `ldmatrix`, `cp.async` → sm_80+ (`vendor/mma.cuh:29-33`, `cp_async.cuh:37`). head_dim 128 only. |
| SageAttention2 (`kernels/sage/sage2_ffi.cu` + `vendor/qattn/qk_int_sv_f8_sm89.cuh`) | INT8 Q·K per-thread scales + K smoothing, **FP8 e4m3 P·V** with per-channel V scales, fp32+fp16 two-level or fp32 accum (`sage2_ffi.cu:559-586`); own kernels: K-sum partials, RoPE(+per-head RMSNorm)+quant, V amax, V FP8 quant+transpose+16-token permute, all merged into a 3-launch task-table kernel (`Sage2QuantKernel`, HANDOVER "SA2 quant fusion"). | `mma.sync m16n8k32 ... e4m3.e4m3` f32/f16 accum and `cvt.rn.satfinite.e4m3x2` are gated `__CUDA_ARCH__ >= 890` (`vendor/mma.cuh:43-53`, `numeric_conversion.cuh:26-30`). V loads **unpredicated**: producer must zero-pad to 64 tokens (`HANDOVER.md:1145-1147`). head_dim 128; `cudaFuncSetAttribute` for ~48 KB dynamic smem. |
| `rope_i_bshd_kernel` (`sage/rope_bshd.cu`) | interleaved RoPE on (B,S,H,D), kept as oracle | bf16, generic. |
| `fused_norm_mod_kernel`, `fused_rmsnorm_scale_kernel`, `fused_rmsnorm128_kernel`, `fused_gated_residual_kernel`, `residual_norm_mod_kernel<HAS_RES,VEC>` (`fusednorm/fused_norm.cu`, `head_rmsnorm.cuh`) | LayerNorm(no affine)·(scale+1); RMSNorm·w (row-strided); 16-lane N=128 head norm; `h + tanh(g)·y` fused into the following LayerNorm+AdaLN reading the `(2,INNER)` modulation rows | bf16 in/out, f32 accum, CTA-per-row, dynamic smem = N f32 (4096 → 16 KB). Generic. |
| `vae_rmsnorm_k`, `vae_bias_residual_k` (`fusednorm/vae_norm.cu`); `vae_nhwc_rmsnorm_k`, `vae_nhwc_epilogue_k`, `vae_nhwc_upsample2x_k` (`vae_nhwc.cu`) | channel RMSNorm·γ(+SiLU)(+pending conv bias), bias+residual/DupUp gather epilogues, nearest 2x | bf16; deliberately reproduce candle's `fast_sum` bit-reversed tree with `__fmul_rn/__fadd_rn` for byte identity (`vae_norm.cu:5-18`). Generic. |
| cuDNN conv (`src/cudnn_conv.rs`, no .cu) | bf16 NCHW/NHWC implicit GEMM, `CUDNN_TENSOR_OP_MATH`, f32 accum, algo cache per shape | needs libcudnn 9 at link time. |
| `kernels/convrot/convrot_int8_gemm.cu` | **Unused reference**: SGLang's torch-coupled ConvRot kernel (Apache-2.0, `:1-14`), includes `torch/all.h` and sgl `cutlass_extensions` (`:25-54`). Not compiled by `build.rs`. Interesting because it already has Sm90 WGMMA / Sm100 tcgen05 / CC 12.x mma.sync dispatch (`:21-23`). |

**sm_120 (Blackwell GB202) assessment.** Consumer Blackwell is `sm_120`/`sm_120a`, which keeps
the Ampere/Ada `mma.sync`, `ldmatrix`, `cp.async` instruction set and Ada's FP8 `mma.sync`
shapes; it does **not** have Hopper `wgmma`/TMA-cluster features nor datacenter `tcgen05`. So:

- Everything above compiles and runs with `CUDA_COMPUTE_CAP=120` and a CUDA ≥ 12.8 nvcc: the
  `__CUDA_ARCH__ >= 800/890` guards are satisfied, no code is gated `== 89`. **Arch-flag change
  only, for correctness.** SGLang's own kernel confirms the CC 12.x parts take the mma.sync path
  (`convrot_int8_gemm.cu:23`).
- Performance is a different matter. The tile table (`gemm_tiles.rs`) and the "exact waves on
  128 SMs" reasoning (`HANDOVER.md:295-312`) are tuned for the 4090's 128 SMs / 72 MB L2; the
  5090 has 170 SMs and 96 MB L2, so plans must be re-benchmarked (`gemm-bench` exists for that).
  CUTLASS 2.x int8 on sm_120 is fine but leaves Blackwell's block-scaled NVFP4/MXFP8 paths
  (CUTLASS 3.x `Sm120` collectives) on the table; those are rewrites, not flags. `cudaFuncSetAttribute`
  smem budgets (~48 KB) are well under sm_120's 99 KB opt-in.
- The ConvRot cache's `qir.policy` tag has no GPU identity in it (`dit.rs:112-127`); the
  prequantized INT8 file is GPU-independent, which is good for us.
- `-arch=sm_120` emits only SASS for that arch; we want `-gencode arch=compute_120a,code=sm_120a`
  plus `sm_89` and a `compute_89` PTX fallback in one fatbin (plan §7.1), which `cc` can do with
  repeated `.flag("-gencode")`.

## 4. Parity method

- **Oracle**: `scripts/oracle.py` runs diffusers `QwenImage21Pipeline` bf16 (CPU offload) on 3
  fixed prompts, seed 42, 40 steps, 1024² (`oracle.py:21-28`), saving PNG + pre-VAE latent
  (safetensors) + manifest (`:49-82`). Per-component dumps (`dit_io.safetensors` with
  `hidden_states/encoder_hidden_states/timestep/output`, `embeds`, `input_ids`) are referenced
  everywhere (`main.rs:1112-1118`, `HANDOVER.md:1105`) but **their generator is not in the repo**
  (a hook into diffusers was done ad hoc) and the artifacts live only on the author's host
  (`~/dev_tmp/oracle_out/`, gitignored `oracle_out/`, `.gitignore:7`).
- **Metrics**: `scripts/compare_dit.py` — f64 overall cosine, MSE, mean per-token cosine over
  the last 4096 rows (`compare_dit.py:30-35`); `compare_latent.py` — cosine/maxabs/MSE with a
  0.99 self-consistency threshold (`:22-41`); `compare_png.py` — PSNR/maxabs on RGB(A) 8-bit
  (`:17-25`). Plus in-binary `*-test` verbs: bit-exactness vs host references (int8 GEMM, offset
  views, split launches), cosine vs f32/f64 references, NaN-poisoned-smem regression
  (`sage_poison_smem_launch`, `sage.rs:17`), and `cmp`/md5 byte identity between builds.
- **Gates used** (`agent-constraints/iteration-limits.md:19-83`): dit-forward `--convrot` cosine
  ≥ 0.99993 originally, relaxed to "within ~1e-5 of the prior build" once SA2 was accepted at
  0.9999; math-neutral changes must be byte-identical; VAE PSNR vs oracle ≥ ~50 dB; no denoise
  regression in a same-session A/B. They note the oracle cosine "reacts chaotically" to 1e-5
  upstream perturbations through 32 INT8/FP8 blocks (`:48-51`, `HANDOVER.md:981-995`), so they
  judge the whole config rather than one number.
- **Reference artifacts in the repo**: only images: `docs/first_generated.png`,
  `docs/generate_standalone.png`, `docs/batch/000-002.png`, `docs/comfyui_grid.jpg`. No tensors.
- **Reuse for us**: the method maps cleanly onto plan §12.1 (`gn-harness`): (a) regenerate
  `oracle_out` ourselves with `oracle.py` plus a small added hook that dumps `dit_io`/`embeds`
  (we need one extra ~30-line script; the capture pattern in `oracle.py:49-52` shows how);
  (b) adopt the three compare scripts verbatim (tiny, Apache-2.0); (c) adopt the discipline:
  byte-identity for math-neutral changes, cosine-vs-prior-build plus cosine-vs-oracle for math
  changes, PSNR for the VAE, same-seed B=1 vs batch-lane-0 as the layout-bug detector
  (`HANDOVER.md:1162-1165`), smem poisoning for partial tiles. Our reference should be ComfyUI
  (plan §7.4) in addition to diffusers; the same tensor dumps apply.

## 5. What is missing for us

- **Editing with reference images**: no vision tower / DeepStack / mRoPE in the encoder
  (`text_encoder.rs:3-6`); DiT forward takes only `(hidden_states, encoder_hidden_states,
  timestep, h, w)` (`dit.rs:1081-1088`); RoPE positions assume one image frame (`:1044-1054`);
  attention is a fixed text-prefix / single-image split (`:728-744`); no prefix KV cache
  (ComfyUI has one, `docs/COMFYUI.md:152-156`). Edit-2511-style conditioning is a new block set.
- **Masks / inpainting / Fill-Expand**: no VAE **encoder** (`vae.rs:1-2`), no img2img noise
  injection, no mask handling anywhere.
- **LoRA**: none (no `lora_A/B` loading, no fusing into the INT8 weights; ComfyUI requantizes).
- **Krea 2-shaped blocks**: single-stream only; no double-stream MMDiT, no 2D axial RoPE, no
  Turbo few-step schedule. Scheduler is diffusers' schedule, not ComfyUI's `simple` sigmas
  (`docs/COMFYUI.md:145-148`), so seeds will not match ComfyUI.
- **ComfyUI repack / GGUF loading**: no key maps, configs hard-coded (`config.rs` defaults), the
  repack's `qwen_image_2.1_int8_convrot.safetensors` layout is unknown to it (their own
  `weight_i8`+`col_scale` format is different), no third-party GGUF DiT (Q8_0 text GGUF only).
- **Cancellation / progress**: the loop is in `main.rs` with `tracing::info!` every 10 steps
  (`main.rs:687-689`); no callback, no cancel token, no per-step hook; CFG does two forwards
  with no way to stop between them.
- **Memory management**: no VRAM budget beyond `free_vram` + `retain_pool`; no offload, no
  streaming weights, no multi-model arena; workspace `cudaMalloc` inside the GEMM launcher
  (`int8_gemm.cu:220-222`); candle's default mempool.
- **Windows build**: nothing is Windows-specific but several things are POSIX-only: `-Wl,-rpath`
  (`build.rs:110`), `$HOME` defaults (`build.rs:15,106`, `convrot_cache.rs:69` — on Windows
  `HOME` is usually unset, so the cache silently falls back to quantize-on-load, `:343`), all
  scripts are bash, cuDNN comes from a Linux wheel. The `cc` crate drives nvcc on Windows too
  (needs MSVC host compiler); CUTLASS EVT headers compile under MSVC+nvcc with caveats. Expect a
  day of build-script work, not a port. WSL2 numbers are the only ones measured.
- **Determinism**: claimed byte-reproducible at fixed seed and B=1 == batch lane 0
  (`README.md:86`, `HANDOVER.md:86-93`): INT8 accumulation is exact, no split-K/stream-K, fixed-
  order reductions (`sage2_ffi.cu:90-93`). Caveats: cuDNN algo heuristics (locked per shape but
  may change across cuDNN versions, `HANDOVER.md:1370-1373`), the seed→noise path is candle's
  CUDA RNG (not ComfyUI's CPU torch RNG), and the prequant file header is HashMap-ordered
  (`HANDOVER.md:527-529`).

## 6. Code quality and licence hygiene

- **unsafe**: 69 occurrences in 10 files, all at the FFI/device-pointer boundary (`convrot.rs` 14,
  `sage.rs` 13, `vae_fused.rs` 10, `fusednorm.rs` 9, `sage2.rs` 8, `main.rs` 6 for
  `from_mmaped_safetensors`, `device.rs` 3 driver calls, `cudnn_conv.rs` 3, `rope.rs` 2,
  `convrot_cache.rs` 1). Each bridge validates shapes, dtypes, alignment and the view offset
  before the call and checks the launcher's return code. Pattern is consistent and auditable.
- **unwrap/expect/panic**: 109 matches, ~95 inside `#[cfg(test)]`. Production ones:
  `scheduler.rs:61`, `rotation.rs:55`, `vae.rs:717,741`, `fusednorm.rs:109,209`,
  `convrot.rs:755,825,845` (all `dims.last().unwrap()` on non-empty shapes), `main.rs:725,729`
  (`expect` after explicit validation). No `panic!`/`todo!`/`unimplemented!`. Fine for a CLI;
  under AGENTS.md's no-unwrap rule they become `?` with a shape error.
- **Tests**: 53 `#[test]` across 9 files (CPU: scheduler, rotation, layout, gemm_tiles,
  convrot_cache, seed, text template, VAE sim of the fast_sum tree, DiT policy tables) plus the
  GPU self-test verbs. clippy `-D warnings` and fmt enforced (`scripts/check.sh`). No CI config.
- **NOTICE** (`NOTICE:1-30`): claims the port is "reverse-engineered from the diffusers
  implementation" (diffusers is Apache-2.0: fine), vendors thu-ml SageAttention (Apache-2.0)
  with the patch disclosed, uses CUTLASS and cuDNN (not vendored), candle, and cites pytorch/ao
  for the ConvRot scheme. Missing from NOTICE: the vendored **SGLang** file
  `kernels/convrot/convrot_int8_gemm.cu` (Apache-2.0 header present in the file itself) and the
  FlashInfer/vLLM/NVIDIA sub-notices inside `vendor/` (present in the files). We should add both
  when we fork, and drop the unused SGLang file or keep it clearly as reference.
- **Restricted sources**: nothing from Qwen's own repo (weights not shipped; the research licence
  applies to weights, which is our BYOM user's business). Model constants come from the public
  `config.json` files. The Comfy-Org repack bench (`bench/comfyui/`) only downloads weights at
  run time. `docs/WEIGHTS.md` and README state the research-only status correctly.
- **Licence mismatch**: `Cargo.toml:6` `MIT OR Apache-2.0` vs `LICENSE`/README Apache-2.0. For
  the fork, keep Apache-2.0 attribution for this code; our crates can stay `MIT OR Apache-2.0`
  only for code we write (plan §12.8).
- Docs are unusually thorough (HANDOVER is a 1380-line lab notebook with every measurement); the
  code is well commented and the Rust is idiomatic candle. Weakness: `main.rs` is a 2000-line
  monolith holding the pipeline, and everything is specialised to one shape (head 128, 1024²,
  B≤4) with `const`s rather than config.

## 7. Restructuring proposal

Order follows plan §7.1; each step compiles and the parity verbs still run before the next.

1. **`gn-kernels`** (first, because everything else only needs its trait):
   - `kernels/convrot/int8_gemm.cu`, `quant_ops.cu` → `Kernels::{int8_gemm_dequant,
     rotate_quantize, swiglu_rotate_quantize}`; `src/gemm_tiles.rs` becomes the CUDA impl's plan
     table, re-tuned per GPU (keyed by SM count).
   - `kernels/sage/{sage_ffi.cu, sage2_ffi.cu, rope_bshd.cu, rope_pair.cuh, vendor/}` →
     `Kernels::{attn_int8_f16, attn_int8_fp8, quant_layer}`; drop `vendor/wgmma.cuh`.
   - `kernels/fusednorm/*` → `Kernels::{residual_norm_mod, rmsnorm_scale, vae_rmsnorm,
     vae_epilogue, upsample2x}`.
   - `src/cudnn_conv.rs` → `Kernels::conv2d_bf16` (keep the algo cache; add a CUTLASS/implicit
     fallback later so Windows builds without the cuDNN wheel).
   - `src/layout.rs`, `src/device.rs` (pool retention, `free_vram`) → the device/arena module.
   - `build.rs` → one `-gencode` fatbin build (`sm_120a`, `sm_89`, `compute_89` PTX), prebuilt
     fatbins committed, nvcc optional behind a `rebuild-kernels` feature.
   - Keep the `*-test` verbs as `gn-kernels` integration tests (they are the bit-exactness suite).
2. **`gn-blocks`**: from `dit.rs` extract `QLinear` → a `Linear` enum over kernel-backed
   precisions; `residual_norm_mod` / `Modulation` → `adaln.rs`; `Attention::{project,
   attend_bshd}` → `attention.rs` (block-causal joint attention as one block, mask-free split as a
   strategy); `SwiGlu` → `mlp.rs`; `rope_cos_sin` → `rope.rs` (3-axis interleaved; add 2D axial
   for Krea); `TimestepEmbed` → `timestep.rs`; `scheduler.rs` → `flow_match.rs` (add ComfyUI
   `simple`/shift and Turbo); `text_encoder.rs` → `encoders/qwen3_text.rs` (then add the vision
   tower beside it); `vae.rs` → `vae/qwen21.rs` + generic `tiled_decode.rs`; `rotation.rs` →
   `quant/convrot.rs`. Each gets a tiny committed reference tensor test (plan §7.5).
3. **`gn-models/qwen_image_21.rs`**: the `QwenImageDit::load/forward` wiring, the
   `BLOCK_LINEARS`/`TAIL_LINEARS` precision policy, the prompt template and the 14-token drop,
   `unpack_latents`, the `generate` sequence from `main.rs:605-692` turned into a library
   `Pipeline` with a per-step callback and a cancel token. Target ≤ 600 lines.
4. **`gn-weights`**: `loader/mod.rs` (trivial) + `convrot_cache.rs` (the prequant cache, cache
   root made platform-aware) + a new key-map layer: official diffusers names (what the code uses
   today) ↔ ComfyUI repack ↔ GGUF; the repack's INT8 ConvRot tensors should be loadable straight
   into `ConvRotLinear::from_prequantized` once we confirm their rotation/scale conventions match
   (same pytorch/ao scheme, `NOTICE:28-29`), which would skip the 26 s quantize.
5. **`gn-harness`**: `scripts/oracle.py` + the three compare scripts + a new `dump_dit_io.py`.
6. `bench/`, `agent-constraints/`, `.attestations/`, `scripts/host.sh`, `setup-host.sh`,
   `convrot_int8_gemm.cu`: not carried (keep in git history; cite in NOTICE).

**Five biggest risks**

1. **Blackwell performance, not correctness**: tiles/plans/L2 reasoning are 4090-specific; the
   INT8 `mma.sync` path may leave a lot of the 5090 idle versus FP8/NVFP4 CUTLASS 3.x kernels.
   Budget a re-tuning pass and expect kernel rewrites for the top GEMMs.
2. **candle 0.11 as the spine**: no CUDA graphs, per-step allocations, CustomOp limited to 3
   inputs (they already escape via `storage_and_layout`, `HANDOVER.md:948-951`), fragile
   `device_ptr` guard lifetimes. The `Kernels` trait must own streams and buffers so candle can
   be swapped later.
3. **Windows + nvcc + CUTLASS EVT + cuDNN**: untested by upstream; MSVC host-compiler quirks with
   `--expt-extended-lambda`, rpath-less cuDNN discovery, `HOME`-less cache paths. Prebuilt
   fatbins reduce but do not remove this.
4. **Scope gap to the product**: encoder vision tower, VAE encoder, masks, LoRA, double-stream
   blocks, repack/GGUF maps are all new code; qwen-image-rs gives us t2i and the fast path, not
   the editing features Phases 3-4 need.
5. **Parity reference drift**: the oracle tensors are not in the repo and the schedule/RNG differ
   from ComfyUI; until `gn-harness` regenerates both references, "parity" claims cannot be
   checked on our machine. The SA2 FP8 path is accepted at cosine 0.9999, below their own
   earlier gate — we must decide our own threshold (plan §12.1).

## 8. Numbers they claim (self-reported, verbatim)

- "Denoise: 0.62 → 0.148 s/step (~4.2×) at 1024². VAE decode: 1.15 → 0.13 s (ComfyUI: 0.20 s).
  A resident pipeline renders a 1024² image every ~6.1 s." (`README.md:9-10`)
- "Measured 2026-10-02 on HEAD (`65e5e7f`), RTX 4090 24 GB, WSL2, CUDA 13.3, 1024², 40 steps,
  guidance 1." (`README.md:20-21`)
- `batch --resident`, 4 prompts: "37.4 s | 6.07 s steady | 20.2 GB"; `generate`, 1 image warm:
  "18.3–18.7 s"; cold: "27.0 s" (`README.md:28-31`)
- Per step: "INT8 GEMMs (CUTLASS, dequant in the epilogue) | 95.6 | 65%"; "SageAttention2 (INT8
  Q·K, FP8 P·V) | 18.7 | 12%" (`README.md:59-60`)
- Quality: "Text encoder (Qwen3-VL, bf16) | per-token cosine 0.9993 (Q8_0 GGUF: 0.9977)"; "DiT
  single forward, recommended build (`--convrot`, SA2) | cosine 0.999911"; "no `--convrot` |
  cosine 0.999945"; "VAE decode | PSNR 56.4 dB"; "same seed → byte-identical image" (`README.md:81-86`)
- vs TensorRT: "qwen-image-rs ... 0.148 | 1.00× | 0.999911"; "TensorRT 11.3 INT8 SmoothQuant
  (ModelOpt, α=0.8) | 0.216 | 1.46× slower | 0.998214"; "diffusers eager, bf16 | 0.506 | 3.43×"
  (`README.md:92-98`; detail `docs/TENSORRT.md:11-21`)
- vs ComfyUI: "ComfyUI INT8 ConvRot + `--use-sage-attention --fast --disable-dynamic-vram`
  (best) | 0.156 | 6.94 s | 21.4 GB | 1.06× / 1.12× slower"; "ComfyUI stock bf16 | 0.419 |
  19.54 s" (`README.md:110-115`; `docs/COMFYUI.md:14-27`)
- Scoreboard, bf16+FlashAttention-2 baseline "0.62" down to row-split plans "**0.1477** (A/B
  0.1524 → 0.1477, −3.1%)" (`README.md:239-255`; same table `HANDOVER.md:45-65`)
- VAE: "NHWC end to end ... **0.133 s** (tiled 32: 0.38 → 0.225 s) | **0.136 s**"
  (`README.md:266`); ComfyUI decoder "203 ms" vs ours "**133 ms**" (`docs/COMFYUI.md:184-185`)
- Loads: first `--convrot` run "builds the prequantized INT8 DiT once (~26 s, a 7.1 GB cache
  entry" (`README.md:162-163`); "cold 7.00 → **2.73 s**, warm 2.08 → **1.12 s**" (`HANDOVER.md:523-524`)
- ConvRot linear alone: "1.59× faster than bf16 (1.65 vs 2.62 ms), cosine 0.99993" (`docs/PHASES.md:40`)
- INT8 GEMM efficiency: "580–610 of ~660 INT8 TOPS kernel time" (`README.md:59`)
- Batch: "B=4 is the efficient limit on 24 GB; from B=5 each image costs more" (`README.md:53`)
- Phase-4 bf16 e2e: "40-step denoise → latent cosine 0.9977 → VAE → image PSNR 30.85 dB vs
  oracle" (`docs/PHASES.md:32`); with FlashAttention "final latent 0.9977→**0.9998**" (`:36`)

None of these were reproduced here; plan §5.2 ("bring up qwen-image-rs on the 5090") is where
they get checked.
