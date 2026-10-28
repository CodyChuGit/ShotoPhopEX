# Qwen-Image-Layered 3:4 performance and parity — 2026-10-28

Both requested targets are met by the production `LayeredPipeline::split` path on
this RTX 5090. Changes are uncommitted. ComfyUI was not started.

## Request and measurement conditions

- Input: `C:/Users/5090/ComfyUI/photocraft-tests/bench/layered/denim-3x4.png`,
  found with the earlier benchmark and matching saved ComfyUI layers. PhotoCraft's
  Lanczos-3 resampler supplies exactly 864x1152 or 480x640 pixels.
- Three layers, 20 Euler/simple steps, CFG 2.5, seed 7, empty negative text.
  Positive text: `separate this image into layers: the background and each foreground subject on its own layer`.
- Models: the owner's `qwen_image_layered_fp8mixed.safetensors`,
  `qwen_2.5_vl_7b_fp8_scaled.safetensors`, and `qwen_image_layered_vae.safetensors`.
- One GPU process at a time; no compilation during the final timing pass.
  Every split loads the VAE, encoder and DiT again. There is no retained model,
  conditioning or inference-result cache. The three BF16 GEMM plans and their
  32 MiB workspace live only until the last DiT linear is dropped.
- Build: `layered-build.cmd`, CUDA 13.4 / `CUDARC_CUDA_VERSION=13030`, sm_120,
  and `cuda,fp8,flash,convrot,sage,sage2,fusednorm`; isolated
  `target/layered-size-perf` inside this worktree. All custom kernels compile
  in `build.rs`; no runtime NVRTC was added.
- The writable CUDA driver kernel cache was populated. OS file-cache state can
  affect loading. An empty driver cache or a true disk-cold startup is not claimed
  to meet these times. Mapping faults are included and now overlap conditioning.
- Baseline production code is repository HEAD
  `766ecb030c59b6a37426f433d48176598af7a413`, rebuilt here before production
  edits. Only the benchmark's input/export dimensions were generalized first;
  its binary was preserved as `target/layered-before.exe`.

## Matched production timings

Seconds, with `QIR_LAYERED_TIMING=1`, `QIR_PROFILE` off and diagnostic step
synchronization off. Phase boundaries synchronize the GPU. End-to-end includes
CUDA/pipeline/tokenizer initialization and input resizing; PNG writing and output
export resizing follow the timer. `split` alone excludes that initialization.

| Phase | 480x640 before | 480x640 after | 864x1152 before | 864x1152 after |
|---|---:|---:|---:|---:|
| Reference VAE load + encode | 0.36 | 0.39 | 0.53 | 0.59 |
| Encoder load + both prompts | 3.16 | 3.30 | 3.22 | 3.44 |
| DiT load, including remaining prefetch wait | 8.83 | 4.50 | 8.62 | 4.79 |
| All 20 denoising steps | 34.00 | 24.89 | 160.78 | 119.07 |
| Mean per denoising step | 1.7000 | 1.2445 | 8.0390 | 5.9535 |
| VAE load + three RGBA decodes | 0.56 | 0.58 | 1.27 | 1.22 |
| Production `split` alone | — | 33.660 | — | 129.113 |
| End-to-end, including initialization | **47.629** | **34.353** | **175.216** | **130.004** |
| Owner's supplied ComfyUI end-to-end target | 36 | 36 | 140 | 140 |

The matched reductions are 27.9% at 480x640 and 25.8% at 1 MP. Relative to the
supplied ComfyUI measurements, the final runs are 1.65 s and 10.00 s faster.
The owner's earlier app baselines were 44 s and 179 s; they are context, not
substitutes for the matched native baseline above. ComfyUI's times were supplied
by the owner and were not remeasured.

A separate final 480x640 process with **all timing flags off** measured
**34.352 s** including initialization, **33.657 s** inside `split`. An earlier
full run of the final implementation measured 35.132 s / 34.508 s respectively.
No final quiet 1 MP run was needed to establish the phase-timed 130.004 s result.
The frontend/app worker was not rebuilt or timed through the UI in this pass.

Prefetch starts inside the timed split, before reference encoding. It opens and
faults the exact mapping later used for upload while conditioning runs; the
worker is joined before DiT construction and stopped/joined on other exits.
Thus the lower DiT-load entry reflects overlap, not excluded work or a GPU model
cache. The encoder entry can rise slightly because its reads overlap prefetch.

## What ComfyUI actually runs

Local ComfyUI checkout: `b0b743566f65daafc423b4fea8a2fbda94b3384a`.
Paths below are relative to
`C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`.

### Linears under `--fast fp8_matrix_mult`

The actual checkpoint contains **839** `.comfy_quant` records; every one is
`{"format":"float8_e4m3fn","full_precision_matrix_mult":true}`.

- `comfy/ops.py:1793` checks `model_config.quant_config` first and returns
  `mixed_precision_ops` at line 1798. The fast-FP8 fallback follows this return,
  so that flag does not override this mixed file's per-layer records.
- `comfy/ops.py:1203` loads `full_precision_matrix_mult`, and lines 1204–1205
  copy it to `_full_precision_mm`. `comfy/ops.py:1445` excludes such layers from
  input quantization / quantized GEMM.
- `comfy/ops.py:1415` dequantizes a quantized weight when full precision is set.
  `comfy/ops.py:1397` calls `torch.nn.functional.linear(input, weight, bias)`.
  `comfy/quant_ops.py:90` derives its FP8 layout from comfy-kitchen;
  installed `python_embeded/Lib/site-packages/comfy_kitchen/tensor/fp8.py:70`
  dispatches weight dequantization. The eager implementation at
  `comfy_kitchen/backends/eager/quantization.py:64` casts the weight and scale
  to the compute dtype before multiplication.

Therefore **all 839 FP8-stored DiT linears dequantize and GEMM in bf16**, with
unquantized activations. There is no hardware FP8 GEMM in this DiT under the
specified flag. The remaining seven linear calls use BF16 weights directly:
`img_in`, `txt_in`, timestep `linear_1` and `linear_2`,
`transformer_blocks.0.img_mod.1`, `norm_out.linear`, and `proj_out`.
Normalization weights are not counted as linears.

Native retains that behavior. Only large image-stream GEMMs (360 calls per
forward at these sizes) select BF16 cuBLASLt; the 479 other FP8-stored calls keep
the existing dequantize/BF16 path. Bias and activation rounding remain separate.
Full-precision-only constructors keep unused FP8 GEMM scale pointers on the CPU;
existing shared FP8 constructors and Krea 2's hardware FP8 behavior are unchanged.

### Attention

- `comfy/ldm/qwen_image/model.py:197` calls `optimized_attention_masked` on joint
  text/image Q/K/V after RoPE. The saved ComfyUI log explicitly says
  `Using pytorch attention`.
- `comfy/ldm/modules/attention.py:904` selects `attention_pytorch`; its
  implementation is at line 579 and calls `comfy.ops.scaled_dot_product_attention`.
- `comfy/ops.py:70` prefers PyTorch Flash, then cuDNN, efficient, and math SDPA;
  line 84 installs that order. `comfy/text_encoders/qwen_image.py:83` removes
  an all-valid text mask, so normal separate CFG calls need no mask.
- An isolated operator probe with the installed **Torch 2.14.0+cu130**, without
  importing or starting ComfyUI, used that priority and BF16 BHSD tensors with
  24 heads, D=128, and representative ragged sequence lengths 6,027 / 19,467.
  At **both** lengths the profiler selected
  `aten::_scaled_dot_product_cudnn_attention`, specifically
  `cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_1_64x64x128_4x1x1_cga1x1x1_kernel0_0`.

ComfyUI consequently uses cuDNN SDPA on this Windows Torch installation, not Sage
or Sage2. Its configured preference for PyTorch Flash is not proof that that
backend is selected. Native keeps its existing nonquantized attention arithmetic:
BF16 Q/K, f32 scores/statistics/accumulators, and the original clamped FP16 PV
operands. It does not introduce INT8/FP8 attention or a cuDNN dependency.

## Kept changes and measurements

1. **Attention shared-memory reuse.** After Q fragments reach registers, Layered's
   64x64 tile reuses that staging area for K. Per-CTA shared memory falls from
   51 KiB to 34 KiB, allowing two CTAs per sm_120 SM. The shared kernel's default
   compilation and public callers retain the original allocation. A seven-line
   Layered translation unit selects the alternative at build time, without
   duplicating the algorithm. Kernel measurements at about 19.5k tokens improved
   from 33.8–34.4 ms to about 24.3–24.6 ms per CFG lane, with zero output error.
2. **BF16 cuBLASLt for large image matrices.** Warm shape probes at M=12,000 and
   M=38,880 found useful gains, especially Q/K/V/output and MLP-up. For example,
   M=38,880 K=3,072 N=12,288 measured 15.8–18.3 ms with cuBLAS versus about 12.0 ms
   with Lt. These probes reuse Lt's output allocation and are not end-to-end
   speedup claims. Production allocates its real outputs and caches only the
   three plans plus workspace for the split. A fallible owner avoids cudarc's
   high-level handle constructor/destructor unwraps. No fused bias epilogue
   removes the existing bf16 GEMM rounding boundary.
3. **Layered-only fusions.** Q/K head RMSNorm writes BHSD directly, retaining the
   bf16 normalized-value rounding before the learned weight. LayerNorm/modulation
   retains Candle's reduction and bf16 norm/scale boundaries. Fresh GEMM outputs
   receive bias and optional MLP GELU in place, preserving Candle's bf16 GELU
   arithmetic. Tests require exact agreement, including nonzero view offsets.
   These changes remove transpose passes, normalized/broadcast temporaries, and
   large bias/GELU allocations. VAE and sampler math are unchanged.
4. **Checkpoint mapping prefetch.** CPU page faults overlap reference/text work.
   Isolated reads of the 20.53 GB checkpoint and transfer probes established the
   loading bottleneck; a pinned 20 GiB H2D probe took 0.65 s. Reusing a prefaulted
   mapping reduced matched DiT-load phases from 8.83 / 8.62 s to 4.50 / 4.79 s.
   Cancellation, I/O errors, and last-owner workspace cleanup are tested.

Not kept: 128-query/128-key tile variants (no better overall tradeoff; changing
key tiling also changed output rounding), converting V once globally (no useful
kernel gain), alternate packed/individual uploads, and double-buffered pinned
uploads (no repeatable end-to-end benefit). The interrupted Sage switch was
read but not copied. FP8 GEMM was excluded by the checkpoint/ComfyUI semantics.

### Diagnostic profiles

`QIR_PROFILE=1` synchronizes every scoped operation and changes allocator/clock
behavior. These two-step profiles locate costs; the 20-step production timings
above decide whether the targets are met. Second-step seconds:

| Scoped work | 480 before | 480 after | 1 MP before | 1 MP after |
|---|---:|---:|---:|---:|
| All FP8-stored linears: dequantization + BF16 GEMM | 0.838 | 0.808 | 3.625 | 3.529 |
| Joint attention, 60 calls over both CFG lanes | 0.470 | 0.306 | 4.507 | 2.907 |
| Q/K RMSNorm plus input RMSNorm | 0.082 | 0.016 | 0.261 | 0.048 |
| Original LayerNorm, 241 calls | 0.031 | — | 0.140 | — |
| Fused LayerNorm/modulation, 241 calls | — | 0.030 | — | 0.120 |
| Remaining add-product work | 0.082 (421 calls) | 0.032 (180) | 0.883 (421) | 0.265 (180) |
| RoPE, unchanged math | 0.014 | 0.014 | 0.043 | 0.041 |
| Complete diagnostic step | 1.94 | 1.36 | 13.41 | 8.05 |

After linear totals combine 479 existing-path calls with 360 Lt calls. Newly
scoped bias/GELU work is 0.064 / 0.194 s; baseline bias and some copies were
outside the old scopes. The table is not an exhaustive or additive wall-time
accounting. Production mean step times are 1.2445 / 5.9535 s, not the diagnostic
values. Final text and VAE phase times above include their model loading.

## Image gate

Saved ComfyUI references were found under
`C:/Users/5090/ComfyUI/photocraft-tests/bench/layered/denim-1mp/` and
`denim-640/`; no substitute native-only gate was necessary.

**All six raw native RGBA layers are byte-identical before versus after** at the
actual graph dimensions: RGB MAE 0, alpha MAE 0, and alpha IoU 1 for every layer.
This holds over all pixels and at any visibility threshold.

For the supplied ComfyUI document exports, native is resized with PhotoCraft's
resampler to 832x1109, clearing RGB at exactly zero alpha as document export does.
The 480px reference's second export is 832x1108; it is placed at the top-left of
the common transparent canvas, without resampling it. Visible RGB and silhouette
IoU below use the union of pixels with alpha >127. RGB MAE is on a 0–255 scale.

| Graph size | Layer | RGB MAE before = after | Alpha MAE before = after | Alpha IoU before = after |
|---|---:|---:|---:|---:|
| 864x1152 | 1 | 0.913724 | 0.005228 | 1.000000 |
| 864x1152 | 2 | 1.424933 | 0.305233 | 0.995284 |
| 864x1152 | 3 | 0.990136 | 0.266824 | 0.999487 |
| 480x640 | 1 | 1.061241 | 0.005404 | 1.000000 |
| 480x640 | 2 | 0.000000* | 0.006193 | 1.000000* |
| 480x640 | 3 | 1.124498 | 0.102426 | 0.999097 |

*Both small-size second layers have no pixels above the visibility threshold;
empty-set RGB MAE is reported as 0 and silhouette IoU as 1.*

The visibility convention matters: the saved exports already have large RGB
differences in near-transparent pixels. At alpha >0, the 1 MP foreground RGB
MAEs are 51.884 / 24.743; those are unchanged from baseline too. The checker
reports full-canvas RGB as well and accepts `--visible-alpha-threshold 0` to
inspect that convention. The unconditional byte-identical native-before/after
gate establishes no parity loss independently of this export convention.

## Validation and reproduction

- 102 library tests pass, with two manual GPU benchmarks ignored by the ordinary
  suite. This includes shared CUDA kernels, mask handling, exact fused rounding,
  FP8 full-precision construction, GEMM offsets/plan reuse/workspace lifetime,
  pool restoration, and prefetch cancellation/I/O errors.
- Nine integration tests pass across `layered_pipeline`, `qwen25_text`, and
  `qwen_image_dit`; six installed/slow tests remain ignored by that command.
- Installed 20-step runs at both sizes pass all layer comparisons above.
- Release Clippy, all targets and requested features, passes with only the existing
  `lora.rs:188` type-complexity warning; existing nvcc compiler-detection and
  unused Cargo-patch warnings remain. CPU-only release `cargo check` passes.
- Edition-2021 rustfmt checks on touched Rust files and `git diff --check` pass.
  No unwrap/expect/panic was added outside tests. Krea 2 / Qwen-Image-2.1 dispatch,
  kernels and existing FP8 constructors retain their behavior; shared Flash's
  original branch is explicitly checked against the Layered branch and f32 math.

From this worktree's root, PowerShell:

```powershell
$env:CARGO_TARGET_DIR = "$PWD/target/layered-size-perf"
& crates/qir/tools/layered-build.cmd test -p qwen-image-rs --release --offline --features cuda,fp8,flash,convrot,sage,sage2,fusednorm --lib --test layered_pipeline --test qwen25_text --test qwen_image_dit --no-run
& crates/qir/tools/layered-size-bench.ps1 -Width 864 -Height 1152 -Label owner-1mp
& crates/qir/tools/layered-size-bench.ps1 -Width 480 -Height 640 -Label owner-small
& crates/qir/tools/layered-size-bench.ps1 -Width 480 -Height 640 -Label owner-quiet -Quiet
# Diagnostic only: -Steps 2 -Profile. Do not compare its end-to-end time to 20 steps.
```

The script selects the portrait and export sizes above; `-InputPath`,
`-ExportWidth`, `-ExportHeight` and `-Binary` override them. Its writable driver
cache is `target/layered-cuda-cache`. Keep only one GPU process active.

```powershell
$py = 'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
& $py -I crates/qir/tools/layered_compare.py target/layered-size-matched-after-864x1152 --comfy target/layered-size-matched-before-864x1152 --native-prefix raw --reference-prefix raw --rgb-limit 1.5
& $py -I crates/qir/tools/layered_compare.py target/layered-size-matched-after-864x1152 --comfy C:/Users/5090/ComfyUI/photocraft-tests/bench/layered/denim-1mp --rgb-limit 1.5
& $py -I crates/qir/tools/layered_compare.py target/layered-size-matched-after-480x640 --comfy C:/Users/5090/ComfyUI/photocraft-tests/bench/layered/denim-640 --pad-canvas --rgb-limit 1.5
```

`QIR_LAYERED_GEMM=cublas` selects the prior GEMM route for an ablation;
`QIR_LAYERED_PREFETCH=0` disables mapping prefetch. Other inherited Layered/pool
flags should be unset for reproduction. Microbenchmarks are ignored library
`layered_tiles_match_and_time` and `layered_gemm_shapes`; run exclusively with
`--ignored --nocapture --test-threads=1`.

Evidence is in `target/layered-size-logs/matched-before-*`, `matched-after-*`,
`quiet-final-480x640.log`, `profile-before-*`, and `profile-final-*`; parity JSONL
is `target/parity-{raw,comfy}-*.jsonl`. Raw and export PNGs are in
`target/layered-size-matched-{before,after}-<size>/`. Operator/microbench logs
include `target/sdpa-probe.log`, `gemm-shapes.log`, `kernel-tile-6464.log`,
`bandwidth-large.log`, and `upload-strategies.log`. Validation logs are
`target/final-all-unit.log`, `final-validated-*.log`, `final-clippy-checked.log`,
and `final-cpu-check.log`. Generated artifacts stay ignored.

## Files changed in this pass

- `crates/qir/build.rs` — build the Layered attention entry point.
- `crates/qir/kernels/flash/flash_fwd.cu` — compile-time query-staging reuse option;
  default shared behavior is unchanged.
- `crates/qir/kernels/flash/layered_flash_fwd.cu` — select that option for Layered.
- `crates/qir/kernels/fusednorm/layered_rope.cu` — Layered head norm, modulation,
  and bias/GELU kernels alongside the existing unchanged RoPE kernel.
- `crates/qir/src/flash.rs` — private Layered entry point and exact regression checks.
- `crates/qir/src/fp8.rs` — host-only metadata constructor for full-precision callers
  and its CUDA equivalence test; existing constructors are unchanged apart from formatting.
- `crates/qir/src/layered_files.rs` — mapping prefetch and full-precision construction.
- `crates/qir/src/layered_pipeline.rs` — bounded prefetch worker and exit-path tests.
- `crates/qir/src/model/qwen_image_dit.rs` — Layered fusions and BF16 GEMM selection.
- `crates/qir/src/model/layered_gemm.rs` — fallible cuBLASLt owner, cached plans,
  stream-scoped workspace, and offset/lifetime tests.
- `crates/qir/tests/layered_pipeline.rs` — size/export controls and separate split timer.
- `crates/qir/tools/layered-build.cmd` — honor an explicit isolated target directory.
- `crates/qir/tools/layered-size-bench.ps1` — reproducible size/profile/quiet harness.
- `crates/qir/tools/layered_compare.py` — explicit visibility, raw/export prefixes,
  canvas handling and numerical gates.
- `crates/qir/tools/layered_size_performance.md` — this report.
- `crates/qir/tools/layered_performance.md` — link from the earlier 640px report.

## Remaining work and limits

Relink/rebuild the consuming app worker with this worktree and measure its UI/IPC
latency; this pass exercised the production split directly. No app installation,
ComfyUI service, deployment or commit was changed. The original 640x640 workload
was not rebenchmarked in this pass, and other seeds/layer counts/sizes were not
gated. Full Krea 2 / Qwen-Image-2.1 image runs were not repeated; their existing
paths are unchanged and the shared-kernel unit suite passes. Cold-cache loading,
loading variance, larger workloads and lower-memory hosts remain separate work.
