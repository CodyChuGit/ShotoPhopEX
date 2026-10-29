# Edit-2511 performance and parity — 2026-10-28

All eight requested warm native cases are faster than the supplied ComfyUI
measurements. All 16 native images and all six Layered layers are byte-identical
to their pre-change references. Changes remain uncommitted; no app deployment
or ComfyUI service change was made. Baseline HEAD is
`d40c536d42cfbb7ce976adea721a4cb73501315c`.

## Installed ComfyUI route

All paths in this section are relative to
`C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`.

The installed `qwen_image_edit_2511_fp8mixed.safetensors` contains 839
`.comfy_quant` records, all
`{"format":"float8_e4m3fn","full_precision_matrix_mult":true}`.
`comfy/ops.py:1794-1798` returns mixed precision operations before considering the
fast-FP8 fallback. Lines 1203-1205 copy each record's full-precision flag to the
linear. Lines 1445-1450 exclude these linears from activation quantization;
1415-1416 dequantize their weights, and 1397 calls `torch.nn.functional.linear`.
Lines 1486-1489 request requantization only for quantized activations. Thus the
specified `--fast fp8_matrix_mult` does not turn this file's linears into FP8
GEMMs: activations and dequantized GEMM weights are BF16. The 2511 precise-scale
CUDA dequantization and stochastic resident LoRA materialization stay unchanged.

`comfy/ldm/qwen_image/model.py:197-199` calls joint optimized attention;
`comfy/ldm/modules/attention.py:904` selects PyTorch attention and
`:579-600` calls `comfy.ops.scaled_dot_product_attention`.
`comfy/ops.py:70-84` orders PyTorch Flash before cuDNN, efficient, and math.
The saved server log says `Using pytorch attention`; that alone does not identify
which SDPA kernel executes. A serialized probe with installed Torch
2.14.0+cu130, BF16 BHSD tensors, 24 heads, D=128 and representative sequence
lengths 8,249 / 16,441 selected `aten::_scaled_dot_product_cudnn_attention`
at both sizes, specifically
`cudnn_generated_fort_native_sdpa_sm120_flash_fprop_f16_knob_1_64x64x128_4x1x1_cga1x1x1_kernel0_0`.
See `target/edit2511/sdpa-probe.log`. The probe uses ComfyUI's exact priority
order; it is an operator probe rather than a newly profiled ComfyUI graph.
Native retains its established BF16 flash Q/K and clamped FP16 PV arithmetic;
no INT8/FP8 attention or cuDNN dependency was introduced.

`comfy_execution/caching.py:83-129` keys node outputs by the node and its ancestor
input signatures. `execution.py:446-451` returns a cached result before executing
its node. Changing only the sampler seed does not invalidate image resize,
reference/target VAE encoding, vision, or text conditioning. The saved warm
histories in `evidence/comfy/*-1.json` explicitly list these cached nodes.
`comfy/model_management.py:939-978` reuses entries in `current_loaded_models`;
`:893-925` unloads only when memory is needed. The model need not reload on every
seed change. Warm sampler outputs and the decode are recomputed.

`comfy/ldm/qwen_image/model.py:463-501` appends reference tokens and duplicates
conditioning time as t/0 for `index_timestep_zero`. Lines 565-576 retain target
conditioning for the output projection and crop the projected reference tokens.
Native follows these conventions, with separate positive/negative forwards when
CFG > 1 and no padded CFG lane.

## Changes and scope

The baseline already selected the Layered 64x64 flash entry point with query
staging reuse, large-image BF16 cuBLASLt, fused head RMSNorm, fused modulation for
each t/0 slice, and fused bias/GELU. `forward_precise` delegates to the same CUDA
linear path; its wider arithmetic applies only on CPU. Edit's split path still
copied modulation slices, concatenated them, and expanded two full-size gates
per image block. It did not use Layered's checkpoint prefetch.

* Retain one fully bound DiT between cached requests. Before a target or
  conditioning cache miss, or an adapter change, drop it and trim the allocator
  before loading encoders. Noise, sampler state, steps and output remain fresh.
* After a new LoRA stack is materialized, reload the exact CPU weight snapshot
  once before retaining the model. This releases the binding originals and
  replaces the allocation layout left by materialization. A cold two-adapter
  baseline reached 31.5 GiB and took 113.76 s in its first step; it was stopped.
  This cold-run observation is separate from the supplied warm baseline.
* Apply t/0 modulation in one CUDA launch with per-token parameter selection.
  Apply residual gates directly from the small parameter tensor. Preserve both
  the BF16 product rounding and subsequent BF16 addition; do not replace Edit's
  residual with Layered's single-round add-product arithmetic.
* Skip the byte-exact identity Kontext resize, and memoize one bounded nonidentity
  RGB resize (input <= 64 MiB) for repeated seed requests. Read RGB blobs directly
  in Edit's worker preparer instead of allocating an opaque RGBA copy and then
  converting that copy back to RGB. Coverage and ignored-alpha semantics stay
  unchanged for 1/3/4-channel inputs.
* Add `QIR_PHASE_PROFILE=1` phase/per-step diagnostics and connect the existing
  `QIR_PROFILE=1` operation reports to the worker replay example.

Only Edit's split path uses the new kernels. The Layered modulation entry point
instantiates the original, unsplit specialization. Krea 2/2.1 dispatch, kernels,
FP8 constructors and sampling are unchanged. All kernels compile in `build.rs`;
there is no runtime NVRTC.

## Measurement and reproduction

Use the harness and fixtures in `qwen_edit_2511_pipeline.md`. The GPU is RTX 5090
32 GB on Windows/WDDM. Every measured request uses seed 43 after seed 42; no
inference-result cache exists. CUDA inference processes run serially. The
ComfyUI reference images/histories are the supplied same-day evidence copied
from worktree `283b`; no server is running during native measurements.

The native worker replay and validation use the debug Cargo profile, matching
the earlier harness. Build in this worktree:

```powershell
$env:CARGO_TARGET_DIR = "$PWD/target/edit2511"
& crates/qir/tools/layered-build.cmd build -p gn-diffusion --example edit2511_parity --features cuda --offline
```

`gn-diffusion/cuda` enables qir's cuda,fp8,flash,convrot,sage,sage2,fusednorm features.
The harness accepts `--binary`, `--native-dir`, `--phase-profile`, and `--profile`
for native replays; `compare --native-dir <after> --baseline <before>` requires
RGB MAE <= 1/255 and no PSNR reduction against the saved ComfyUI image.
`QIR_EDIT2511_RESIDENT=0` is the residency ablation. Operation profiles synchronize
every operation and are diagnostic, not production timing measurements.

## Evidence and validation

The original instrumented binary measured warm edit/fill at 13.386/13.077 s.
A separate full reload ablation uses the same arithmetic with the cold snapshot
repack already enabled and residency disabled. That ablation retains the DiT
through decode, then drops it at the next request's start; the original drops it
before decode. Its phase table therefore includes release at the start rather
than the end. All eight ablation warm images are byte-identical to the original
saved native outputs (`evidence/baseline-original-gate.json`). This is an explicit
ablation, not a claim that its cold runs are an unchanged HEAD benchmark.

Intermediate warm edit observations: resident DiT 9.248 s; split fusions 8.872 s;
identity/cached Kontext preparation 8.206 s; direct RGB preparation is included
in the final full replay below. Intermediate fill observations were 9.385,
8.963 and 8.783 s respectively. The last value still missed the 8.76 s target,
which prompted the direct RGB change. These are individual observations with
clock/loading variance, not additive isolated speedup estimates.

## Final warm timings and parity

Seconds; 1024x1024, seed 43 after seed 42. Native phase profiling is on, per-operation profiling is off. ComfyUI times/images are the supplied same-day harness results, not a fresh server run. All eight final times beat those targets. Individual observations are not latency-distribution guarantees.

| Template | Steps / CFG | Supplied native before | Reload ablation | Final native | ComfyUI | PSNR before = after |
|---|---:|---:|---:|---:|---:|---:|
| edit-lightning-8 | 8 / 1 | 13.47 | 12.550 | **8.288** | 8.398 | 43.26 dB |
| fill-lightning-8 | 8 / 1 | 13.49 | 12.672 | **8.332** | 8.759 | 49.71 dB |
| remove-lightning-8 | 8 / 1 | 11.44 | 10.932 | **8.271** | 8.513 | 49.97 dB |
| expand-lightning-8 | 8 / 1 | 13.46 | 12.778 | **8.298** | 8.504 | 33.25 dB |
| fill-ref-lightning-8 | 8 / 1 | 26.00 | 25.044 | **20.269** | 21.136 | 48.59 dB |
| image-ref | 8 / 1 | 25.14 | 24.425 | **20.415** | 21.131 | 35.47 dB |
| angles | 4 / 1 | 7.35 | 6.791 | **4.368** | 4.459 | 43.15 dB |
| edit | 40 / 4 | 90.53 | 88.942 | **78.166** | 81.880 | 36.62 dB |

**All 16 images (both seeds, all eight templates) are byte-identical in full RGBA** against both the saved original native outputs and this pass's reload ablation. RGB MAE is zero, so every ComfyUI PSNR is unchanged. The original cold image-ref / expand parity limitations documented in `qwen_edit_2511_pipeline.md` are unchanged.

## Warm phases per template

Each cell is reload-ablation before -> final after, in seconds except cache milliseconds. Target cache and conditioning cache are measured separately in the raw logs; the column sums their hit costs. On every seed-43 request reference VAE encode, vision and text execute **zero times**, including both CFG prompts. DiT release/load includes releasing the prior request's ablation model at entry. Worker/other is the measured end-to-end remainder, including preprocessing, sampler preparation and final RGBA conversion. Step-by-step values are in the phase JSON/logs.

| Template | Target + conditioning hits, ms | DiT release/load, s | All steps, s | Mean step, s | VAE load/decode/readback, s | Worker/other, s |
|---|---:|---:|---:|---:|---:|---:|
| edit-lightning-8 | 0.833 -> 0.721 | 3.331 -> 0.001 | 7.977 -> 7.695 | 0.997 -> 0.962 | 0.443 -> 0.511 | 0.799 -> 0.079 |
| fill-lightning-8 | 0.793 -> 0.711 | 3.376 -> 0.001 | 7.982 -> 7.731 | 0.998 -> 0.966 | 0.442 -> 0.517 | 0.871 -> 0.083 |
| remove-lightning-8 | 0.769 -> 0.698 | 1.636 -> 0.001 | 7.968 -> 7.701 | 0.996 -> 0.963 | 0.436 -> 0.485 | 0.890 -> 0.083 |
| expand-lightning-8 | 0.880 -> 0.734 | 3.559 -> 0.001 | 7.975 -> 7.698 | 0.997 -> 0.962 | 0.438 -> 0.496 | 0.804 -> 0.102 |
| fill-ref-lightning-8 | 1.933 -> 1.851 | 3.356 -> 0.003 | 20.275 -> 19.721 | 2.534 -> 2.465 | 0.436 -> 0.463 | 0.975 -> 0.081 |
| image-ref | 1.724 -> 2.008 | 3.393 -> 0.002 | 20.365 -> 19.866 | 2.546 -> 2.483 | 0.435 -> 0.468 | 0.231 -> 0.078 |
| angles | 0.898 -> 0.777 | 1.564 -> 0.001 | 3.993 -> 3.846 | 0.998 -> 0.962 | 0.432 -> 0.449 | 0.801 -> 0.071 |
| edit | 0.958 -> 0.700 | 6.744 -> 0.001 | 80.953 -> 77.641 | 2.024 -> 1.941 | 0.446 -> 0.451 | 0.797 -> 0.072 |

## First-request encoder/load diagnostics

Before -> after seconds for request 0 in the full replays. A dash means that stage was cached from the preceding template. These include checkpoint I/O and show substantial OS-cache / memory-pressure variance; they are not cold-start performance guarantees or isolated encoder speedup claims. Native peak VRAM is sampled once per second in `native-final.log.vram.json`.

| Template | Reference VAE load/encode | Vision load/encode | Text load/encode | DiT load | Binding/snapshot/reload |
|---|---:|---:|---:|---:|---:|
| edit-lightning-8 | 0.504 -> 0.503 | 0.481 -> 0.870 | 2.513 -> 4.855 | 6.690 -> 18.940 | 16.992 -> 17.810 |
| fill-lightning-8 | — -> — | — -> — | — -> — | 3.330 -> — | — -> — |
| remove-lightning-8 | 0.484 -> 0.633 | 0.989 -> 1.137 | 6.808 -> 3.616 | 7.437 -> 10.232 | 22.278 -> 28.506 |
| expand-lightning-8 | 0.481 -> 0.617 | 1.053 -> 0.978 | 9.498 -> 12.129 | 13.389 -> 20.392 | 13.983 -> 16.813 |
| fill-ref-lightning-8 | 0.903 -> 1.236 | 0.904 -> 1.987 | 2.621 -> 3.580 | 3.142 -> 5.079 | — -> — |
| image-ref | — -> — | — -> — | — -> — | 3.247 -> — | — -> — |
| angles | 0.540 -> 0.590 | 0.439 -> 0.678 | 2.612 -> 3.029 | 7.779 -> 8.929 | 22.824 -> 21.744 |
| edit | 0.478 -> 0.585 | 0.988 -> 1.123 | 9.087 -> 10.469 | 17.544 -> 15.494 | 6.409 -> 7.902 |

## Operation profile and quiet replay

Two-step `QIR_PROFILE=1` diagnostic runs use the warm request's second step.
These synchronize every scoped operation and alter allocation/clock behavior;
the final run also scopes the new residual work which the baseline did not.
They locate costs, not end-to-end speedups:

| Work | Before calls / seconds | After calls / seconds |
|---|---:|---:|
| Large-image BF16 Lt | 360 / 0.523 | 360 / 0.519 |
| Other FP8-stored, dequantize/BF16 GEMM | 479 / 0.061 | 479 / 0.073 |
| Joint flash attention | 60 / 0.271 | 60 / 0.273 |
| Original modulation | 361 / 0.022 | 121 / 0.004 |
| Split modulation | two slices above | 120 / 0.019 |
| Split residual | unscoped expanded gates/copies | 120 / 0.015 |
| Complete synchronized diagnostic step | 1.059655 | 1.066838 |

The production phase replay instead measures 0.997 -> 0.962 s per single-lane
step, and 2.024 -> 1.941 s per CFG step. The large warm gain is residency and
preprocessing, with a smaller step gain from eliminating split copies and gates.
A separate process with **both profiling flags off** measured warm edit-lightning-8
at **8.157466 s**, still below 8.397778 s ComfyUI. Its image is checked against
the phase replay separately. No denoising result is reused across seeds.

## Validation

* Full CUDA-feature suites pass: 11 gn-diffusion tests, 117 qir library tests,
  and 23 active integration tests, plus empty binary/doc suites. Existing
  installed-model/manual cases remain ignored as annotated. The installed
  Layered benchmark is explicitly run separately at both sizes.
* New tests require exact split normalization/residual BF16 equivalence for
  batches 1/2, splits at the start/middle/end, and nonzero tensor view offsets.
  CPU tests check identity/cached Kontext resizes and changed pixels, and
  1/3/4-channel RGB/coverage/ignored-alpha preparation.
* Both 20-step Layered splits at 480x640 and 864x1152 pass the original gate:
  **all six full raw RGBA layers are byte-identical** to the saved outputs in
  worktree `9252`'s `target/layered-size-matched-after-<size>/`.
  `target/edit2511/layered-gates.json` records zero error for every layer.
  These are debug-build output gates; no cross-build Layered latency claim is
  inferred from them. The small run's 22.60 s DiT load illustrates cold-I/O
  variance; denoising itself took 24.29 s.
* Native full-replay sampled peak is **31,700 MiB (30.96 GiB)** including the
  desktop. Sampling is once per second, not a strict allocation-bound proof.
* Edition-2021 rustfmt and `git diff --check` pass. No unwrap/expect/panic was
  introduced outside tests. Krea 2/2.1 source paths are unchanged; their active
  available tests pass. Full Krea/2.1 GPU image generation was not repeated.

Test command from the worktree root:

```powershell
$env:CARGO_TARGET_DIR = "$PWD/target/edit2511"
& crates/qir/tools/layered-build.cmd test -p qwen-image-rs -p gn-diffusion --features gn-diffusion/cuda,qwen-image-rs/cuda,qwen-image-rs/fp8,qwen-image-rs/flash,qwen-image-rs/convrot,qwen-image-rs/sage,qwen-image-rs/sage2,qwen-image-rs/fusednorm --offline -- --test-threads=1
```

## Files changed

* `crates/qir/src/edit2511_pipeline.rs`: bounded/context-aware DiT residency,
  exact snapshot repack, Kontext preparation cache, and phase diagnostics/tests.
* `crates/qir/src/model/qwen_image_dit.rs`: Edit split modulation/residual
  dispatch, checked CUDA bridges, and exact BF16 regression tests.
* `crates/qir/kernels/fusednorm/layered_rope.cu`: split parameter selection
  specialization and separately rounded residual kernel. The existing Layered
  entry point instantiates the unsplit specialization.
* `crates/gn-diffusion/src/lib.rs`: Edit-only direct RGB preparation and its
  channel/alpha tests; worker preparation timing.
* `crates/gn-diffusion/examples/edit2511_parity.rs`, its `Cargo.toml` and root
  `Cargo.lock`: enable existing tracing operation reports in the replay example.
* `crates/qir/tools/edit2511_gpu_parity.py`: alternate binary/output directories,
  phase/operation profiling, and a strict complete warm baseline/parity gate.
* This report.

## Evidence and remaining work

Ignored evidence lives entirely in this worktree's `target/edit2511/`:
`evidence/{before,resident,split,pre-rgb,after,quiet,comfy}/`, manifests,
`evidence/{comparison,baseline-original-gate,all-seed-gates}.json`,
`{before,after}-phases.json`, `baseline.log`, `before-phases.log`,
`before-repacked-phases.log`, `resident-phases.log`, `split-phases.log`,
`pre-rgb-phases.log`, `native-final.log` and its VRAM samples,
`profile-{before,final}.log`, `parity-final.log`, `tests-final.log`,
`layered-*.log`, `layered-gates.json`, `sdpa-probe.log`, and `quiet-final.log`.
`target/run-final.py` documents the serial replay/test/probe sequence. Original
ComfyUI PNGs/histories are copied unchanged from worktree `283b`.

Cold checkpoint I/O and LoRA materialization remain expensive and variable.
The warm route bypasses loading, so Layered mapping prefetch is not needed for
these hits; an Edit-specific prefault/streamed materialization path remains a
cold-start opportunity. VAE loading and per-forward positional preparation also
remain recomputed. The existing cold ComfyUI transient-BF16 LoRA discrepancy,
expand's stricter proposed 35 dB threshold, and broader scene/size/seed coverage
remain as documented in the original pipeline report. No warm template or
requested byte-equivalence gate remains unmeasured. Native advantages over
ComfyUI are small for angles/edit-lightning, and these single warm observations
are not a statistical latency guarantee under desktop or other GPU contention.

