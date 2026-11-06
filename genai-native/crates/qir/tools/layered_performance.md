# Layered RTX 5090 performance and parity (2026-10-27)

The request is the 1024px `gen-tshirt.png`, resampled with PhotoCraft's actual
Lanczos-3 routine to 640x640, three layers, 20 Euler/simple steps, CFG 2.5, seed 7,
positive prompt `separate this image into layers: the background and each foreground subject on its own layer`,
and empty negative text. Each run loads the VAE, text encoder, and entire DiT;
there is no model or conditioning cache between splits. The idle PhotoCraft worker
was left running and untouched. No Krea 2, Qwen-Image-2.1, shared FP8, or shared
flash implementation was changed.

## Measurements

These are GPU-synchronized **phase boundaries**, with no intermediate tensor
observations or extra diagnostic step synchronization. Production uses step
barriers with a scoped, bounded workspace reservation, described below. The original implementation was
rebuilt from HEAD and run with the same benchmark, input pixels, features, and
CUDA kernel cache as the optimized implementation. Model files were in the OS
page cache. The requested build environment and an isolated `target/layered-perf`
were used.

| Phase | Original | Optimized |
|---|---:|---:|
| Reference load + encode | 0.32 s | 0.32 s |
| Encoder load + both prompts | 5.18 s | 2.46 s |
| DiT load | 8.83 s | 6.36 s |
| All 20 denoising steps | 69.22 s | 47.19 s |
| Average per step | 3.461 s | 2.360 s |
| VAE load + three RGBA decodes | 0.55 s | 0.57 s |
| End to end, including device/pipeline initialization | 84.688 s | 57.493 s |

With **all timing flags off**, the production path took **57.408 s** in a fresh
process and **56.836 s** for a second split in the same process. Both loaded all
weights again, and returned identical RGBA bytes. The table is a separate run of
the final default configuration with phase reporting enabled.
ComfyUI's supplied fresh-model result is 58 s;
the supplied native worker baseline is 86 s (text ~8 s, DiT load ~20 s,
steps ~2.9 s). Those measurements are context, not substituted into the matched
local comparison above.

The GPU kernel cache matters independently of model loading. With the default
cache location in this restricted execution environment, fresh processes spent
about 10.8 s in their first reference phase, versus 0.17 s on repeat. A writable
`CUDA_CACHE_PATH` populated once brought the fresh-process reference phase to
0.32 s. This is a kernel cache, not a checkpoint or inference-result cache.
A completely empty CUDA kernel cache still incurs first-use compilation overhead.

`QIR_PROFILE=1` synchronizes each operation and considerably inflates times.
`QIR_LAYERED_TIMING=1` only synchronizes phase boundaries. The benchmark's
`QIR_LAYERED_SYNC_STEPS=1` is an optional diagnostic which also perturbs allocator
behavior; it is **not** used for the table. Quiet production results above,
not the diagnostic timings, determine whether the latency target is met.

## Parity

Compare full 1024x1024 RGBA planes after PhotoCraft's resampler, with RGB cleared
at exactly zero alpha to match document PNG export. RGB error includes all three
channels and every pixel. Alpha IoU uses alpha >127. Without matching export's
zero-alpha RGB behavior, arbitrary invisible colors dominate the raw RGB metric.

| Layer | Original RGB MAE (0-255) | Optimized RGB MAE (0-255) | Original alpha IoU | Optimized alpha IoU |
|---|---:|---:|---:|---:|
| Background | 1.06208 | 1.14159 | 1.000000 | 1.000000 |
| Foreground 1 | 2.97302 | 2.96783 | 0.997645 | 0.997489 |
| Foreground 2 | 3.21013 | 3.19066 | 0.993964 | 0.994234 |

Optimized alpha MAEs are 0.02685 / 0.36464 / 0.21758 per 255. These stay within
the requested approximate RGB 1.1 / 3.0 / 3.3 and alpha IoU 1.000 / 0.998 / 0.994
baseline. The CUDA RoPE and broadcast addcmul tests additionally require exact
agreement with the chains they replace, including nonzero tensor offsets and
shared/per-batch RoPE tables. Attention masking is checked against physically
removing padded keys, on CPU and CUDA, for either CFG lane being shorter.

## Kept changes and measured experiments

- Mmap payload reads remove per-tensor opens, seeks, zero-fill buffers, and host
  copies. Early ablation: text 6.27 -> 4.00 s; DiT 8.83 -> 6.27 s.
- The contiguous FP8 payload is uploaded once for resident DiT execution; matrices
  are views into its device storage. Noncontiguous files fall back to individual
  reads. The header/offset checks remain. Isolated later load comparison was
  approximately 6.79 -> 6.46 s; final repeated full runs measured 6.31 / 6.42 s.
- Quantization JSON/scalars stay on the CPU; CUDA no longer uploads and immediately
  reads them back. The CUDA text encoder retains blocks for both prompts, while
  CPU keeps streamed execution. Combined load/text ablation improved text
  3.96 -> 3.71 s (2.44 s in the final kernel-cache-warm run).
- Both CFG branches share one batched DiT forward. Either text can be shorter;
  padding keys are excluded by exact flash attention's existing `KeyBias`, with
  a score bias in the CPU fallback. Flash itself is unchanged. CFG batching alone
  measured 3.45 -> 3.41 s per synchronized diagnostic step.
- CUDA LayerNorm/RMSNorm reductions use fused candle operations, preserving f32
  accumulation and the bf16 norm-output rounding. The existing tested gate-add
  kernel replaces large f32 addcmul temporaries. This measured 3.41 -> 2.67 s
  per diagnostic step. Text's first residual retains its separate bf16 multiply
  and add, as ComfyUI does.
- A small Layered-only NVRTC RoPE kernel preserves the first bf16 product's
  rounding before addcmul. It measured 2.67 -> 2.36 s per diagnostic step.
  Tables are computed once per split. Scalar sampler/CFG arithmetic and
  timesteps remain f32; full-precision FP8 dequantize-to-bf16 matmuls remain.
- The final quiet-path check exposed allocator overhead: without scoped pool
  retention it took 62.876 s. Stage-boundary synchronization alone did not fix it
  (62.675 / 62.022 / 62.937 s). Retaining 2 GiB of workspace above live DiT
  storage across completed-step barriers reduced this to 57.408 / 56.836 s.
  The prior release threshold is restored on success and error; the test checks
  restoration on cancellation. Loading scratch and checkpoint storage are trimmed
  at stage transitions. `QIR_LAYERED_POOL=0` or `QIR_POOL_RETAIN=0` opts out.
- Rejected: merged QKV GEMM (4.7 ms versus separate 4.3 ms), cuBLASLt BF16
  (4.2-4.3 ms, no useful gain), CUDA graph replay (about 2.60 s per step), and
  parameter caching alone (about 2.65 s per step). Graph and parameter-cache
  experiments were removed. The ignored GEMM microbenchmark remains reproducible.

Ablations used three/four-step runs and synchronized diagnostic steps; their
absolute times must not be compared with unsynchronized production timings.
ComfyUI sources checked locally were `comfy/ldm/qwen_image/model.py`
(`_modulate`, `_apply_gate`, text residuals, output normalization) and
`comfy/ldm/flux/math.py` (`_apply_rope1`).

## Reproduction

From the repository root in PowerShell (the build wrapper mirrors
`crates/gn-segment/tools/f16-build.cmd`):

```powershell
$env:CUDA_CACHE_PATH="$PWD/target/layered-cuda-cache"
$env:CUDA_CACHE_MAXSIZE='1073741824'
New-Item -ItemType Directory -Force $env:CUDA_CACHE_PATH | Out-Null
$env:QIR_LAYERED_TIMING='1'
$env:QIR_LAYERED_REPEAT='2'
$env:QIR_LAYERED_OUTPUT="$PWD/target/layered-final"
Remove-Item Env:QIR_PROFILE,Env:QIR_LAYERED_SYNC_STEPS -ErrorAction SilentlyContinue
& crates/qir/tools/layered-build.cmd test -p qwen-image-rs --release --offline --features cuda,fp8,flash,convrot,sage,sage2,fusednorm --test layered_pipeline installed_benchmark -- --ignored --nocapture --test-threads=1
& 'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe' -I crates/qir/tools/layered_compare.py target/layered-final
```

For quiet end-to-end timing, unset `QIR_LAYERED_TIMING` as well. For a short
ablation set `QIR_LAYERED_STEPS=3`; use 20 for the parity gate. `QIR_COMFY_MODELS`,
`QIR_LAYERED_INPUT`, and `QIR_LAYERED_OUTPUT` override the installed paths.
The benchmark writes raw 640px outputs as well as the 1024px export comparisons;
PNG saving and output resampling are outside its split timer.

Validation commands use the same feature set and build wrapper:

```powershell
& crates/qir/tools/layered-build.cmd test -p qwen-image-rs --release --offline --features cuda,fp8,flash,convrot,sage,sage2,fusednorm --lib layered_ -- --nocapture --test-threads=1
& crates/qir/tools/layered-build.cmd test -p qwen-image-rs --release --offline --features cuda,fp8,flash,convrot,sage,sage2,fusednorm --test layered_pipeline --test qwen25_text --test qwen_image_dit -- --nocapture --test-threads=1
& crates/qir/tools/layered-build.cmd clippy -p qwen-image-rs --release --offline --all-targets --features cuda,fp8,flash,convrot,sage,sage2,fusednorm
```

The mmap test closes the reader before deliberately corrupting a file; mapped
checkpoints must remain immutable while in use. Tests cover both contiguous FP8
views and noncontiguous fallback. Existing slow CPU full-checkpoint tests remain
ignored.

## Validation and remaining work

- Four focused unit tests pass, including exact RoPE/addcmul rounding, CPU/CUDA
  padding masks, packed/fallback readers, and pool-policy restoration on error.
- Nine ordinary integration tests pass across `layered_pipeline`, `qwen25_text`,
  and `qwen_image_dit`; six installed/slow tests remain ignored by that command.
- Full CUDA 20-step runs pass, including repeated returned-byte determinism and
  the saved ComfyUI image comparison above.
- Release Clippy with all requested features and all targets passes. Its sole
  source warning is the pre-existing `lora.rs:188` type complexity warning;
  nvcc compiler-detection and unused Cargo-patch build warnings also remain.
- Direct edition-2021 rustfmt checks and `git diff --check` pass. Changed
  production code contains no unwrap/expect/panic calls. Cargo's unrelated
  lockfile regeneration is restored before delivery.

The measured target is met with populated CUDA kernel and OS file caches,
including every model load/upload. An empty driver kernel cache still adds
first-use compilation time; reaching 58 s in that condition is not claimed.
Only the specified 640x640, three-layer request was timed and image-gated.
Larger workloads and a rebuilt PhotoCraft worker/UI were not benchmarked here;
the production `LayeredPipeline::split` path was exercised directly. Shared
Krea/2.1 source and kernels are unchanged, and the pool policy does not leak into
subsequent pipelines.

Raw logs are under `target/layered-logs/`, particularly `before-layered.log`,
`pool-full-layered.log`, `final-default-layered.log`, `final-unit-layered.log`,
`final-regression-layered.log`, and `final-lint-layered.log`. Raw and resized
images are under `target/layered-pool-full/` and `target/layered-final-default/`.
These are generated artifacts and are not committed.

Changed files (all uncommitted):

- `crates/qir/src/layered_files.rs`
- `crates/qir/src/layered_pipeline.rs`
- `crates/qir/src/model/qwen25_text.rs`
- `crates/qir/src/model/qwen_image_dit.rs`
- `crates/qir/tests/layered_pipeline.rs`
- `crates/qir/tests/qwen25_text.rs`
- `crates/qir/tests/support/layered_resize.rs`
- `crates/qir/tools/layered-build.cmd`
- `crates/qir/tools/layered_compare.py`
- `crates/qir/tools/layered_performance.md`

The 2026-10-28 864x1152 and 480x640 follow-up, including ComfyUI execution details and matched parity gates, is in [layered_size_performance.md](layered_size_performance.md).
