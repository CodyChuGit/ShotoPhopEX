# Qwen-Image-2.1 runtime LoRA performance

2026-10-28; RTX 5090 32 GB, driver 617.42, Windows/WDDM. Worktree base
`03fb72e5424639a79d34c30e7e2602e772a31ce2`. Changes are uncommitted.

**The resident Qwen path removes the measured warm adapter tax.** Six warm 992x992
Restore tiles take **29.135 s**, versus **29.171 s without LoRA** and **41.563 s in
ComfyUI with Restore**. Native denoising is 21.838 / 22.003 s with/without Restore.
The final Restore run is 36.4% faster than the initial native baseline (45.833 s) and
29.9% faster than the matched ComfyUI replay. Every before/after tile passes the
1/255 mean RGB error gate; Krea 2 remains byte-identical.

## Scope and reproduction

One resident pipeline runs two sequences of six different 992x992 tiles. Timings below
use the second sequence, seeds 48â€“53; the first uses 42â€“47. Settings: 25 steps, CFG 1,
Euler/simple, PhotoshopEX's exact `RESTORE_PROMPT`, Restore rank-32 BF16 at 0.5.
The adapter file has 192 pairs; splitting fused gate/up rows gives 224 native logical
pairs. The source `photocraft-tests/bench/speed/portrait-864x1152.png` is enlarged with
Catmull-Rom to 1728x2304 and cropped at `(0,0), (736,0), (0,864), (736,864),
(0,1312), (736,1312)`. The harness saves those six actual inputs for ComfyUI.
This measures the pipeline, excluding PhotoshopEX compositing/alignment.

Models root:
`C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models`.
Native uses the existing `qwen_image_2.1_bf16.safetensors` loader and ConvRot cache,
with the existing text cache
`C:/Users/5090/AppData/Local/genai-native/qir-cache/qwen3vl_8b_bf16.safetensors-d9566dfa31628476/text_encoder.q8_0.gguf`.
Vision is enabled. ComfyUI uses its installed template's
`qwen_image_2.1_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors`,
and `qwen_image_2.1_vae_bf16.safetensors`. These are the installed routes' existing
representations, not identical quantized checkpoints. The image gate compares native
against its own frozen executable. Production does not rewrite model/cache files.

GPU inference processes ran sequentially. ComfyUI's python process was stopped before
native inference. The instrumentation bootstrap preserves ComfyUI main.py allocator
initialization order; it uses fast-launcher flags, with previews disabled. Its writable
input/output/temp/user directories are inside the worktree. Installed sources were
read, not modified.

Native uses `QIR_PHASE_PROFILE=1`, synchronizing phases/steps. `BENCH seconds` includes
pipeline execution and RGBA readback, excluding PNG/artifact writes. Comfy request wall
includes SaveImage and 100 ms history polling. Its block CUDA-event subtotals are nested
inside KSampler. Consequently the wall times and phase scopes are close workload
comparisons, not identical instrumentation or PhotoshopEX application wall times.

Ignored evidence is under `target/lora-perf/`: frozen `baseline.exe`, raw RGBA/PNGs,
inputs, logs, histories, summaries and parity JSON. Final runs are
`final-resident-lora.log` and `final-resident-none.log`; fresh frozen-baseline controls
are `control-lora.log` and `control-none.log`. Initial baselines are
`baseline-lora.log` and `baseline-none.log`.

## What the installed ComfyUI computes

Source root `C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`, version 0.39.0.
Kitchen root `../python_embeded/Lib/site-packages/comfy_kitchen`, version 0.2.37.
Line references describe these installed versions.

1. `comfy/model_patcher.py:1938` installs `LowVramPatch` for dynamic parameters.
   `LowVramPatch.__call__` at `:202` passes the supplied weight dtype as
   `intermediate_dtype`, BF16 in this setup. The ordinary patcher path at `:902-937`
   has different default intermediate precision and must not be substituted here.
2. `comfy/ops.py:168-178` checks the resident vbar signature; `:240-256` stages patches
   on a miss; `:274-294` resolves gathered/resident parameters and their signature.
3. `comfy/ops.py:306-325` casts/dequantizes the weight into compute dtype, applies
   `lowvram_fn` only if nonresident, then calls
   `orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)`
   at `:318` and copies it into resident storage at `:325`. `:329-335` marks it resident.
   `:1445` selects quantized inference; `:1489` requests requantization for quantized
   input. Later forwards reuse patched INT8 weights. There are no two adapter activation
   GEMMs per forward and no permanently expanded BF16 patched weight in this setup.
4. `comfy/weight_adapter/lora.py:236-244,275-283` casts A/B, computes BF16 `B @ A`,
   BF16 scaled delta, then BF16 weight addition. Effective scale is
   `strength * alpha/rank`; absent alpha means just strength. Restore has no alpha
   tensors. Separate BF16 rounding at these boundaries matters.
5. Kitchen `tensor/base.py:302-309` preserves layout on requantization;
   `tensor/int8.py:128-143,220-232` uses ConvRot group size 256 and per-output-channel
   scales. `backends/cuda/__init__.py:1700-1737` selects the weight rotate/requantize
   implementation; `:1745-1772` dequantizes/unrotates into requested dtype.
6. **Dynamic seeds use the module name, without `.weight`.**
   `comfy/model_patcher.py:1969` sets `m.seed_key = n`;
   `comfy/utils.py:1501` computes CRC32. For
   `diffusion_model.transformer_blocks.0.attn.to_q` the seed is **706425777**.
   Using the parameter-key CRC instead produces different weights and was rejected.

The matching native transform reverses the native quantized column groups into Comfy's
basis, dequantizes/inverse-H4 transforms in F32 to BF16, applies the BF16 patch above,
then rotates with four radix-4 butterflies in F64. Rotated values become F32; row absmax
is taken **before BF16 rounding**. Scale is F32 `amax * (1.f / 127.f)`, clamped to
at least 1e-30; amax is bounded by BF16 finite max. Numerator, scale, division,
PCG-generated stochastic noise and addition observe Kitchen's BF16 rounding boundaries;
then floor, clamp to [-128,127], INT8 conversion, and reverse into native column order.
Noise uses the row-major index and module seed, folding 64-bit hash halves and retaining
24 bits before BF16 conversion. Native's regular-Hadamard storage layout is preserved.

`tools/lora_comfy_probe.py` exercises the real installed resolver, supplying gathered
tensors directly (not testing virtual-memory page faults). Evidence:
`comfy-probe-corrected.log`: one patch call over two resolves; resident QuantizedTensor;
bytes/scales exactly match explicit BF16 patch plus seeded requantization; second resolve
leaves bytes unchanged. Side-path versus merged activation MAE is 0.0239014, a diagnostic
showing their rounding differs, not an image error measurement.

`tools/lora_weight_fixture.py` obtains real native-cache rows and uses installed Kitchen
as the oracle. `examples/lora_merge_check.rs` checks 16 rows each of block-0 to_q
(K4096), MLP out (K12288), and gate_up (K4096). Final fixture
`materialize-fixture-final.safetensors` and `weight-integrated-final.log` establish:
**327,680/327,680 INT8 bytes and all scales match exactly**, both for direct quantization
and integrated native B@A materialization. Dequantized BF16 values also match exactly.
The negative control rounding rotated values before amax yields mismatches and is not
used in production. The fixture is a targeted numerical check, not an exhaustive
assertion for all possible tensors or backend versions.

## Implementation

`src/model/dit.rs:1238` and `src/model/krea2.rs:702` retain the bound immutable file
instances, order, exact strength bits and unmatched-module diagnostics. Equal stacks
skip rebuilding slots/materializations. Changed or empty stacks rebind/clear; failed
binding invalidates identity so retries cannot incorrectly hit the old cache. Zero
strength skips patching, avoiding unnecessary quantization.

Qwen's floating linears materialize in their weight dtype. ConvRot linears dequantize,
patch and seeded-requantize once on binding, replacing resident GPU weights. Exact
original bytes/scales are kept on CPU, so strength/file changes start from originals
and clearing restores them without another quantization. This avoids cumulative drift
and a second complete original-weight copy on GPU. Physical interleaved Q/K matrices
are separated for their individual module seeds and interleaved again. Fused gate/up
rows use Comfy's fused module seed, including the full row index. Unsupported Q8
representations retain a side path. This applies through common Qwen bindings to all
existing supported routes/LoRA module names, not just the Restore filename.

The fallback shares one A projection for Q/K/V or gate/projection inputs. A single
BF16 adapter's B GEMM uses cuBLASLt `beta=1`, eliminating the separate scaled-add pass.
C and D are separate to preserve Candle aliases and interleaved views. Descriptors,
heuristic algorithms and actual pointer alignments are cached; 4 MiB workspace is
reused per stream. Offset views, small/unaligned fallback, stacked adapter sum order and
FP16 boundaries are retained. **The final default ConvRot Qwen route does not execute
these adapter GEMMs at all.** Kernels are compiled at build time; no NVRTC was added.

Krea 2 keeps original contiguous transpose layouts and projection rounding through
`pairs_exact`. Its attempted grouped/fused arithmetic failed the image gate (MAE
0.100248) and was rejected. Krea benefits from binding reuse, while its established
per-forward arithmetic stays unchanged.

Noncausal BF16 CUDA prefix attention now uses existing compiled FlashAttention,
keeping causal/other-device/dtype chunked fallbacks and V's original dtype boundary.
Installed Comfy logs `Using pytorch attention`;
`comfy/ldm/qwen_image21/model.py:197-224` routes the block-causal segments and cached
prefix attention through `optimized_attention`; `:397` onward builds prefix caches.
Native no longer allocates the former F32 chunked score matrices for eligible segments.
Mask visibility, schedule, seeds, strengths and VAE memory margin are unchanged.

First cold Restore bind was 15.529 s including lazy CUDA startup. In a separate
base-primed reset run, materialization took **1.540 s** once; the final six warm binds
were each at most **3.1 microseconds**. These costs are outside the steady six-tile totals.
`final-reset.log` generates base seed 777, six Restore requests, clears adapters and
regenerates base seed 777: **RESET byte_identical=true**.

## Timings

All values are single-run observations, not statistical confidence intervals. Initial
native baselines: with Restore 45.833 s (conditioning 3.157, prefix 3.911, denoise
36.068, decode 2.619); without 30.387 s (conditioning 3.151, prefix 3.300,
denoise 21.260, decode 2.665). A later frozen-source control is included below because
clock/available-memory variation was material. No inference processes overlapped.

Measured development stages, six warm Restore tiles:

| Stage | Conditioning | Prefix | Denoise | Decode | Wall (s) |
|---|---:|---:|---:|---:|---:|
| Original initial | 3.157 | 3.911 | 36.068 | 2.619 | 45.833 |
| Bind once | 3.273 | 4.026 | 37.528 | 2.789 | 47.628 |
| cuBLASLt B epilogue | 3.442 | 4.058 | 33.700 | 2.886 | 44.099 |
| Shared input projection | 3.420 | 4.028 | 33.405 | 2.874 | 43.741 |
| Cached GEMM resources + Flash prefix | 3.080 | 1.567 | 29.727 | 2.582 | 36.969 |
| Final resident INT8 merge | 3.318 | 1.221 | 21.838 | 2.745 | 29.135 |

Later side-path replays ranged 42â€“49 s, so the 36.969 s observation alone did not
establish the target. Resident materialization supersedes that path for default Qwen.
Early wrong-parameter-seed merged experiments are retained as diagnostics only; the
final numbers below use the verified module seed.

| Phase, six warm tiles (s) | Before, none | After, none | Before, Restore | After, Restore | Comfy none | Comfy Restore |
|---|---:|---:|---:|---:|---:|---:|
| Conditioning | 3.137 | 3.188 | 3.470 | 3.318 | 8.162 | 8.159 |
| Prefix | 3.249 | 1.240 | 4.350 | 1.221 | 0.955 | 0.943 |
| Denoising / target blocks | 21.182 | 22.003 | 40.095 | 21.838 | 28.437 | 28.471 |
| Decode | 2.637 | 2.725 | 2.924 | 2.745 | 1.041 | 1.032 |
| Pipeline / request wall | 30.217 | 29.171 | 50.923 | 29.135 | 41.526 | 41.563 |

Comfy KSampler total: 29.926 s without Restore, 29.962 s with Restore; the target-block
row above excludes its prefix and orchestration. SaveImage: 0.443 / 0.433 s.
Native prefix averages 0.204 s/tile with Restore (formerly 0.652 s/tile initially),
versus Comfy's 0.157 s/tile block subtotal. The remaining prefix difference includes
different scope/backend overhead. Native's six warm LoRA denoising tax is effectively
zero (with-LoRA is 0.166 s faster in these two consecutive runs, within run variation).

Per-tile native cells: **conditioning / prefix / denoising / decode**, seconds:

| LoRA | Tile (seed) | Before C / P / S / D | After C / P / S / D | Before wall | After wall |
|---|---:|---|---|---:|---:|
| None | 0 (48) | 0.536 / 0.540 / 3.519 / 0.438 | 0.528 / 0.200 / 3.552 / 0.444 | 5.035 | 4.726 |
| None | 1 (49) | 0.520 / 0.541 / 3.525 / 0.439 | 0.531 / 0.218 / 3.750 / 0.445 | 5.027 | 4.946 |
| None | 2 (50) | 0.520 / 0.541 / 3.530 / 0.440 | 0.541 / 0.220 / 3.755 / 0.472 | 5.034 | 4.990 |
| None | 3 (51) | 0.520 / 0.541 / 3.532 / 0.440 | 0.530 / 0.201 / 3.802 / 0.471 | 5.036 | 5.007 |
| None | 4 (52) | 0.521 / 0.542 / 3.535 / 0.440 | 0.529 / 0.201 / 3.569 / 0.445 | 5.041 | 4.746 |
| None | 5 (53) | 0.520 / 0.542 / 3.540 / 0.440 | 0.529 / 0.201 / 3.575 / 0.449 | 5.044 | 4.756 |
| Restore 0.5 | 0 (48) | 0.583 / 0.714 / 6.582 / 0.486 | 0.550 / 0.203 / 3.634 / 0.459 | 8.376 | 4.848 |
| Restore 0.5 | 1 (49) | 0.582 / 0.722 / 6.592 / 0.467 | 0.552 / 0.204 / 3.632 / 0.458 | 8.378 | 4.847 |
| Restore 0.5 | 2 (50) | 0.583 / 0.732 / 6.791 / 0.467 | 0.555 / 0.203 / 3.635 / 0.459 | 8.586 | 4.855 |
| Restore 0.5 | 3 (51) | 0.584 / 0.738 / 6.758 / 0.510 | 0.557 / 0.204 / 3.641 / 0.457 | 8.604 | 4.861 |
| Restore 0.5 | 4 (52) | 0.554 / 0.725 / 6.747 / 0.509 | 0.552 / 0.205 / 3.646 / 0.458 | 8.549 | 4.862 |
| Restore 0.5 | 5 (53) | 0.583 / 0.720 / 6.625 / 0.487 | 0.552 / 0.203 / 3.650 / 0.455 | 8.430 | 4.862 |

## Parity and tests

The gate is mean absolute RGB byte difference / 255 <= 1/255, without alignment or
color fitting. Alpha is included in max-byte/identity checks. Final **all 12 Restore
and all 12 no-LoRA images pass** against the frozen original executable. Worst MAE:
**0.003624406** Restore and **0.001097029** without LoRA. BF16/requantized weight and
Flash attention arithmetic differ from the old activation side path, so Qwen is not
expected to be byte-identical. Bind-only arithmetic was byte-identical on all 12 images.
The epilogue, grouped and cached-resource/prefix stages also passed all 12, worst MAE
0.001331, 0.001181 and 0.001266 respectively.

| Warm tile | Restore RGB MAE | No-LoRA RGB MAE |
|---|---:|---:|
| 0 | 0.002230711 | 0.001045107 |
| 1 | 0.002786053 | 0.001091616 |
| 2 | 0.003624406 | 0.001056350 |
| 3 | 0.002281756 | 0.000815310 |
| 4 | 0.002262024 | 0.000864170 |
| 5 | 0.001288950 | 0.000850006 |

Krea 2 AnyPaint rank32 at 1.0, 1024x1024, 8 steps, seed 42, prompt
`A colorful painting of a red fox beside a river in autumn, expressive brush strokes.`:
**pixel-byte-identical, MAE 0, maximum difference 0**. Evidence:
`krea-baseline.png`, `krea-resident-final.png`, `krea-resident-parity.json`.
Exact restore is separately verified on GPU by `final-reset.log` and unit tests for
non-cumulative strength changes, repeat binding, original bytes/scales, and preserving
an unadapted half of an interleaved matrix.

Final release-feature test suite: **125 passed, 0 failed, 9 ignored** across the library
and integration binaries (`target/lora-final-tests.log`). Ignored cases are the existing
opt-in GPU/model checks. Release build of the main binary and all examples passed.
`cargo check -p qwen-image-rs --all-targets --no-default-features` also passed
(`target/lora-final-cpu-check.log`).
Touched Rust files pass `rustfmt --edition 2021 --config skip_children=true --check`;
`git diff --check` passes. Python tool sources parse successfully. No new
`unwrap()`, `expect()` or `panic!()` was added outside tests.

## Memory and decode

Final CUDA default-pool high-water counters:

| Run | Used peak MiB | Reserved peak MiB |
|---|---:|---:|
| No LoRA | 19606.652 | 21024 |
| Restore | 19758.652 | 23296 |

Used-peak difference is **152 MiB**, the adapter file tensor payload. Original LoRA
slots added 160 MiB of transposed copies (including duplicated gate/up A matrices);
resident materialization removes those slots and their per-forward activations. The
new original-weight CPU backup is approximately **6.505 GiB** for this Restore stack.
The process retains those host originals for future exact strength changes/clearing.
This trades host RAM and one-time transfer for steady GPU space and speed.

A separate NVML-sampled final-representation run peaked at **28,124 MiB (27.46 GiB)**
of total board memory, including desktop/driver, out of 32,607 MiB. See
`final-resident-monitored-vram.json`; sampling was excluded from the main timing runs.
Comfy was observed at 23,543 MiB in a snapshot, not a measured peak. Its Torch counters
omit aimdo-managed resident storage and cannot be compared directly to board usage.

Every harness decode selected **whole** before and after. Final decode totals with and
without Restore are 2.745 / 2.725 s: no adapter-induced decode tax was observed here.
The PhotoshopEX process's tiled-decode cliff was not reproduced by this pipeline
harness. Its other resident allocations and application headroom need an application
replay to establish that particular cliff has disappeared. The 1024 MiB VAE safety
margin was not reduced. This report does not claim a measured PhotoshopEX end-to-end
time or universal speed/parity qualification of every possible LoRA file.

Remaining limits: Krea retains its original per-forward path because the rounding
change failed parity; Q8 fallback can still pay adapter GEMMs. Default Qwen ConvRot
and floating bindings now have rank-independent warm inference cost once materialized.
Other adapter files/strengths use the same implementation, but have not each received
an image golden in this pass. Host RAM cost and cold materialization remain explicit
tradeoffs. Native VAE decoding remains slower than Comfy in these runs.

## Changed files and commands

Production files (relative to `crates/qir`):

- `build.rs`, `kernels/convrot/lora_weight.cu`: build-time materialization kernels.
- `src/lora_materialize.rs`: dtype/layout bridges, patching, module CRC seed.
- `src/convrot.rs`, `src/convrot_cache.rs`: mutable resident weights, exact CPU originals,
  fused/interleaved patch handling and export contract.
- `src/lora.rs`, `src/lora_gemm.rs`, `src/lib.rs`: shared binding identity,
  merge arithmetic and optimized fallback.
- `src/model/dit.rs`, `src/model/krea2.rs`, `src/pipeline.rs`: binding reuse,
  Qwen resident patches/Flash prefix, Krea reference path, bind profiling.

Harness/evidence tools: modified `examples/qwen21_ref_bench.rs`; new
`examples/lora_runtime_bench.rs`, `examples/lora_merge_check.rs`,
`tools/lora_comfy_probe.py`, `tools/lora_comfy_server.py`, `tools/lora_comfy_bench.py`,
`tools/lora_compare.py`, `tools/lora_monitor.py`, `tools/lora_weight_fixture.py`,
`tools/lora_merge_experiment.py` (offline diagnostic only), and this report.
No Cargo dependencies, model source files or cache formats were changed.

From repository root:

```powershell
$env:CARGO_TARGET_DIR = Join-Path $PWD 'target/layered-perf'
cmd /c crates\qir\tools\layered-build.cmd build -p qwen-image-rs --release --examples --bin qwen-image-rs --features cuda,fp8,flash,convrot,sage,sage2,fusednorm
$env:PATH = 'C:/Users/5090/scoop/apps/cuda/current/bin;' + $env:PATH
cmd /c crates\qir\tools\layered-build.cmd test -p qwen-image-rs --release --features cuda,fp8,flash,convrot,sage,sage2,fusednorm -- --test-threads=1
```

With `$models`, `$cache`, `$portrait`, `$text` set to the paths above:

```powershell
$env:QIR_PHASE_PROFILE = '1'
target/layered-perf/release/examples/qwen21_ref_bench.exe --models $models --cache $cache --picture $portrait --reference $portrait --text-gguf $text --routes upscale --repeats 2 --lora "$models/loras/qwen-image-2.1-restore.safetensors" --lora-strength 0.5 --out target/lora-perf/after-lora
python crates/qir/tools/lora_compare.py target/lora-perf/baseline-lora target/lora-perf/after-lora
```

Omit `--lora` for the base control. Add `--check-lora-reset` to generate base seed 777
before/after the LoRA sequence and require byte-identical reset. The Comfy scripts
use the portable installation's embedded Python and retain workflow/history JSON;
stop that inference server before native execution. `lora_merge_check` requires the
ConvRot feature and a fixture produced by `lora_weight_fixture.py` with installed Kitchen.
No changes were committed.
