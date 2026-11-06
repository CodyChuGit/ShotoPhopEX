# Native Qwen-Image-Edit-2511 pipeline and worker gate

2026-10-28; RTX 5090 32 GB, driver 617.42, Windows/WDDM, ComfyUI 0.39.0 / Kitchen 0.2.37.
Changes are uncommitted. 2511 remains an explicitly selected alternative: its 13
worker templates are appended after the existing templates, and each is advertised
only when its base files, tokenizer and required adapters validate.

## Reproduction and scope

Build environment: `tools/layered-build.cmd`, `CARGO_TARGET_DIR=target/edit2511` in
this worktree, CUDA compute capability 120, features
`cuda,fp8,flash,convrot,sage,sage2,fusednorm`. Custom kernels compile at build time.
No NVRTC was added. Native per-operation profiling is disabled in the timed replay. GPU inference processes ran sequentially. The ComfyUI server
used the fast launcher's `--fast fp8_matrix_mult`, with previews disabled and its
input/output/temp/user directories in this worktree; its tracked python.exe was
stopped before native execution.

`tools/edit2511_gpu_parity.py fixtures` builds the manifest under
`target/edit2511/evidence`. The picture is installed ComfyUI's
`input/bench-lighthouse.png`, Lanczos-resized to 1024x1024. Additional references
are 768x1024 and 1024x640. Fill/ref masks use the PNG red channel; expand uses a
768x768 original rectangle at (128,128), a 1024x1024 nearest-edge-replicated canvas,
and a protected-rectangle mask. Both reference templates exercise three unequal
pictures. The exact prompts, input pixels and RunRequests are saved in the manifest.
All target outputs are 1024x1024, including angles' square Kontext target.

Replay `tools/edit2511_gpu_parity.py comfy`, stop the server, then
`tools/edit2511_gpu_parity.py native`; `compare` computes RGB metrics.
The native harness is `crates/gn-diffusion/examples/edit2511_parity.rs`: it exercises
actual worker routing, not a separately assembled pipeline. Each template runs
seed 42 then 43. The table uses seed 43, the second request, so the noise changes
while conditioning remains cached. ComfyUI time includes SaveImage and 100 ms
history polling. Native time includes worker preprocessing and RGBA readback;
artifact writes follow timing. Values are individual observations, not averages.

## GPU image results

All eight comparisons exceed the requested roughly 30 dB picture-agreement goal.
Expand is **33.25 dB**, below the spec's proposed, uncalibrated 35 dB threshold;
that stricter threshold is not claimed as passed. The base edit at CFG 4 is 36.62 dB.
Mean absolute RGB error is normalized to 0..1 (one byte level is 1/255).
Per-operation profiling was disabled (`QIR_PROFILE=0`) for all timings below;
earlier synchronized profiling runs were discarded.

| Template suffix | Steps / CFG | Mean absolute RGB | PSNR dB | Native warm s | ComfyUI warm s | Native warm GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| edit-lightning-8 | 8 / 1 | 0.004186 | 43.26 | 13.47 | 8.40 | 22.61 |
| fill-lightning-8 | 8 / 1 | 0.001267 | 49.71 | 13.49 | 8.76 | 22.54 |
| remove-lightning-8 | 8 / 1 | 0.001176 | 49.97 | 11.44 | 8.51 | 22.54 |
| expand-lightning-8 | 8 / 1 | 0.007650 | 33.25 | 13.46 | 8.50 | 22.61 |
| fill-ref-lightning-8 | 8 / 1 | 0.001543 | 48.59 | 26.00 | 21.14 | 23.45 |
| image-ref | 8 / 1 | 0.010757 | 35.47 | 25.14 | 21.13 | 23.45 |
| angles | 4 / 1 | 0.003483 | 43.15 | 7.35 | 4.46 | 22.51 |
| edit | 40 / 4 | 0.009208 | 36.62 | 90.53 | 81.88 | 22.67 |

VRAM is total device usage, including desktop processes, sampled with nvidia-smi at
approximately one-second intervals. The complete native run peaked at **31,756 MiB
(31.01 GiB)**, during cold binding; all requests completed on the 32 GB card.
Warm per-request peaks are in the table. ComfyUI was observed at 30,169 MiB during
multi-reference execution; this was a point sample, not a continuous full-run peak.

The unprofiled run initially exposed unbounded allocator reservations between
sampling barriers. Edit-2511 now reuses Layered's bounded workspace guard and
synchronizes once per completed Euler step, restoring the previous pool policy on
exit. Only that existing helper's visibility changed in Layered; its behavior is
unchanged and its CUDA restoration test passes. GPU encoders and DiT are still
released at their stage boundaries. Native is slower than ComfyUI on these warm
fixtures; staged reload/preparation costs remain, and no latency superiority is claimed.


## Effective-weight and sampler gates

`tools/edit2511_lora_probe.py` runs the installed LowVramPatch/resolver with real
checkpoint weights and adapters, saving complete tensors. The CUDA test
`tests/edit2511_pipeline.rs::installed_fp8_patched_weight` verifies actual FP8
bytes (including signed zero), scales, and identical-input linear projections:

| Projection / adapters in order | Exact FP8 bytes | Scale | Projection relative RMS |
|---|---:|---|---:|
| block 0 to_q / Lightning 8 | 9,437,184 / 9,437,184 | exact | 0.00006942 |
| block 0 image MLP down / Lightning 8, Remover | 37,748,736 / 37,748,736 | exact | 0.00019094 |
| block 1 image modulation / Angles, Remover | 56,623,104 / 56,623,104 | exact | 0.00012953 |

All 103,809,024 bytes match. Header/module-map tests cover Lightning 4/8's 720
rank-64 modules with alpha 8, Remover's 720 rank-16 modules without alpha, and
Angles' 840 rank-16 modules. Binding visits all 14 addressed linears in every block
and rejects unmatched modules. Dense bindings stack in order and restore exact
originals. Equal stacks reuse materialized CPU snapshots when GPU stages reload;
changed stacks start from the immutable checkpoint, avoiding cumulative drift.

The small checked-in `tests/fixtures/edit2511_sampler.safetensors` is generated by
`tools/edit2511_sampler_fixture.py` using installed ComfyUI's CONST,
ModelSamplingDiscreteFlow, CFGNorm and KSamplerX0Inpaint. Fixed oracle velocities
isolate all eight schedule/noise/input replacement/CFGNorm/output replacement/Euler
boundaries; maximum absolute tolerance is 2e-6. These tests pass, including soft
mask values, rather than only testing binary protected regions.

## Installed ComfyUI behavior that mattered

Paths below are relative to `C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`.

- `comfy/ops.py:1450-1489` selects the full-precision linear route and passes
  `want_requant=False` for ordinary BF16 activations. `:270-335` resolves dynamic
  weights: the first call applies LowVramPatch in compute dtype, commits a
  requantized resident weight, and later calls reuse it. All three real probe
  cases observe exactly one patch across two resolves.
- `comfy/quant_ops.py:94-123` overrides Kitchen's FP8 layout: recalculated scale is
  F32 `amax(weight)/448`; the reciprocal is cast to the weight dtype before the
  scaled BF16 multiply, followed by seeded stochastic rounding. Kitchen's base
  FP8 class alone is the wrong oracle: ComfyUI registers its own subclass.
- `comfy/float.py:65-77` uses a freshly module-seeded CUDA generator and uint8
  `torch.randint(0,256,shape)` noise. PyTorch's installed
  `torch/include/ATen/native/cuda/DistributionTemplates.h:50-87,319-344` supplies
  the launch/thread/Philox ordering. The native kernel reproduces that ordering
  and Kitchen's FP16 mantissa arithmetic. Seeds use the module name without
  `.weight` (`comfy/model_patcher.py:1969`, `comfy/utils.py:1501`).
- Kitchen's installed CUDA dequantizer, called at
  `../python_embeded/Lib/site-packages/comfy_kitchen/backends/cuda/__init__.py:595`,
  multiplies the FP8 value by the F32 scale before casting to BF16. The real-weight
  probe exposed the difference from rounding the scale first. Native enables
  this only for Edit-2511 CUDA DiT/vision/text calls. Existing Krea, 2.1, Layered
  and CPU arithmetic retain their established paths.
- `comfy/model_sampling.py:86-99` converts velocity to denoised and initializes flow
  noise; `comfy_extras/nodes_cfg.py:99-108` applies CFGNorm in denoised space after
  CFG. `comfy/samplers.py:634-644` mixes protected input before the model and the
  original target after prediction; Euler's derivative uses the outer state.
- `comfy/samplers.py:1223` deliberately skips `process_latent_in` for an entirely
  zero empty target. Applying Wan21 normalization to that empty image was rejected.
- `comfy_extras/nodes_qwen.py:75-101` separately resizes VL and VAE references;
  `comfy_extras/nodes_flux.py:105` supplies Kontext's fixed size/crop choices.
  The PhotoshopEX workflow JSONs determine which images traverse Kontext:
  first picture and guided mask, raw additional references, exact expand crop,
  and angles' scaled target. The original canvas remains the other templates'
  target. Expand references have no 2.1 positional offset or token-grid widening.

## Implementation and limits

The one-entry conditioning key follows 2.1 at `03fb72e`: reference pixels, prompt,
and negative prompt only when CFG > 1. Seed, schedule, CFG magnitude, target size,
mask and LoRAs do not enter that key. A separate exact-picture target cache avoids
repeating target VAE encoding on seed changes. Encoder-input caches are bounded to
64 MiB; large weights live on CPU between stages, not alongside encoders on CUDA.
Only positive text/DiT branches execute when CFG <= 1. Up to three references are
encoded independently and all participate in each DiT forward.

Templates supply LoRAs and sampling settings, including Lightning's alpha/rank
scale, Remover after Lightning 8, and Angles before Lightning 4. Unsupported
sampler/scheduler/denoise overrides are rejected explicitly: this implements
Euler/simple with denoise 1. Additional request LoRAs follow template defaults.

Native uses the committed, requantized resident weight from its first model call.
ComfyUI's cold first dynamic call transiently returns the BF16 patch before its
resident FP8 copy is used. Cold first-call byte identity is not claimed; the
reported image gate and timings compare the repeated-request resident route.
Supplementary seed-42 first-request comparisons are saved in
`target/edit2511/evidence/cold-comparison.json`: **cold image-ref is 27.35 dB**
(mean absolute RGB 0.018249), while its warm result is 35.47 dB. Other seed-42
results range from 35.03 to 49.99 dB. The known cold materialization difference is
consistent with this gap; its full contribution has not been isolated. Cold
image-ref remains a measured limitation, not a passed 30 dB gate.
The CPU streamed path does not implement stochastic FP8 LoRA materialization;
that operation explicitly requires CUDA. Independent scientific stage gates for
all CUDA vision/text intermediates are not inferred from the image metrics.

## What remains

No requested warm template comparison remains unmeasured. Remaining parity work is
cold image-ref / transient BF16 materialization, expand's stricter proposed 35 dB
target, and isolated full CUDA vision/text stage fixtures. Broader scene/seed and
non-square GPU coverage is not established by this one-scene benchmark. CPU
stochastic LoRA materialization is intentionally unsupported. Warm latency still
exceeds the installed ComfyUI route on these fixtures.

## Validation and changed files

CUDA-feature qir and gn-diffusion suites pass; heavy pre-existing model-fixture and
exclusive benchmarks remain ignored as annotated. The explicit installed-weight
and sampler integration run passes all four tests. The final worker rerun passes
all 10 tests; the shared CUDA pool-policy restoration test also passes. The
installed header/ComfyUI adapter-map test was explicitly rerun: all four adapters
match exactly (720/720/720/840 modules, rank/alpha/scale included). Existing 2.1/Krea/Layered
source paths and defaults retain their behavior; their available tests pass.

Changed/added:

- `crates/qir/src/edit2511_pipeline.rs`: staged pipeline, caches, sampler.
- `crates/qir/src/model/qwen_image_dit.rs`: binding/materialization and CPU overlay.
- `crates/qir/src/fp8.rs`, `kernels/fp8/fp8_cast.cu`: opt-in F32-scale dequantization
  and deterministic stochastic FP8 requantization.
- `crates/qir/src/layered_files.rs`, `model/qwen25_text.rs`,
  `model/qwen25_vision.rs`: Edit-only CUDA encoder scale/device entry points.
- `crates/qir/src/lib.rs`: pipeline export.
- `crates/qir/src/layered_pipeline.rs`: crate visibility for the unchanged workspace guard.
- `crates/gn-diffusion/src/lib.rs`: optional family, all 13 routes, defaults,
  availability, inputs, masks, references and family switching.
- `crates/gn-diffusion/examples/edit2511_parity.rs`: actual-worker replay.
- `crates/qir/tests/edit2511_pipeline.rs`, its sampler fixture and `.gitignore`
  exception; `tools/edit2511_{gpu_parity,lora_probe,sampler_fixture}.py`.
- This report.

Raw ignored evidence is under `target/edit2511/evidence/`: manifest/pixels,
ComfyUI histories and PNGs, native RGBA/DoneInfo/timings, resolver JSON and weights,
final native log, VRAM samples, cold/warm comparison JSON and `contact-sheet.png`
(native / ComfyUI / amplified absolute error). Root test logs are
`tests-final-2511.log`, `sampler-weight-final-2511.log`, `worker-final-2511.log`,
`pool-final-2511.log` and `inventory-final-2511.log`.
