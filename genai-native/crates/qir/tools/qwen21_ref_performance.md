# Qwen-Image-2.1 picture performance, 2026-10-28

The second-request regression was text-encoder eviction, not a seed cache failure. The
original worker log records a 4.54–4.73 s encoder reload on each subsequent picture request.
The native pipeline also recomputed deterministic conditioning which ComfyUI retains when
only the seed changes. Both issues are addressed below. No model quantization, sampling
steps, CFG, reference resolution, or attention arithmetic was changed.

These are **qir/backend measurements**, not a rebuilt PhotoshopEX speed table. The app build
patch points at the main checkout, not this worktree. The real app upscale, its Restore LoRA,
tile extraction, alignment, blending, and IPC remain an app-level verification item.

## Evidence and reproducibility

GPU: RTX 5090, 32 GB, driver 617.42. Release build in `target/qwen21-perf`, using
`crates/qir/tools/layered-build.cmd` and features
`cuda,fp8,flash,convrot,sage,sage2,fusednorm`. All kernels were built ahead of execution.
Native loads the installed bf16 source files and the existing ConvRot DiT cache; no
`--text-quant` or GGUF override. ComfyUI uses the PhotoshopEX graphs' installed INT8 ConvRot
DiT/text files and bf16 VAE, with `--fast fp8_matrix_mult` and `--preview-method auto`.
Only one inference process occupied the GPU at a time. The ComfyUI process started here
was stopped before the final native runs.

Local, ignored artifacts are under `runs/qwen21/`:

- `before.log`, `before/*.rgba`: original arithmetic, seed 42 then 43, each route, one resident
  process. `baseline.exe` preserves that executable.
- `cache.log`, `cache/*.rgba`: conditioning reuse alone; all eight pictures byte-identical.
- `warm.log`, `warm/*.rgba`: final implementation, two text-to-image warmups, fresh seeds
  for each picture request, then nine deliberate conditioning misses.
- `profile.log`, `profile/*.rgba`: `QIR_PROFILE=1`, including prefix/cached attention and
  MLP scopes. Profiling synchronizes around operations and is not an end-to-end speed claim.
- `comfy-events.log`, `comfy-tiles.log`, `comfy.stderr.log`: actual WebSocket node events,
  fresh seeds, cache-hit node lists, startup attention/backend/dtype selection.
- `nine-before/`, `image-before/`: additional parity oracles. The text-to-image oracle runs
  with `QIR_CPU_EMBEDDING=0 QIR_VAE_CONV_ROWS=0`, reproducing its previous arithmetic/policy.
- `tests.log`: full requested-feature qir tests. `parity-final.json`: final RGBA comparisons.

The harness takes the benchmark portrait and hat from the installed speed-table fixtures.
The hat is capped near one megapixel before qir sees it. Its Lanczos resize uses the Rust
image library, not PhotoshopEX's custom resize implementation. Fill uses a constructed
768×672 crop/mask with a 500×420 white rectangle. These inputs are identical between the
native before/after runs and the backend comparison, but are not an app pixel-export oracle.

```powershell
$env:CARGO_TARGET_DIR = "$PWD/target/qwen21-perf"
cmd /c crates\qir\tools\layered-build.cmd build -p qwen-image-rs --release --example qwen21_ref_bench --features cuda,fp8,flash,convrot,sage,sage2,fusednorm
$env:PATH = "C:/Users/5090/scoop/apps/cuda/current/bin/x64;$env:PATH"
$env:RUST_LOG = 'info'
$env:QIR_PHASE_PROFILE = '1' # phase synchronization, without per-operation profiling
& target/qwen21-perf/release/examples/qwen21_ref_bench.exe `
  --models C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models `
  --cache C:/Users/5090/AppData/Local/genai-native/qir-cache `
  --picture C:/Users/5090/ComfyUI/photocraft-tests/bench/speed/portrait-864x1152.png `
  --reference C:/Users/5090/ComfyUI/photocraft-tests/bench/stock/st-hat.jpg `
  --routes image,image-ref,fill,edit,matte,tiles --repeats 2 --tile-repeats 9 `
  --out runs/qwen21/warm
# Set QIR_PROFILE=1 instead for the detailed operation scopes.
```

`qwen21_comfy_profile.py` runs under ComfyUI's embedded Python, with a running server on
8188. It loads the actual PhotoshopEX graphs, prunes unused reference loaders, and replaces
SaveImage with PreviewImage (same sampling/decode, no persistent server-side output PNG).
`--tiles-only` submits nine changed pictures and seeds. Writable ComfyUI input/output/temp/user
folders were redirected into `runs/qwen21/` because its default temp folder was not writable.

## What ComfyUI actually computes

ComfyUI paths below are relative to
`C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`;
PhotoshopEX paths are relative to `C:/Users/5090/Documents/PhotoshopEX`.

| Item | Installed ComfyUI / PhotoshopEX evidence | Native comparison |
|---|---|---|
| Edit/fill/matte settings | `crates/genai/workflows/qwen-2.1-edit.json:20`, `qwen-2.1-fill.json:20`, `qwen-2.1-matte.json:40`: 25 steps, CFG 1, Euler/simple, denoise 1, `resolution=0` | `crates/gn-diffusion/src/lib.rs:725` builds the corresponding Request; pipeline defaults are 25/1/Comfy |
| Image-ref size policy | `crates/genai/workflows/qwen-2.1-image-ref.json:114`: **resolution 1024**, not 0; `:126` creates the independent requested-size empty latent | Native uses `ref_size`, i.e. resolution-0 sizing. The capped benchmark hat maps to 1216×864 on both sides. Arbitrary larger/smaller references can differ; not silently changed because that changes the existing native output |
| Reference resizing | `comfy_extras/nodes_qwen.py:146`: resolution 0 rounds each side to multiples of 32; positive resolution rounds square-root area-preserving sides; Lanczos when resized; only the first batch image | `pipeline.rs:ref_size` / `prepare_ref` follow the resolution-0 policy, with placed-reference handling retained |
| VAE input | `comfy_extras/nodes_qwen.py:166`: VAE gets resized RGBA; vision gets RGB with alpha over white. `comfy/sd.py:837` detects Qwen 2.1's Wan-2.2-style VAE, 64 latent channels, compression 16, bf16-first dtypes. `comfy/ldm/wan/vae2_2.py:713` uses the posterior mean | Same channels, normalization and posterior mean, no stochastic reference encode |
| Vision input | `comfy/text_encoders/qwen3vl.py:65`: patch size 16, mean/std .5, tower input **float32**. `comfy/text_encoders/qwen_vl.py:9`: min 3136, max 12,845,056 pixels; 2×2 merge, duplicated temporal pair | Same pixel limits, patch/merge sizes and f32 tower. There is no hidden 384px vision resize to remove |
| Vision/text tokens | `comfy/text_encoders/qwen_image21.py:18` adds numbered image markers; `:63` removes vision rows from final conditioning and records insertion slots when VAE latents exist. Qwen3-VL still processes those rows and DeepStack features | Same placement/drop policy; the large VL prefill is real work, but is seed-independent |
| Negative branch | `comfy_extras/nodes_qwen.py:175` encodes **both** positive and negative on an encoding-node miss. `comfy/samplers.py:610` bypasses negative **sampling** at CFG 1 | Native correctly encodes/runs only positive at CFG 1. No denoising negative branch was removed as an optimization |
| DiT prefix | `comfy/ldm/qwen_image21/model.py:273` resets the sampling-run cache; `:283` chooses GPU/CPU storage; `:350` uses prefix-only prefill and target-only subsequent passes. `:198` implements segmented block-causal prefix attention; `:216` joins cached prefix keys/values to the target | Native already had equivalent per-generation prefix caching. It continues to rebuild that cache each generation, including after LoRA/target changes |
| Schedule | `comfy/supported_models.py:2074` selects shift .69 / multiplier 1; PhotoshopEX KSampler is Euler/simple | `crates/qir/src/comfy.rs:174`: same 10,000-entry Flux schedule and existing Comfy CPU noise/timestep rounding |
| Attention backend | Startup explicitly logs **Using pytorch attention**, including VAE. `comfy/ldm/modules/attention.py:893` selects backend; `:579` dispatches SDPA. Both installed INT8 files have zero `comfy_attention` overrides (`:76` handles overrides) | Native DiT uses existing Sage2(F16), INT8 QK/FP8 PV; prefix uses existing f32 chunked scores. This was retained for native-before/after parity |
| Linear precision | Graphs name `qwen_image_2.1_int8_convrot` and `qwen3vl_8b_int8_convrot`. Their safetensors `comfy_quant` entries say `int8_tensorwise`, `convrot=true`, group 256; `comfy/ops.py:1367` / `:1480` select quantized input/linear dispatch. Runtime text compute dtype logs **float16**; DiT supports bf16 (`supported_models.py:2089`) | Native DiT uses its existing ConvRot cache/policy, text backbone bf16, vision f32, VAE bf16. Switching the native text encoder to INT8/Q8 was deliberately not used to claim a speedup |
| Warm graph cache | `comfy/execution.py:120` configures output caches. Actual `execution_cached` events include TextEncodeQwenImage21 when only KSampler seed changes | This was missing natively and is the largest straightforward fix |

At 864×1152, VAE references contain 54×72 = **3888** latent tokens and the tower produces
27×36 = **972** merged vision tokens. The benchmark hat at 1216×864 gives **4104** VAE tokens
and **1026** merged vision tokens. Each 768×672 fill input gives **2016** latent tokens and
**504** merged tokens; fill has two references. Text tokens add to the DiT prefix.

## Changes

1. `pipeline.rs:572`: one exact conditioning-cache entry, with a 64 MiB input-data limit.
   Key includes prompt, used negative prompt/CFG branch, ordered reference dimensions,
   offsets, and every RGBA byte. It excludes seed, steps and target grid. Values retain
   reference latents, conditioning tensors and image slots; large vision activations are
   released. Prefix caches are **not** reused across LoRAs or target grids. Misses release
   the previous entry before computing new conditioning. Tests cover pixel, geometry,
   offset, prompt, reference-count and negative-branch invalidation.
2. `model/text_encoder.rs:415`, `pipeline.rs:877`: keep the 151936×4096 bf16 vocabulary on
   CPU and gather only requested rows before copying them to CUDA. This releases about
   1.2 GB of device memory without quantizing weights. Only the Qwen-2.1 pipeline opts in;
   Krea 2's loader and Layered remain unchanged.
3. `model/vae.rs:46`: bound decoder im2col scratch to 128 output rows, including the
   convolution halo. Norms, residuals, receptive fields and attention still see the full
   image. Biases are added at the original point. GEMM shape changes explain the small
   output differences; this is not approximate latent/image tiling. Picture auto-decode
   uses a 4 KiB/pixel estimate plus the existing 1 GiB margin. Text-to-image keeps its
   former auto-tiling estimate. cuDNN/NHWC and CPU selection are retained.
4. `pipeline.rs:487`: retain the existing cache/score-buffer memory estimate with 512 MiB
   headroom; release the per-generation DiT prefix before decoding. In the final warm
   sequence every request reports `te_resident=true`, including nine cache misses; there
   are no text-encoder reload or eviction events.
5. `profile.rs:20`, pipeline phase scopes and DiT prefix/cached attention/MLP scopes make
   synchronized phase attribution reproducible. `QIR_PHASE_PROFILE=1` avoids per-operation
   synchronization. `QIR_CPU_EMBEDDING=0` and `QIR_VAE_CONV_ROWS=0` disable the two memory
   changes for diagnosis. Normal sampling math/backends are retained.

The attempted reduction of DiT attention query chunks from 1024 to 256 was rejected:
one changed-picture output exceeded 1/255 after it also selected approximate tiled decode.
The final code retains the original 1024-query attention implementation.

## Warm measurements

Seconds, seed 43 (seed 42 preceded it), same prompt/pixels, 25 steps, CFG 1. Native total
is `Pipeline::generate` through RGBA readback; PNG is outside that total. ComfyUI total is
HTTP submission to completion, including queueing and PreviewImage PNG. These are backend
boundaries, not interchangeable with the app's wall time.

| Route | Native before | Native after | ComfyUI this session | Native PNG after |
|---|---:|---:|---:|---:|
| image-ref | 9.762 | 4.568 | 4.895 | .010 |
| fill | 7.830 | 2.688 | 2.796 | .004 |
| edit | 9.608 | 4.509 | 5.124 | .008 |
| matte | 9.808 | 4.518 | 5.127 | .008 |
| text-to-image control | 3.444 | 3.497 | not rerun | .010 |

| Native phase | image-ref before → after | fill before → after | edit before → after | matte before → after |
|---|---:|---:|---:|---:|
| Reference pixels | .0555 → cached | .0013 → cached | .0010 → cached | .0013 → cached |
| Reference VAE, all inputs | .1554 → cached | .1397 → cached | .1462 → cached | .1498 → cached |
| Vision, all inputs | .2594 → cached | .1746 → cached | .2306 → cached | .2311 → cached |
| Text, including reload before | 4.8109 → cached | 4.8757 → cached | 4.8193 → cached | 5.0076 → cached |
| DiT prefix | .6029 → .624 | .4857 → .503 | .5350 → .554 | .5331 → .555 |
| 25 DiT steps | 3.460 → 3.498 | 1.942 → 1.959 | 3.459 → 3.511 | 3.467 → 3.520 |
| VAE decode | .403 → .443 | .198 → .224 | .403 → .441 | .403 → .440 |

After, reference/vision/text are skipped on those cache hits. On a **changed** picture,
they run: roughly .001 s pixels, .15 s VAE, .23 s vision, .127 s text, .54–.55 s prefix,
3.53–3.56 s sampling and .44 s decode. Crucially text is about .127 s, not a 4.5–4.8 s
reload plus encoding. The slight sampling/decode spread across runs reflects clock/power
and profiling conditions; no step count was reduced.

Input PNG decode in the harness was about .008 s; JPEG decode about .010 s. Native worker
transport is raw RGBA (`crates/gn-diffusion/src/lib.rs:578`, PhotoshopEX
`crates/genai/src/native.rs:314`), not a reference PNG encode/decode round trip. Worker IPC
and app postprocessing were not measured here. PNG save costs above are harness-only.

ComfyUI warm node durations measured this session:

| Route | KSampler (prefix + steps) | VAE decode | Preview PNG | total |
|---|---:|---:|---:|---:|
| image-ref | 4.642 | .188 | .059 | 4.895 |
| fill | 2.634 | .113 | .042 | 2.796 |
| edit | 4.579 | .188 | .068 | 5.124 |
| matte | 4.591 | .187 | .059 | 5.127 |

Cold encoding-node misses for fill/edit/matte were 1.365/1.263/1.277 s. Warm cached-node
lists explicitly include node 6 (node 5 for image-ref); changed seed requests execute real
KSampler passes. An already-cached identical image-ref seed completed in .022 s and was
excluded. The samplers' progress logs show all 25 steps on the reported warm runs.

For repeated misses, both harnesses cumulatively flip one RGBA byte of the same 864×1152
portrait for each request, using the edit prompt and seeds 42–50. No request can reuse its
conditioning. The nine native calls total **45.551 s**, versus **55.961 s** for ComfyUI;
native per-call time is 5.040–5.078 s, with the text encoder resident throughout. This tests
the repeated-miss cost relevant to tiled upscale, **not** the app's actual nine tile crops,
Restore LoRA, resampling, composition, or the historical 40.8 s app result.

## Parity and checks

The cache-only intermediate produced eight byte-identical outputs (same seeds, all four
picture routes). Final row-convolution arithmetic has the following mean absolute errors
in **byte units**, averaged over RGBA. Divide by 255 for normalized error; the gate is ≤1.

| Route | seed 42 | seed 43 |
|---|---:|---:|
| image-ref | .00576845 | .00537361 |
| fill | .05973695 | .05802360 |
| edit | .00566421 | .00583350 |
| matte | .00700900 | .00733050 |

All pass. A separate profiling/cold-layout image-ref run selected the tiled fallback and
reached .713586/255 mean error, also within the gate; the final warmed sequence uses full
decode. All **19 final artifacts** pass (`parity-final.json`). Text-to-image's maximum byte MAE is
.000028885; the nine changed-picture outputs have maximum .006192431. The final four-route
maximum is .059736948 byte units, or .000234263 normalized. The comparison command is:

```powershell
python crates/qir/tools/qwen21_ref_compare.py runs/qwen21/baseline-all runs/qwen21/warm
```

Requested-feature test command:

```powershell
cmd /c crates\qir\tools\layered-build.cmd test -p qwen-image-rs --release --features cuda,fp8,flash,convrot,sage,sage2,fusednorm -- --test-threads=1
```

117 passed, 0 failed, 9 existing ignored tests/benchmarks. A separate default-feature run
of `cargo test -p qwen-image-rs -p gn-diffusion --release -- --test-threads=1` also passed:
114 passed, 0 failed, 7 existing ignored tests (`cpu-tests.log`). New CPU checks cover cache-key
invalidation and convolution halos/image edges, including partial final stripes and 1×1
convolutions. `git diff --check` and edition-2021 rustfmt are applied to touched Rust files.

## Remaining work / limits

- Rebuild PhotoshopEX against this worktree and run its real speed table, especially the
  actual upscale route. The backend proxy is faster than this session's ComfyUI proxy;
  it does not establish beating the historical **40.8 s end-to-end upscale** measurement.
- The one-entry conditioning cache is intentionally bounded; alternating prompts/images
  miss it. The resident-encoder fix handles the tested one/two-reference 1 MP workload,
  but larger reference sets can still correctly evict the encoder.
- Image-ref's arbitrary-input resolution-1024 policy and Python tie-to-even size rounding
  can differ from existing native resolution-0/round-away behavior. The measured sizes
  match. Correcting arbitrary-input semantics needs separate image-parity acceptance.
- Actual ComfyUI precision differs from native's established text/attention arithmetic.
  No attempt was made to hide that difference by changing native model precision.
- Krea 2 and Layered pipelines/kernels were not changed. No commit was made.

Changed files: `src/pipeline.rs`, `src/model/text_encoder.rs`, `src/model/vae.rs`,
`src/model/dit.rs` (profiling only), `src/profile.rs`, `examples/qwen21_ref_bench.rs`,
`tools/qwen21_comfy_profile.py`, `tools/qwen21_ref_compare.py`, and this report (under
`crates/qir/`). `crates/gn-diffusion/src/lib.rs` needs no change: its resident Qwen pipeline
already survives requests (`:361`), and its pool trimming (`:250`) does not evict live cache
tensors. Source files outside this worktree were not modified.
