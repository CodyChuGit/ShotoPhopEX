# Layered R3 steps 5 and 6

The pipeline follows `qwen-layered-split.json`: RGB reference with opaque alpha,
positive prompt and empty negative prompt, AuraFlow shift 1, Euler/simple, CFG
2.5, 20 default steps, and `layers + 1` jointly denoised slices. Slice zero is
discarded; slices 1 through L become independent RGBA images in the graph's order.
The caller supplies the graph's resolution, divisible by 16. Resizing to 640 on
the longer side and resizing outputs to document size belong to engine step 7.

`LayeredPipeline::load(models_root)` resolves the installed component names and
ComfyUI tokenizer. `load_on_device` allows explicit CPU selection. Weights load
lazily during `split`: VAE reference encode, streamed text encoder, streamed DiT,
then VAE decodes. The encoder and DiT never coexist. CUDA uses bf16 activations;
CPU uses f32. Both CFG branches receive the same normalized reference. The
sampler state and CFG/Euler arithmetic remain f32. Cancellation is checked
between model calls and layer decodes; progress counts completed Euler steps.
`split_observed` exposes complete tensors for comparison without engine changes.

## CPU reproduction

From the repository root, with the installed models and portable ComfyUI Python:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:RAYON_NUM_THREADS='8'
$python='C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
& $python -I crates/qir/tools/layered_pipeline_fixtures.py
& $python -I crates/qir/tools/layered_pipeline_fixtures.py --full
cargo test -p qwen-image-rs --release --test layered_pipeline --offline -- --nocapture
cargo test -p qwen-image-rs --release --test layered_pipeline --offline -- --ignored --nocapture --test-threads=1
cargo clippy -p qwen-image-rs --all-targets --offline
rustfmt --check --edition 2021 --config-path rustfmt.toml crates/qir/src/layered_pipeline.rs crates/qir/tests/layered_pipeline.rs
```

Fixtures use the safetensors format with a `.bin` extension, since qir's
`.gitignore` excludes `*.safetensors`. The complete sampler fixture is 42,652
bytes; the complete pipeline fixture is 394,032 bytes. No tensors are sampled.
Their JSON sidecars record the oracle arguments, PyTorch version, time and peak
working set. The oracle invokes ComfyUI's actual noise, empty-latent, patching,
schedule, flow denoising, CFG, Euler, latent cutting, text blocks, DiT and VAE
functions. It materializes one encoder or DiT block at a time. The ignored Rust
test requires the installed checkpoints and runs one 64x64 step with two layers,
checks every captured tensor for finiteness, checks progress and slice geometry,
and repeats the split to check exact latent and returned-byte determinism.
`QIR_COMFY_MODELS` overrides the installed models directory.

## Measured CPU parity

ComfyUI PyTorch 2.14.0+cu130, CPU f32, eight threads, seed 7, `A red cat.`,
64x64, two layers, one step. These are complete-tensor comparisons:

| Boundary | Maximum absolute error | Relative RMS error |
|---|---:|---:|
| Noise / packed noise, 32x48 primitive fixture | 1.07288e-6 | 1.43805e-7 |
| All 21 default sigmas | 0 | 0 |
| All 21 timestep inputs | 0 | 0 |
| Primitive denoised-CFG Euler update | 1.19209e-7 | 9e-9 |
| Normalized reference | 2.83e-7 | 1.41e-7 |
| Positive text | 0.000362396 | 3.937e-6 |
| Negative text | 0.000864029 | 6.010e-6 |
| Positive velocity | 0.000346094 | 6.0381e-5 |
| Negative velocity | 0.000589088 | 1.33885e-4 |
| Updated latent | 0.001187801 | 6.39660e-4 |
| Retained slices | 0.001187801 | 5.35193e-4 |
| Decoded layer 0 | 0.018111736 | 0.001613597 |
| Decoded layer 1 | 0.005893141 | 0.001267924 |
| Layer 0 RGB / alpha | 0.003683865 / 0.018111736 | 0.001365822 / 0.002016855 |
| Layer 1 RGB / alpha | 0.005735338 / 0.005893141 | 0.001446833 / 0.000707791 |

The decoder errors above include propagated sampler error; the VAE receives each
implementation's own sampled latent. They are not a same-latent VAE error bound.
Patch inversion and slice order are exact. The ComfyUI primitive fixture took
0.084 s, and its complete pipeline oracle took 82.900 s with a 3,137,277,952-byte
peak working set. Rust's first split and comparisons took 253.183 s.
The complete Rust smoke test, including its second split, passed in 524.048 s
with peak working set 2,253,762,560 bytes (2.099 GiB). Every repeated boundary
and both returned RGBA byte arrays matched exactly. Both processes stayed well
below 25 GB; the CPU comparison and determinism test finished within ten minutes.
An independent rerun of the oracle and comparison script took 63.786 s and
reproduced these errors. Clipped/truncated bytes differ by at most 3 for layer 0
(1,446 of 16,384 bytes) and 1 for layer 1 (1,389 of 16,384 bytes). Returned PNG
pixels are therefore not bitwise equal to the ComfyUI oracle.

Ordinary tests cover noise, packing/inversion, sigmas, timestep behavior, CFG/Euler,
slice order, truncating RGBA conversion and invalid inputs. Clippy passes with no
new-file warnings; the pre-existing `lora.rs:188` type-complexity warning remains,
along with this machine's unused Cargo patch warnings. Direct rustfmt checks of
the new Rust files pass. Cargo regenerates unrelated stale lockfile entries on
this checkout; those changes are restored before delivery.
Checking `lib.rs` with `--config skip_children=true` reports its pre-existing
Krea module ordering; only the authorized registration line was changed there.

## GPU comparison for Claude

No CUDA execution was performed during CPU bring-up. Run the two captures on
the same device, at the same dimensions and with the same PNG and prompt. The
PNG must already have graph dimensions. Example for the full template:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$python='C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$env:RAYON_NUM_THREADS='8'
$env:QIR_LAYERED_DEVICE='cuda'
$env:QIR_LAYERED_WIDTH='640'
$env:QIR_LAYERED_HEIGHT='640'
$env:QIR_LAYERED_LAYERS='3'
$env:QIR_LAYERED_STEPS='20'
$env:QIR_LAYERED_SEED='7'
$env:QIR_LAYERED_PROMPT='A red cat.'
$env:QIR_LAYERED_INPUT='C:\path\picture-640.png'
$env:QIR_LAYERED_CAPTURE="$env:TEMP\layered-rust.safetensors"
$env:QIR_LAYERED_ONCE='1'
cargo test -p qwen-image-rs --release --features cuda,flash --test layered_pipeline --offline -- --ignored --nocapture --test-threads=1
& $python -I crates/qir/tools/layered_pipeline_fixtures.py --full --device cuda --width 640 --height 640 --layers 3 --steps 20 --seed 7 --prompt 'A red cat.' --input $env:QIR_LAYERED_INPUT --out "$env:TEMP\layered-comfy.safetensors" --compare $env:QIR_LAYERED_CAPTURE
```

Use the CUDA feature set supported by the GPU build host; `flash` is optional.
The oracle streams dequantized blocks and uses ComfyUI's PyTorch attention;
the Rust build can use exact flash attention. Allow for their numerical baseline
variation. The oracle writes max absolute and relative RMS errors for both text
conditionings, reference, initial noise, each CFG velocity, every updated latent,
retained slices, and each decoded layer. Decode results additionally separate RGB,
alpha, and clipped/truncated byte differences. The comparison fails on missing
keys, shape mismatches or non-finite values; it reports numerical errors without
inventing GPU acceptance limits. Captures run sequentially and retain only one
DiT block. Full-size captures are larger than the checked-in tiny CPU fixtures.

## Source qualifications

- Layered's timestep is `sigma` in f32, including on CUDA: AuraFlow has multiplier
  one; `BaseModel._apply_model` calls `timestep(t).float()`; QwenImage does not
  override `process_timestep`; its timestep projector applies scale 1000 itself.
  `comfy_timestep` is specifically Qwen-Image-2.1's double bf16 rounding. Reusing
  it here would change the time input. Tests include both paths to expose this.
- The shipped `LoadImage` supplies RGB plus a separate alpha mask. The graph
  ignores that mask, so an RGBA argument supplies RGB and the VAE adds opaque
  alpha. Generated alpha is preserved. Direct RGBA-to-VAE conditioning would be
  a different graph.
- CFG operates on denoised predictions before Euler's `to_d`, rather than
  simplifying to guided velocity algebra. Keeping that order matches f32
  cancellation at each step. SaveImage truncates clipped bytes; the existing
  VAE conversion helper rounds, so this pipeline has its own small converter.
- The existing step-4 report's CPU-f32 FP8 dequantization caveat still applies.
  This work uses the tested components unchanged; no prerequisite file needed
  modification for the tested path.
- Reusing `comfy::torch_randn` preserves PyTorch's RNG sequence and storage order,
  with measured float differences up to 1.073e-6 from the CPU transcendental math.
  Bitwise noise equality would require matching PyTorch's vector math inside that
  shared helper. It is unchanged here; the fixtures test numerical parity.

Changes are limited to steps 5 and 6; no worker, engine, routing, or existing
pipeline changes are included. All changes remain uncommitted.

Files:

- `crates/qir/src/layered_pipeline.rs`
- `crates/qir/src/lib.rs` (one module registration line)
- `crates/qir/tests/layered_pipeline.rs`
- `crates/qir/tests/fixtures/layered_pipeline.bin`
- `crates/qir/tests/fixtures/layered_pipeline.json`
- `crates/qir/tests/fixtures/layered_pipeline_cpu.bin`
- `crates/qir/tests/fixtures/layered_pipeline_cpu.json`
- `crates/qir/tools/layered_pipeline_fixtures.py`
- `crates/qir/tools/layered_pipeline_parity.md`
