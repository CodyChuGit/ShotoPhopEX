# Qwen-Image-Edit-2511 DiT, spec step 2

CPU only, eight Rayon/OpenMP/MKL/PyTorch threads, cargo `-j 6`, no CUDA features.
The original `qwen_image_edit_2511_fp8mixed.safetensors` is read without conversion.
Oracle: installed ComfyUI `comfy.ldm.qwen_image.model.QwenImageTransformer2DModel`
and `QwenImageTransformerBlock`, PyTorch 2.14.0+cu130 on CPU, with ComfyUI's
PyTorch cross attention. Every comparison covers the complete tensor.
Relative RMS is `RMS(actual-oracle) / max(RMS(oracle), 1e-12)`.

## Interface and behavior

`LayeredDit` remains the shared backbone. `DitKind::detect` selects `Edit2511`
from `__index_timestep_zero__`, or `Layered` from the addition embedding key;
conflicting/absent markers return errors. All existing constructors detect the kind.
`LayeredInput` makes the addition embedding optional and uses the first timestep
linear's bias dtype for computation. Layered still adds entry zero by default.

Use `LayeredDit::forward_latents(target, references, text, timestep)` for editing.
It independently circular-pads and patchifies each latent, builds centered grids,
appends references, computes `[t,0]` only when edit references are present, selects
both modulation halves and gates at the target boundary, and uses only t for text
and the final norm. It returns target-only unpatchified velocities cropped to the
original target size. References have positive frame indices for 2511 and negative
indices for Layered. There is no reference-count cap. `PackedImages` exposes packed
tokens/positions and target size for inspection; `LayeredBlock::forward_split` and
`image_conditioning` permit isolated block/conditioning checks.

The existing packed-token `forward`/`forward_prepared` interfaces retain their
Layered contract: callers supply positions and slice reference outputs themselves.
They do not infer a reference boundary; use `forward_latents` for editing with refs.
No existing Layered test or pipeline file was edited.

Edit CPU timestep projection uses fixed float32 frequency bit patterns from
ComfyUI's `exp(-ln(10000)*arange(128)/128)`, avoiding scalar/vector exp drift before
the phase is multiplied by 1000. The fixture checks 0, .125, .625 and .999;
maximum projection error is 5.960464478e-8, relative RMS 1.550029685e-8.

The edit timestep MLP and split blocks use f64 CPU linear accumulation; split-block
normalization statistics also use f64,
rounding to f32 at the original tensor boundaries. This reduces absolute errors
amplified by large checkpoint gates. Dequantization still rounds the weight/scale
product in f32 before wider accumulation. Layered takes its original arithmetic
path, and CUDA takes the original kernels; no GPU execution was performed.

## Fixtures

`qwen_edit_2511_fixtures.py` invokes ComfyUI's own classes. A lazy block container
lets the actual ComfyUI model forward handle timestep concatenation, reference
packing, the 60-block loop, final normalization and cropping while retaining one
block. Parameters are read/assigned individually, never as a whole-model f32 copy.

Latent inputs: target `[1,16,1,5,6]`, references `[1,16,1,3,7]`, `[1,16,1,6,3]`,
and `[1,16,1,1,1]`. These cover odd heights/widths, a one-pixel circular edge,
unequal reference grids, and frame indices +1/+2/+3. Geometry tests also cover
negative Layered reference indices, two batches and an empty reference list.
Target is `sin(i*.017)*.7`; refs are `cos(i*.013)*.4`, `sin(i*.023)*.25`,
`sin(i*.031)*.2`; text `[1,5,3584]` is `sin(i*.019)*.3`; timestep is .625.
Full forwards use one and two references.

Independent blocks use common identical boundary inputs, as the Layered oracle
does, to separate block errors from upstream drift. A CPU torch generator seeded
with zero produces standard-normal image `[1,17,3072]` and text `[1,5,3072]`, then
normal timestep embeddings `[2,3072]` with stddev .1. The complete boundary tensors
are saved; Rust does not reproduce the RNG. Separate modulation/gate fixtures use
the *actual learned* temb(.625) and temb(0), for both attention and MLP halves.
An additional two-batch block-0 check verifies the `[t_batch, zero_batch]` layout.

The small geometry fixture is safetensors-format `qwen_edit_2511_geometry.bin`
(117,180 bytes), following existing pipeline fixture naming so it is visible to
Git without changing ignore rules. The approximately 4 MB
`qwen_edit_2511_real.safetensors` is a generated, ignored installed-model fixture;
regenerate it instead of committing it.

## Reproduction

```powershell
$env:CARGO_TARGET_DIR="$PWD/target-edit2511"
$env:CUDA_VISIBLE_DEVICES=''
$env:RAYON_NUM_THREADS='8'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$python='C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
& $python -I crates/qir/tools/qwen_edit_2511_fixtures.py --full
cargo test -p qwen-image-rs --release --offline -j 6 --test qwen_edit_2511_dit -- --include-ignored --nocapture --test-threads=1
cargo test -p qwen-image-rs --release --offline -j 6 --test qwen_image_dit -- --include-ignored --nocapture --test-threads=1
$env:QIR_LAYERED_DEVICE='cpu'
cargo test -p qwen-image-rs --release --offline -j 6 --test layered_pipeline -- --include-ignored --nocapture --test-threads=1 --skip installed_benchmark --skip layered_qkv_gemm_benchmark
```

`--geometry-only` regenerates just the small fixture; without `--full`, the tool
regenerates the geometry and selected-module fixture but omits velocities.
`--checkpoint`/`--comfy` override oracle locations; `QIR_EDIT2511_DIT` overrides the
Rust checkpoint path. Installed-model tests are ignored by default and explain
which generator/checkpoint is needed. Missing dependencies fail explicitly.

## Results

All 2511 gates pass. Limits remain 0.005 / 1e-4 for independent blocks and
0.002 / 5e-4 for full velocities. Packing, positions and RoPE are exact.

| Tensor | Maximum absolute error | Relative RMS |
|---|---:|---:|
| velocity_1 | 0.00012087822 | 6.492414e-05 |
| velocity_2 | 0.000831961632 | 0.000481246221 |
| time_projection | 5.96046448e-08 | 1.55002968e-08 |
| packed | 0 | 0 |
| image_ids | 0 | 0 |
| text_ids | 0 | 0 |
| rope | 0 | 0 |
| target without references | 0 | 0 |
| temb_0 | 5.34057617e-05 | 1.70050951e-07 |
| temb_1 | 3.05175781e-05 | 1.55542065e-07 |
| block_0_modulated_0 | 0.000152587891 | 8.35931492e-07 |
| block_0_gates_0 | 4.43458557e-05 | 4.80673878e-06 |
| block_0_modulated_1 | 0.00207519531 | 4.17020014e-07 |
| block_0_gates_1 | 0.000732421875 | 4.38041493e-07 |
| block_0_batch_image | 0.00109863281 | 4.06736955e-07 |
| block_0_batch_text | 0.000366210938 | 6.94054596e-07 |
| block_0_image | 0.00109863281 | 4.06736955e-07 |
| block_0_text | 0.000366210938 | 6.94054596e-07 |
| block_29_modulated_0 | 0.00122070312 | 4.57914934e-07 |
| block_29_gates_0 | 0.000732421875 | 4.12765343e-07 |
| block_29_modulated_1 | 0.000366210938 | 4.10884385e-07 |
| block_29_gates_1 | 0.0009765625 | 4.61238935e-07 |
| block_29_image | 0.0048828125 | 4.28351294e-07 |
| block_29_text | 0.000427246094 | 7.15756748e-07 |
| block_59_modulated_0 | 0.00146484375 | 4.41186482e-07 |
| block_59_gates_0 | 0.00128173828 | 5.00474917e-07 |
| block_59_modulated_1 | 0.000671386719 | 4.71541343e-07 |
| block_59_gates_1 | 0.000114440918 | 4.48886798e-07 |
| block_59_image | 0.00439453125 | 5.15070586e-07 |
| block_59_text | 1.01327896e-06 | 1.31934869e-07 |

The full 2511 suite passed all three tests in 222.02 s. Peak Rust working set
was 21,902,389,248 bytes (20.40 GiB), below 24 GB. The full ComfyUI oracle peaked
at 2.412 GiB. Regenerating with the final one-block proxy produced all 27 real
fixture tensors bitwise identical to the previously validated oracle, including
both complete velocities. Installed weights
remain FP8 between multiplications, and only one block is materialized.

## Verification and remaining work

- Unchanged `qwen_image_dit.rs`: all three CPU tests passed, including both
  installed-model tests, in 114.26 s. The full velocity retains the previous
  max absolute 0.000923038 and relative RMS 0.000301894. Peak working set:
  21,515,898,880 bytes.
- Unchanged `layered_pipeline.rs`: all four CPU tests passed in approximately
  465 s. Only the two explicitly GPU-only benchmarks were excluded. The installed
  one-step check passed every original oracle bound, repeated the complete split,
  and obtained exact equality of all ten captured tensors and returned RGBA images.
  Peak working set: 21,710,319,616 bytes.
- CPU build used this worktree's `target-edit2511`, no CUDA features, offline,
  `cargo -j 6`. Test executables ran sequentially with `--test-threads=1` while
  working-set counters enforced a 24,000,000,000-byte ceiling.
- Targeted release Clippy checks, direct edition-2021 rustfmt checks and
  `git diff --check` pass. Clippy's existing `lora.rs:302` type-complexity warning
  and the machine's unused Cargo patch warnings remain outside this assignment.

No step-2 gate remains. No GPU was used; later encoder, VAE, LoRA, sampler and
worker-integration steps in the spec are outside this change. The wider CPU
accumulation costs some CPU throughput; CUDA continues through the existing
kernels and is unvalidated here.

Files changed/added:

- `crates/qir/src/model/qwen_image_dit.rs`
- `crates/qir/tests/qwen_edit_2511_dit.rs`
- `crates/qir/tests/fixtures/qwen_edit_2511_geometry.bin`
- `crates/qir/tools/qwen_edit_2511_fixtures.py`
- `crates/qir/tools/qwen_edit_2511_dit_parity.md`

The large generated real fixture stays ignored. Changes are left uncommitted.

