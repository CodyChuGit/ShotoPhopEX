# Layered DiT step 4 parity

CPU-only, 2026-10-27. `CUDA_VISIBLE_DEVICES=""`, eight PyTorch/Rayon threads.
Oracle: installed ComfyUI `comfy/ldm/qwen_image/model.py`, PyTorch 2.14.0+cu130,
`qwen_image_layered_fp8mixed.safetensors` (20,533,591,821 bytes).
No encoder, VAE, sampler, pipeline, or other agents' files were changed.

## Implementation and boundaries

`LayeredDit` implements all 60 width-3072 dual-stream blocks, 24 heads of 128,
3584-wide normalized text input, biased mixed-file projections, 256-wide sinusoidal
timestep projection with scale 1000, learned timestep MLP and two-entry additional
condition, separate image/text modulation and tanh GELU FFNs, joint bidirectional
attention, per-head Q/K RMSNorm, three-axis adjacent-pair RoPE, adaptive output norm,
and 64-value patch velocity. Additional condition defaults to entry zero.

`forward` accepts packed tokens and explicit image/text positions. Image tokens may
include trailing references; its output includes their velocities, so the caller
retains the generated prefix before unpatchifying. Generated frames use 0..T;
single-frame references use -1, -2, etc. Spatial coordinates are centered; text
starts at the integer half of the larger target patch-grid dimension. Helpers
require even latent dimensions, as supplied by the Layered template. The template's
negative-index route uses ordinary timestep modulation for reference tokens.
This implementation does not add alternate reference methods, masks for padded text,
ControlNet, patches, or sampling behavior.

`load` supports resident weights. `open_streamed` keeps only root modules and one
block resident; its safe file backend reads individual tensor payloads without
mapping/loading the complete checkpoint. Stored fp8 tensors stay fp8 until each
linear's matmul. Full-precision weights and biases retain their stored values,
cast to f32 on CPU or bf16 on CUDA. CUDA uses public exact flash attention when
compiled with `flash`; fallback attention chunks queries and accumulates in f32.
No CUDA run was made because the GPU was reserved for other measurements.

## Fixtures and numerical results

Regenerate with ComfyUI's embedded Python:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe' -I crates/qir/tools/qwen_image_dit_fixtures.py --full
```

The script and Rust test document every formula. The picture has two layers plus
composite, a 32x32 latent grid, one 32x32 reference, and 12 text tokens: 1024 image
tokens in total. Blocks 0, 29, and 59 each receive the same independent formula
inputs, rather than the preceding block's outputs. Separate complete f32 and bf16
forwards stream all 60 blocks. The fixture is **1,313,844 bytes**, with up to 4096
uniformly spaced samples and full-tensor population mean/std/min/max per tensor.

Errors below are Rust CPU f32 against matching ComfyUI CPU f32. Relative RMS is
sample error RMS divided by reference RMS. Maximum pointwise relative error uses
`max(abs(reference), 1e-6)` and is sensitive to near-zero values.

| Tensor | Max absolute | Max relative | Relative RMS |
|---|---:|---:|---:|
| patches | 5.96e-8 | 1.67e-7 | 1.9e-8 |
| reference patches | 2.98e-8 | 1.44e-7 | 1.9e-8 |
| image positions | 0 | 0 | 0 |
| text positions | 0 | 0 | 0 |
| RoPE matrices | 0 | 0 | 0 |
| timestep projection | 2.8849e-5 | 8.8140e-5 | 2.697e-6 |
| timestep embedding, entry 0 | 1.2207e-4 | 0.0047759 | 3.14e-7 |
| timestep embedding, entry 1 | 1.2207e-4 | 0.0022070 | 3.14e-7 |
| image input projection | 2.3842e-7 | 0.0007653 | 3.7e-8 |
| text norm + input projection | 4.3869e-5 | 0.0007609 | 7.41e-7 |
| block 0 image | 0.002061844 | 0.0031793 | 4.891e-6 |
| block 0 text | 0.000366211 | 0.0199429 | 1.283e-6 |
| block 29 image | 0.003417969 | 0.0412946 | 7.12e-7 |
| block 29 text | 0.000312805 | 0.0021540 | 1.084e-6 |
| block 59 image | 0.000297546 | 0.0026881 | 1.823e-6 |
| block 59 text | 1.3113e-6 | 0.0018229 | 8.85e-7 |
| norm_out | 1.9073e-6 | 0.0001681 | 4.06e-7 |
| proj_out | 2.3842e-7 | 0.0024568 | 3.59e-7 |
| full generated velocity, f32 | 0.000923038 | 7.3723404 | 0.000301894 |

Block 29's image output reaches 7391.98, explaining its larger absolute error.
Tests also compare full-tensor mean/std/min/max, assert patch/unpatch inversion,
check odd patch-grid text positioning, and test both additional-condition entries.
Measured backend differences set the limits: independent blocks use max absolute
0.005 and relative RMS 1e-4; the 60-block forward uses 0.002 and 5e-4 respectively.
Timestep projection allows 4e-5 for vector-exp/libm differences before the 1000x
phase scale. No bitwise-parity or CUDA-parity claim is made.

The ComfyUI oracle took 167.913 s total: selected blocks including weight reads
1.612 / 1.454 / 1.653 s, complete streamed f32 111.598 s, and complete streamed bf16
50.694 s. The bf16 final velocity is captured independently; Rust CPU computes in
f32, so the parity assertion uses the matching f32 oracle. PyTorch peak working
set was **2.522 GiB**, measured with Windows process counters. Rust peak working
set stayed approximately **1.11 GiB**. Both are below the 25 GB budget.

The final Rust run took 114.768 s for the complete streamed forward and comparison,
0.054 s for input modules, 1.784 / 1.811 / 1.827 s for the selected blocks (including
loading and comparisons), and 0.071 s for output modules. The ordinary test passed
in 0.01 s; both ignored tests passed in 120.41 s total. Final Rust peak working set
was 1,191,604,224 bytes (1.110 GiB).

## Verification

The Cargo package is `qwen-image-rs`, not `qir`; the literal requested `-p qir`
commands fail package selection. Equivalent commands used:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:RAYON_NUM_THREADS='8'
cargo test -p qwen-image-rs --release --test qwen_image_dit --offline -- --nocapture
cargo test -p qwen-image-rs --release --test qwen_image_dit --offline -- --ignored --nocapture --test-threads=1
cargo clippy -p qwen-image-rs --all-targets --offline
rustfmt --check --edition 2021 --config-path rustfmt.toml crates/qir/src/model/qwen_image_dit.rs crates/qir/tests/qwen_image_dit.rs
```

Checkpoint-dependent tests are ignored by default and return successfully with a
skip message when `QIR_LAYERED_DIT` (or the default installed checkpoint) is absent.
Pure geometry tests run without the checkpoint. Clippy reports no warnings in
the new files; the existing `lora.rs:188` type-complexity warning remains.
Package-wide `cargo fmt -p qwen-image-rs --check` reports pre-existing formatting
differences, starting in `build.rs`; direct checks of both new Rust files pass.

## Spec findings

The architecture, 60-block inventory, keys, additional condition, and reference
positions agree with the spec. One qualification matters: the shared `fp8.rs`
dequantization rounds the scale and product to bf16 even for CPU f32 activations.
ComfyUI/comfy-kitchen's eager rule is
`q.to(output_dtype) * scale.to(output_dtype)`. Therefore the spec's blanket claim
that the current reader is appropriate requires a CPU-f32 caveat. The private
biased linear implements the actual compute-dtype rule without changing `fp8.rs`.
Text-start "half the larger grid dimension" is integer division, including odd
patch-grid sizes. No checkpoint conversion is needed.

## Files

- `crates/qir/src/model/qwen_image_dit.rs`
- `crates/qir/src/model/mod.rs` (one registration line)
- `crates/qir/tests/qwen_image_dit.rs`
- `crates/qir/tests/fixtures/qwen_image_dit.json`
- `crates/qir/tools/qwen_image_dit_fixtures.py`
- `crates/qir/tools/qwen_image_dit_parity.md`

Changes are left uncommitted in the worktree.
