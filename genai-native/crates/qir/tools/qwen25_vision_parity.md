# Qwen2.5-VL vision parity (CPU, 2026-10-28)

`Qwen25Vision` in `src/model/qwen25_vision.rs` streams the 32 visual blocks
from the unmodified ComfyUI text encoder. It uses `TensorFile::builder` and
`LayeredLinear`, including their FP8 -> BF16, BF16 scale multiplication -> F32
weight conversion. Norms, activations, attention and matrix products compute
in F32. No CUDA features or GPU allocations are used.

The public `encode(&image::RgbImage)` returns `(tokens x 3584, [1,h,w])`,
where h and w count unmerged 14-pixel patches. For the EditPlus pipeline use
`vision.encode_pixels(&resize_editplus(&rgb)?)`: the helper returns HWC F32
RGB in [0,1], preserving the fractional area-resize pixels. Converting that
helper result back to 8-bit RGB introduces unwanted quantization. `encode`
itself performs only the tower's bilinear resize, normalization and packing.

The reference generator imports the installed `comfy.text_encoders.qwen_vl`
classes, `comfy.ops.manual_cast`, ComfyUI CPU attention, and
`comfy.utils.common_upscale(..., "area", "disabled")`. It loads one block at
a time and records complete stage tensors, not samples. Large fixtures live
in `target-qwen25-vision/fixtures` and are regenerated locally. The 1,940,596
byte small fixture supplied for version control covers patch packing, bilinear normalization,
area upsampling, nontrivial window permutation, partial windows and dimensions
exactly divisible by the window (including ComfyUI's empty padding windows).

Inputs (dimensions listed as height x width):

| Case | Source | RGB size | EditPlus size | Grid | Merged tokens |
|---|---|---:|---:|---|---:|
| Portrait | bench/stock/auto7-vs-q2-half.png | 4054 x 1216 | 701 x 210 | [1,50,16] | 200 |
| Landscape | bench/stock/dog-m50.png | 342 x 2000 | 159 x 929 | [1,12,66] | 198 |
| Square | bench/swap/gen/gen-tshirt.png | 1024 x 1024 | 384 x 384 | [1,28,28] | 196 |

All source paths are under `C:/Users/5090/ComfyUI/photocraft-tests/`.
The suggested gen-tshirt picture is square, so a separate portrait is included.

Full-tensor errors, including the Rust EditPlus resize before tower execution:
relative RMS = sqrt(mean((actual-reference)^2)/mean(reference^2)). Blocks are
in window order; merged output is restored to spatial raster order.

| Case | Stage | Max absolute | Relative RMS |
|---|---|---:|---:|
| Portrait | Area resize | 0.000000060 | 0.000000042 |
| Portrait | Normalized patches | 0.000000715 | 0.000000088 |
| Portrait | Patch embedding | 0.000011921 | 0.000000783 |
| Portrait | Block 0 | 0.000015259 | 0.000000717 |
| Portrait | Block 7 | 0.000026703 | 0.000001302 |
| Portrait | Block 31 | 0.927734375 | 0.000011143 |
| Portrait | Merger, restored | 0.000133514 | 0.000006038 |
| Landscape | Area resize | 0.000000060 | 0.000000035 |
| Landscape | Normalized patches | 0.000000715 | 0.000000055 |
| Landscape | Patch embedding | 0.000008583 | 0.000000883 |
| Landscape | Block 0 | 0.000011444 | 0.000000683 |
| Landscape | Block 7 | 0.000032783 | 0.000001468 |
| Landscape | Block 31 | 13.513183594 | 0.000085977 |
| Landscape | Merger, restored | 0.000865042 | 0.000026875 |
| Square | Area resize | 0.000000060 | 0.000000024 |
| Square | Normalized patches | 0.000000715 | 0.000000056 |
| Square | Patch embedding | 0.000010014 | 0.000000987 |
| Square | Block 0 | 0.000012398 | 0.000000838 |
| Square | Block 7 | 0.000068665 | 0.000002246 |
| Square | Block 31 | 0.664062500 | 0.000015737 |
| Square | Merger, restored | 0.001078606 | 0.000015372 |

Grid shapes, grid values, window indices and cumulative window lengths match
exactly. Normalized patches computed from the exact oracle pixels are bitwise
equal for all three images. Small-fixture patches and area resize are also
bitwise equal. Every stage
passes relative RMS <= 1e-4, including the merger (stricter than the requested
5e-4 whole-tower gate); normalized patches pass max absolute <= 1e-6.
Large block-31 absolute errors reflect large reference activations; the gate
uses relative RMS. No error thresholds were relaxed.

Reference generation peak working set: 1,838,268,416 bytes (1.71 GiB).
Observed Rust parity process peak working set: 1,111,060,480 bytes (1.04 GiB).
The three-image release parity run took 23.08 seconds, excluding compilation.
Both reference and Rust execution use eight threads; cargo uses six jobs.

Regenerate and verify from the repository root (PowerShell):

```powershell
$env:RAYON_NUM_THREADS='8'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$env:CARGO_TARGET_DIR="$pwd/target-qwen25-vision"
$py='C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
& $py -I crates/qir/tools/qir_qwen25_vision_fixtures.py --small
& $py -I crates/qir/tools/qir_qwen25_vision_fixtures.py
cargo test -p qwen-image-rs --release --no-default-features --test qwen25_vision -j 6 -- --include-ignored --nocapture
```

The installed-model test is ignored by default and requires the regenerated
large fixtures. `QIR_COMFY_MODELS` can override the model root for the Rust
test. `qwen25_text.rs` was not changed. Text embedding splicing, M-RoPE and
pipeline integration remain assigned to the other jobs.

The fixture is below 2 MB, but staging it with `git add -f` was denied because
this worktree's Git index is outside the permitted writable directory. It
remains present at `tests/fixtures/qwen25_vision_small.safetensors` and needs
`git add -f crates/qir/tests/fixtures/qwen25_vision_small.safetensors` when
the changes are collected. No commit was made.
