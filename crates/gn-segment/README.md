# gn-segment

The candle SAM 3.1 port, step 1 of `docs/comfy/goals/rust-rebuild.md` R1 in
PhotoshopEX. This crate provides the checkpoint and numerical foundations; it
does not yet run the model or change gn-perception's ONNX runtime.

Step 1 covers:

- An expanded table of raw SAM 3.1 checkpoint names and expected shapes for the
  detector, both FPNs, language encoder/resizer, geometry, scoring/segmentation,
  interactive prompt/mask encoders and no-memory embedding. Complete selected
  modules are validated, including dormant geometry-point and segmentation heads.
  Video propagation and memory modules are excluded.
- `weights::validate(path)`: reads only the length-prefixed safetensors JSON
  header, returning missing keys and expected/actual shape mismatches. Extra keys
  are allowed. This is shape validation, not payload or dtype integrity checking.
- Dimension-zero Q/K/V splitting for fused attention weights and biases.
- CPU f32 reference ROIAlign (7 by 7, adaptive, unaligned), antialiased bilinear
  NCHW resize and affine channel LayerNorm. These copy through host vectors and
  are parity references, not optimized inference kernels.

CPU is the default. The `cuda` feature enables candle-core/cuda and candle-nn/cuda;
the reference operations still require CPU f32 tensors.

## Validation

```powershell
cargo test -p gn-segment
cargo clippy -p gn-segment --all-targets
```

Tests require maximum absolute error below `1e-5` against committed PyTorch CPU
fixtures. The real-checkpoint test runs when the checkpoint exists at
`C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors`
and returns early otherwise. No checkpoint is distributed with this crate.
Synthetic headers test missing/mismatched shapes without any tensor payload.

## Regenerate fixtures

From the repository root, using ComfyUI's embedded Python with torch, torchvision
and safetensors installed:

```powershell
& 'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe' -I tools/gn_segment_fixtures.py
```

The script writes deterministic float32 inputs and expected outputs under
`tests/fixtures/`, with torch/torchvision versions in safetensors metadata. It
covers inside, edge, partly outside and tiny boxes; resize down by 3.5 and 4,
up by 2, non-square and mixed directions, singleton and identity sizes; and
channel LayerNorm. Fixtures total less than 2 MB. Rust tests load them through
candle's safe safetensors loader; Python is only needed for regeneration.
