# gn-segment

The candle SAM 3.1 port, steps 1 and 4 of `docs/comfy/goals/rust-rebuild.md` R1 in
PhotoshopEX. This crate provides checkpoint validation, numerical foundations and
the text encoder. It does not change gn-perception's ONNX runtime.

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
the reference operations still require CPU f32 tensors. `text::TextEncoder` uses
the caller's device and keeps both CLIP and the resizer in f32. `load(path, device)`
reads raw checkpoint keys with shape checks, without unsafe memory mapping;
`encode(&[u32; 32], &[bool; 32])` returns f32 `[1,32,256]` features and the unchanged
boolean validity mask. Padded queries retain their features; padding masks keys.

## Validation

```powershell
cargo test -p gn-segment
cargo clippy -p gn-segment --all-targets
```

Primitive tests require maximum absolute error below `1e-5` against committed
PyTorch CPU fixtures. The text parity test requires `max_abs < 1e-3` and prints
maximum absolute, elementwise relative (denominator floored at `1e-6`), and
relative infinity-norm errors with `-- --nocapture`. Real-checkpoint tests run at
`C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors`
and return early otherwise. No checkpoint is distributed with this crate.
Synthetic headers test missing/mismatched shapes without any tensor payload.
The text test also accepts `GN_SAM31_CHECKPOINT` to override its checkpoint path.

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

For text fixtures, run:

```powershell
& 'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe' -I crates/gn-segment/tools/gn_segment_text_fixtures.py
```

The generator uses ComfyUI's SAM CLIP configuration, key conversion and CLIP
forward implementation, followed by the checkpoint resizer in f32. It compares
ComfyUI's first-window IDs to the exact gn-perception tokenization procedure
using its tokenizer JSON. Override `--comfy`, `--checkpoint` or `--tokenizer` as
needed. Four prompts cover padding and a full context. `text.safetensors` stores
IDs, boolean masks as U8 0/1 (Candle cannot load safetensors BOOL), and resized
features, totaling about 130 KiB. Metadata includes prompts, tokenizer SHA-256,
PyTorch version, checkpoint name and tokenization comparison results.

Two reference boundaries matter: ComfyUI makes extra windows for overlong text,
whereas gn-perception truncates to the first 30 pieces. Also, the existing ONNX
export casts CLIP features to f16 before an f16 resizer; these step 4 fixtures
deliberately keep the entire text path in f32, including the resizer. They test
ComfyUI parity at that precision, not the mixed-precision `text.onnx` boundary.
