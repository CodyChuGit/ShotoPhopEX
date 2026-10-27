# gn-segment

The Candle SAM 3.1 port for `docs/comfy/goals/rust-rebuild.md` R1 in
PhotoshopEX. This crate provides checkpoint validation, numerical foundations,
all model parts and the composed detector/tracker flows. It does not change
gn-perception's ONNX runtime.

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

## Composed flows (step 9, first half)

`sam::Sam31::load(checkpoint, &device)` reads the checkpoint once through
`weights::Checkpoint`. Every part also exposes `from_checkpoint`; its existing
path-based `load` remains a wrapper. Shared loaders retain compact F16 trunk
weights and F32 compute. Geometry and detector share fused attention, but their
encoder layers remain separate: detector self-attention adds image positions,
whereas geometry uses a different cross-attention ordering.

Call `set_tokenizer(path)` with the existing local
`C:/Users/5090/Documents/_eval/native-models/sam3.1/tokenizer.json` before text
selection. No tokenizer files are downloaded. This uses the ONNX route's JSON
normalization and BPE, disables special-token postprocessing, keeps 30 pieces,
adds BOS/EOS, and pads the 32-slot window with zeros.

`encode(rgb_or_rgba, width, height)` returns a `Frame` retaining the original
pixels, both feature pyramids, image positions and geometry CLS. Keep that frame
while the image is unchanged; all prompt methods borrow it. Alpha is ignored.
`select_text` uses the detector. `detect_text` additionally accepts positive box
tokens. `select_box` uses those detector box tokens with an empty text prompt;
this is an additional detector API, not the worker's box-only route.
`segment_box` reproduces the worker's box-only tracker route. `segment_points`
reproduces its positive/negative point route and iterative feedback. `interactive`
is a single tracker pass with optional combined points/box and 1008-square logit
feedback, returning a picture-sized binary mask and four raw predicted IoUs.
The worker runs point and box requests separately when both are supplied.

Defaults remain threshold 0.5, one detection, two refinement passes. Scores use
strict comparison, stable descending order, and no NMS or presence multiplier.
Detector logits resize directly to the picture. Refinement pads each side of the
predicted box by 10%, truncates in f64, encodes the crop using only the interactive
pyramid, iterates mask feedback and ORs foreground into the coarse binary mask.
Tracker selection uses the best of tokens 1..3 for exactly one point (first wins
ties), otherwise token 0. Nonpositive object score yields -1024 logits. Tracker
refinement always executes at least one pass. Like `sam31.rs`, these methods can
return empty instances; the worker should continue dropping `pixels == 0`.

The existing parts deliberately compute in F32; production ONNX graphs use
mixed F16/F32. This numerical difference is checked by the end-to-end harness,
without changing the earlier ComfyUI F32 parity boundaries.

### CPU comparison

Run the two backends separately so their weights and runtime arenas do not
coexist. The example uses the installed checkpoint, stock JPEGs and ORT 1.31 CPU
provider at the local paths in its source, with no GPU provider registered:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
cargo run -p gn-segment --release --example sam31_parity -- onnx
cargo run -p gn-segment --release --example sam31_parity -- candle
```

The example uses dog/bench/coffee cup prompts and a fixed positive point inside each detected
object, with the worker's default refinement. It saves binary mask
planes and metadata under `target/sam31-parity/{onnx,candle}`, reports counts,
ordered per-instance IoU, score differences and stage wall times, and fails if
counts differ or any ordered pair has IoU below 0.99. ONNX is a dev-dependency
only; production `gn-segment` does not depend on ORT. No worker code is changed.

### Measured CPU results (2026-10-27)

Every route returned one nonempty instance in the same order, with identical
foreground bounds. Score differences below are Candle minus ONNX. All six
ordered mask comparisons pass the 0.99 gate.

| Picture | Text IoU | Text score difference | Point IoU | Point score difference |
| --- | ---: | ---: | ---: | ---: |
| st-dog | 0.999563232 | +0.000014365 | 0.999978445 | +0.000000000 |
| st-bench | 0.998404349 | -0.000009894 | 0.999921213 | +0.000000000 |
| st-cup | 0.995583538 | +0.000028133 | 0.999990066 | +0.000000000 |

Wall seconds, including default refinement; encode / select / interactive:

| Picture | Candle | ONNX CPU |
| --- | ---: | ---: |
| st-dog | 59.557 / 70.836 / 0.328 | 6.578 / 11.687 / 0.132 |
| st-bench | 55.321 / 74.684 / 0.313 | 6.935 / 10.366 / 0.124 |
| st-cup | 52.209 / 71.124 / 0.322 | 7.175 / 11.179 / 0.114 |

The fixed positive points are dog (930,2140), bench (1100,690) and cup (590,870).
The mixed-precision production graphs ran using ORT 1.31's CPU provider. Peak
working sets observed through Windows Process counters were 4.26 GB for Candle
and 5.63 GB for ONNX; backends were run separately. Measurements share the host
with other jobs and include wall-time scheduling effects.

Validation passed: 28 regular tests and both ignored full-model tests,
`cargo clippy -p gn-segment --all-targets`, and
`cargo fmt -p gn-segment --check`. Cargo emits pre-existing unused-patch warnings;
there are no gn-segment compiler or Clippy diagnostics. Existing tests were not
edited. Pre/postprocessing agrees with the spec and current route. The retained
F32 model math differs from the mixed-precision export as documented above.
