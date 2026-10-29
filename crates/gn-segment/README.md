# gn-segment

The Candle SAM 3.1 port for `docs/comfy/goals/rust-rebuild.md` R1 in
PhotoshopEX. This crate provides checkpoint validation, numerical foundations,
all model parts and the composed detector/tracker flows. It is gn-perception's only SAM 3.1
backend (the ONNX Runtime route was removed on 2026-10-29). ONNX references in the measurement
records below are the retired route's recorded outputs.

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

CPU is the default. The `cuda` feature enables candle-core/cuda and candle-nn/cuda.
CPU reference operations retain their f32 boundaries; ROIAlign also supports device pooling. `text::TextEncoder` uses
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
selection. No tokenizer files are downloaded. This uses the tokenizer JSON's
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

Run candle against a recorded reference folder (`GN_SAM31_REFERENCE`, default `onnx`: the retired
ONNX Runtime route's saved masks). The example uses the installed checkpoint and stock JPEGs at
the local paths in its source:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
cargo run -p gn-segment --release --example sam31_parity -- candle
```

The example uses dog/bench/coffee cup prompts and a fixed positive point inside each detected
object, with the worker's default refinement. It saves binary mask planes and metadata under
`crates/gn-segment/benchmark-output/<backend>`, reports counts, ordered per-instance IoU, score
differences and stage wall times, and fails if counts differ or any ordered pair has IoU below 0.99.

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


## CUDA optimization work (step 10, incomplete)

The speed/parity acceptance gate is **not met**. The fast F16 option remains explicit,
and `Sam31::load` keeps F32. No worker, qir, gn-perception or gn-diffusion code was changed.
All timings below are local measurements on the RTX 5090, sm_120, CUDA 13, ORT 1.31 CUDA EP.
The new ONNX references were run on CUDA, not reused from CPU. Backends ran in separate
processes, with one warm-up per picture and five measured repetitions in the final runs.

### Running the harness

From the repository root in PowerShell:

```powershell
cmd /c crates\gn-segment\examples\cuda-bench.cmd run -p gn-segment --release --features cuda --example sam31_parity -- cuda 5
cmd /c crates\gn-segment\examples\cuda-bench.cmd run -p gn-segment --release --features cuda --example sam31_parity -- cuda-f16 5
```

`cuda-bench.cmd` applies the vcvars64/CUDA environment from the app build script and uses
`crates/gn-segment/target-cuda` as its own target directory. Reference masks, request
outputs and timing JSON are under the ignored `crates/gn-segment/benchmark-output`.
Both CUDA Candle modes currently exit with a failing gate on cup text; failures are not suppressed.
The CUDA modes default to the recorded `onnx-cuda` reference folder, `candle` to `onnx`.

By default, profiling synchronizes CUDA before and after every recorded stage, and
reports encode, select and point stages separately. Set `GN_SAM31_NO_PROFILE=1` for
wall-time runs without those intrusive stage barriers; complete requests still finish
before their timer stops. Other diagnostic options are `GN_SAM31_OUTPUT_DIR`,
`GN_SAM31_REFERENCE` (a reference subfolder), `GN_SAM31_PICTURE` and `GN_SAM31_ITERATIONS`.
Leave refinement at its default two for acceptance.

### Final request times

Wall-time medians in milliseconds, without stage profiling. Text includes the full-image
encode, detector, crop encode and two refinement passes. Points reuse the full-image cache.
The baseline was measured earlier with synchronized stages (three repetitions), so its
wall times include profiling overhead and some run-to-run variation.

| Picture | Baseline F32 text / point | Final F32 text / point | Experimental F16 text / point | ONNX CUDA text / point |
| --- | ---: | ---: | ---: | ---: |
| st-dog | 1925.1 / 29.4 | 519.0 / 12.6 | 260.6 / 15.3 | 210.6 / 24.3 |
| st-bench | 1772.6 / 19.7 | 515.1 / 9.2 | 258.0 / 12.2 | 198.1 / 16.0 |
| st-cup | 1756.0 / 17.9 | 516.3 / 8.9 | 257.4 / 10.8 | 186.6 / 15.3 |

Synchronized stage medians for **st-dog**, milliseconds. The baseline was rerun
from original source at `7619fd62e5de6d15d9b41f687bdf88c46608130d`, with the final harness
and stage instrumentation added in an isolated, ignored crate. That rerun used five
measured repetitions and the same Candle 0.11 / cudarc 0.19.9 versions.
Detector totals include encoder, decoder and segmentation/scoring heads. Tracker-head
totals include two text-refinement passes and two cached-point passes. Host and mask
rows cover recorded transfer / sampling boundaries; small framework work, feedback input
construction and the final host coverage-bounds scan are not separately attributed.

| Stage | Baseline F32 | Final F32 | Experimental F16 |
| --- | ---: | ---: | ---: |
| Full-image trunk | 515.33 | 205.09 | 87.90 |
| Detector FPN | 29.09 | 28.64 | 14.43 |
| Interactive FPN | 28.41 | 26.81 | 14.16 |
| Text | 6.90 | 6.95 | 7.14 |
| Geometry CLS | 5.69 | 1.20 | 1.77 |
| Detector, all modules | 531.25 | 86.76 | 53.34 |
| Refinement crop trunk | 501.47 | 202.18 | 89.78 |
| Refinement crop FPN | 22.71 | 23.01 | 12.00 |
| Tracker heads, select + point | 16.67 | 13.43 | 19.09 |
| Image preprocessing | 23.17 | 3.02 | 2.33 |
| Crop preprocessing | 9.42 | 0.81 | 0.78 |
| Mask postprocessing | 25.37 | 2.93 | 2.98 |
| Host transfers / materialization | 22.62 | 9.85 | 10.39 |

Synchronized new-picture text medians in the baseline rerun were dog 1737.58, bench 1720.72 and cup 1700.73 ms.
Final synchronized F32 and F16 dog requests were 606.59 and 307.89 ms, respectively.

Incremental dog text measurements (synchronized, three repetitions; different runs):

| Change accumulated | New-picture text ms |
| --- | ---: |
| Original F32 CUDA | 1925.14 |
| Convert compact weights once at load | 1746.04 |
| F16 compute + batched heads / query chunks | 929.36 |
| Half softmax fast path, device feedback / sampling caches, fused small linear bias | 323.15 |
| Bias-inclusive trunk GEMMs | 313.11 |
| Bias-inclusive convolutions and 1x1 channel GEMMs | 310.51 |
| Rejected: F32 trunk QK / softmax | 404.53 |

F32 softmax, higher-precision vision, higher-precision detector and altered residual/norm
boundaries were also measured. None fixed the cup gate; the slower variants were discarded.
These experiments are not cumulative, independently controlled speedup claims.

### Parity and precision

All three pictures return one nonempty instance in the same order. Ordered final mask IoUs
and score differences (Candle minus ONNX CUDA) are:

| Picture | F32 text IoU / score delta | F16 text IoU / score delta | F32 point IoU | F16 point IoU |
| --- | ---: | ---: | ---: | ---: |
| st-dog | 0.999522544 / +0.000078559 | 0.999357086 / +0.000000000 | 0.999946111 | 0.999948805 |
| st-bench | 0.997818986 / -0.000058353 | 0.997430799 / -0.000047326 | 0.999873525 | 0.999861087 |
| st-cup | 0.976301843 / +0.000257075 | 0.981694099 / +0.000114560 | 0.999965231 | 0.999925497 |

Cup text fails the 0.99 gate in both modes. **The unmodified F32 CUDA baseline also
failed against these freshly generated CUDA references: 0.976341992.** This is separate
from the historical CPU/CPU result above. The mismatch is concentrated in the coffee
foam, with logits near zero; bounds alone hide it. No thresholds or picture-specific
corrections were introduced to make the numbers pass.

`Sam31::load` and the default `Checkpoint::open` use F32.
`Sam31::load_with_dtype(path, device, DType::F16)` or
`Checkpoint::open(path)?.with_compute_dtype(DType::F16)?` explicitly selects the
experimental CUDA mode. CPU always computes in F32. Parameters are converted during
loading, including the formerly compact trunk/FPN parameters; forward passes no longer
convert checkpoint weights on each operation.

| Part | Default CPU / CUDA | Experimental CUDA |
| --- | --- | --- |
| Trunk, FPNs, geometry, detector, interactive heads | F32 | F16 |
| CLIP | F32 | F32 |
| Language resizer | F32 | F16 |
| RoPE, Fourier projection, coordinate sine construction | F32 | F32, cast back |
| LayerNorm / spatial GroupNorm statistics | F32 | F32 accumulation, F16 outputs / affine boundaries |
| Masked or biased attention softmax | F32 | F32, cast back for PV |
| Unmasked attention | F32 | F16 scores / exponentials, F32 softmax sum and GEMM accumulation |
| External image / mask interpolation | F32 | F32 |

Bias-inclusive GEMMs append a constant lane and padded zero lanes, so bias accumulates
before F16 output rounding. Pointwise convolutions project NCHW channels directly.
Convolutions with bias append constant input channels and a center-only bias coefficient.
No new CUDA kernels or cuDNN production dependency were added. Nonoverlapping transpose
convolutions already used GEMM before this work and retain that implementation.

Full-image positions, dense prompt PE and antialias filter matrices are cached. CUDA image
preprocessing, ROI feature pooling, detector-mask resizing, feedback, selected tracker
logits, thresholding and the coarse/refined union stay on-device. Only the selected final
mask is downloaded, plus small score/box/IoU metadata required by the public API. Geometry
box coordinates still make a small host metadata read; the feature map does not. Mask
coverage bounds remain a host scan of the required output vector.
The crop-only vision path already skipped the detector FPN before this task.

### Diagnostics and remaining work

The `sam31_stages` diagnostic (FPN, prompt and detector boundaries against the ONNX graphs) was
removed with the ONNX Runtime route on 2026-10-29. Its `cross` run, exact ONNX FPN/prompt inputs
fed into the experimental Candle detector, achieved 0.99223 low-mask IoU with the bias-inclusive
linear variant while the full path still failed.

Candle 0.11's built-in `candle_nn::ops::sdpa` has no CUDA backend here. The optional
`candle-flash-attn` package was not cached. Its fetch failed because the configured
network proxy could not connect to static.crates.io; the unavailable dependency was
removed again. The available implementation batches heads together, bounds score storage
by query chunks and uses Candle's softmax kernels for D=64/32/16.

`sam31_attention` measured window/global attention at 0.6715 / 5.7005 ms and detector
self-attention at 1.9113 ms. Across two 32-block vision passes this represents 83.21 ms,
plus 11.47 ms across six detector self-attention layers. Thus about 94.68 ms is an upper
bound on that measured unfused work that a fused implementation could replace; actual
savings are unmeasured and necessarily smaller. Tracker D=16 token-to-image / reverse
attention measured 0.0920 / 0.2791 ms per call.

Remaining: fix the CUDA cup discrepancy, make fused attention available and measure it,
then rerun the full gate before promoting F16. F16 is still 46?71 ms slower than the
measured ONNX CUDA route. General 3x3 convolution tiling and broader allocation reuse
were not implemented: the remaining FPN/head convolutions were secondary in this profile.

### Validation and changed files

All 32 CPU tests passed, including the ignored full detector and vision fixtures, using
`cargo test -p gn-segment --release -- --include-ignored`. All 13 CUDA unit tests passed. They cover head
dimensions 16/32/64, masks, fully masked rows, chunk tails, fused bias cancellation,
pointwise projection, ROI pooling and convolution padding. CPU and CUDA all-target
Clippy run with `-D warnings`; rustfmt uses the workspace's width of 160.
The GPU composed gate is intentionally reported as failing, as detailed above.

Changed files (all under `crates/gn-segment`; existing integration tests / fixtures unchanged):

- `.gitignore`
- `Cargo.toml`
- `README.md`
- `examples/cuda-bench.cmd`
- `examples/sam31_attention.rs`
- `examples/sam31_parity.rs`
- `examples/sam31_stages.rs`
- `src/attention.rs`
- `src/detector.rs`
- `src/detector_heads.rs`
- `src/geometry.rs`
- `src/lib.rs`
- `src/linear.rs`
- `src/mask_decoder.rs`
- `src/ops.rs`
- `src/position.rs`
- `src/profile.rs`
- `src/prompt.rs`
- `src/sam.rs`
- `src/text.rs`
- `src/vision.rs`
- `src/weights.rs`
- `tools/sam31_observe.py`
