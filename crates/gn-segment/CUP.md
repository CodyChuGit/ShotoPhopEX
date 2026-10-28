# SAM 3.1 coffee-cup investigation

Measured 2026-10-27, CPU only. **Candle reproduces ComfyUI's f32 binary mask exactly when both receive identical decoded pixels**, before refinement and after two refinement iterations. The installed ONNX route is mixed precision and differs already in the coarse detector mask. Its final CPU IoU against that reference is **0.995583538**, reproducing the reported result.

There is also an independent JPEG-decoding difference: ComfyUI/Pillow and Rust's `image` crate do not supply identical RGB pixels from this JPG. Running each application's usual JPEG loader gives substantially different masks. That must be controlled before adjudicating model parity. This experiment establishes parity with ComfyUI f32, not semantic ground truth or CUDA parity.

## Runs and reference

Input: `C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-cup.jpg`, 1920 × 1280; prompt `coffee cup`; threshold strictly greater than 0.5; highest scoring detection only; two refinement passes; final mask is coarse OR refined, thresholded at logit zero. Refinement-off results use zero passes.

The script uses the requested embedded Python with `-I`, adds ComfyUI to `sys.path`, enables its CLI argument parsing with `--cpu --use-pytorch-cross-attention`, loads the specified fp16 checkpoint with model and text compute explicitly f32, and executes **`comfy_extras.nodes_sam3.SAM3_Detect.execute`** under inference mode. PyTorch is `2.14.0+cu130`; the actual device is asserted CPU. The CUDA build suffix does not indicate CUDA execution. No export wrappers replace the ComfyUI model.

An observation wrapper captures the node's selected coarse logits and invokes its own `_refine_mask(..., iterations=0)` for the coarse binary mask. This is the same refinement-off branch the node uses, without another detector run. Both actual `forward_segment` refinement outputs are saved. Native routes use the unchanged `sam31_parity` release example, `candle` and `onnx` modes, once each at zero and two passes. Its additional point-prompt work also ran but is excluded from the text-mask results below. All processes inherit `CUDA_VISIBLE_DEVICES=""`.

Two ComfyUI inputs were evaluated:

- **Comfy/Pillow:** the normal `PIL.Image.open(...).convert("RGB")` JPEG decode.
- **Comfy/shared RGB:** the exact Rust `image::open(...).to_rgba8()` bytes used by both native routes, converted to RGB and f32 in ComfyUI. A tiny decoder is compiled from stdin against the already-built `image` library; its executable and raw pixels stay under `target/`. No Rust source file is added or modified.

The decoders differ at **724,662 pixels** / 1,255,696 channels. Maximum channel difference is **5/255**, mean absolute channel difference **0.235796712/255**. Holding the model fixed and swapping only those input bytes changes the final ComfyUI mask at 6,030 pixels.

## Pairwise mask IoUs

All binary comparisons use original-resolution masks, intersection divided by union. Bounds below use original-image `(x0,y0,x1,y1)` coordinates with exclusive upper bounds; `none` means identical masks.

| Input/reference comparison | Coarse IoU, 0 passes | Final IoU, 2 passes |
| --- | ---: | ---: |
| Comfy/Pillow vs candle | 0.938102142 | 0.938205183 |
| Comfy/Pillow vs ONNX | 0.940684434 | 0.940272739 |
| candle vs ONNX | 0.996278466 | 0.995583538 |
| Comfy/shared RGB vs candle | **1.000000000** | **1.000000000** |
| Comfy/shared RGB vs ONNX | 0.996278466 | 0.995583538 |

| Comparison | Coarse differing pixels / bounds | Final differing pixels / bounds |
| --- | --- | --- |
| Comfy/Pillow vs candle | 6,020 / (264,633,908,1097) | 6,030 / (262,633,894,1111) |
| Comfy/Pillow vs ONNX | 5,751 / (264,633,908,1097) | 5,812 / (262,633,914,1111) |
| candle vs ONNX | 361 / (272,660,736,1092) | 430 / (262,639,914,1111) |
| Comfy/shared RGB vs candle | 0 / none | 0 / none |
| Comfy/shared RGB vs ONNX | 361 / (272,660,736,1092) | 430 / (262,639,914,1111) |

The final candle/shared-RGB mask has 97,311 foreground pixels; ONNX has 96,985; Comfy/Pillow has 91,821. Of the final 430 candle/ONNX disagreements, 378 are candle-only and 52 ONNX-only. Visually these are mostly sparse differences over the coffee/foam, with some rim/edge differences, rather than a different object being selected. The JPEG-loader differences are much denser over the cup's interior.

The PNGs crop the cup for visibility; titles identify red-only and cyan-only pixels. Top row: ordinary Comfy/Pillow comparisons and candle/ONNX. Bottom row: shared-RGB controls.

![Final disagreements](../../target/sam31-parity/cup/disagreement-2.png)

[Coarse disagreement PNG](../../target/sam31-parity/cup/disagreement-0.png)

## Scores and divergence stage

Detector scores are unchanged by the refinement setting:

| Route | Selected detector probability | Pixel-space detector xyxy box |
| --- | ---: | --- |
| Comfy/Pillow f32 | 0.937667429 | (264.315735,631.520691,916.326843,1109.788208) |
| Comfy/shared RGB f32 | 0.937583029 | (264.299561,631.493652,916.293640,1109.762085) |
| candle CPU f32 | 0.937582910 | (264.299622,631.493774,916.293762,1109.762207) |
| ONNX CPU | 0.937554777 | (264.140625,631.562500,915.937500,1110.000000) |

Candle differs from Comfy/shared RGB in detector probability by only -1.19209e-7, and its binary coarse/final masks have zero pixel differences. Against ONNX, the coarse masks already disagree at **361 pixels**, so refinement is not the origin. Of those disagreements, 360 persist, one disappears and 70 new disagreements appear after refinement. Refinement adds 322 pixels to candle's coarse mask and 329 to ONNX's; it removes none because the worker unions with the coarse mask.

The integer refinement image crop also differs: Comfy and candle use **(199,583,981,1157)**, whereas ONNX uses **(198,583,981,1157)**. ONNX's slightly different box crosses the left crop's integer truncation boundary. Thus refinement receives both different coarse feedback and a crop one pixel wider. This experiment does not isolate how much each contributes to the 70 new disagreement pixels.

The installed graphs are **not an all-f32 oracle**, even on CPU. Inspection found FLOAT16 initializers and explicit FLOAT16 Cast targets:

| Graph | FLOAT16 initializers | FLOAT initializers | Casts to FLOAT16 |
| --- | ---: | ---: | ---: |
| vision | 514 | 3 | 4 |
| geometry | 99 | 1 | 3 |
| decoder | 487 | 14 | 26 |
| heads | 148 | 5 | 8 |
| text | 2 | 393 | 1 |

This explains why an f32 module parity result does not imply equality to these exports. Precision changes are a supported explanation for the native-route discrepancy; the first offending internal operation was **not** isolated. No all-f32 ONNX re-export or CUDA run was performed. The reported 0.976–0.982 CUDA results cannot be adjudicated numerically by this CPU-only experiment. For an f32 correctness reference, use ComfyUI with identical input pixels; use the deployed ONNX output separately if compatibility with that mixed-precision route is the requirement.

## Reproduction, memory and artifacts

From the repository root in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
cargo build -p gn-segment --example sam31_parity --release
& C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe -I crates/gn-segment/tools/sam31_cup.py run
```

The script runs inference processes sequentially and samples working set and private memory every 100 ms, with a 14.5 GB per-process guard. In this investigation the extra shared-RGB diagnostic overlapped part of the last candle run; the sum of their independently measured peak working sets is **12.293 GB**, still below 15 GB. Reported figures are decimal GB:

| Process | Seconds | Peak working set GB | Peak private GB |
| --- | ---: | ---: | ---: |
| Comfy/Pillow | 33.31 | 7.129 | 10.507 |
| Comfy/shared RGB | 47.31 | 7.150 | 10.461 |
| ONNX, 0 passes | 12.47 | 4.911 | 4.018 |
| candle, 0 passes | 66.97 | 5.032 | 5.193 |
| ONNX, 2 passes | 18.94 | 5.514 | 4.595 |
| candle, 2 passes | 146.23 | 5.142 | 5.193 |

These are diagnostic single-run times, not benchmark medians. All inference runs exited successfully. Sampling may miss very short memory spikes.

Artifacts are left in [`target/sam31-parity/cup/`](../../target/sam31-parity/cup/):

- `comfy-{0,2}.{bin,png}`: ordinary ComfyUI/Pillow masks.
- `native-input/comfy-{0,2}.{bin,png}`: ComfyUI on identical Rust-decoded pixels.
- `{candle,onnx}-{0,2}.{bin,png}`: copies of native binary masks plus PNGs; original example outputs and score metadata are in `iter{0,2}/{candle,onnx}/`.
- `comfy-coarse-logits.npy`, `comfy-refinement-{1,2}-logits.npy`, with corresponding shared-RGB files in `native-input/`.
- `metrics.json`, `jpeg-difference.json`, process logs, memory/timing JSON, and `provenance.json` including image/checkpoint/graph/external-weight SHA-256 values.

Every `.bin` is 1920 × 1280 row-major uint8, values 0 or 1. `compare` regenerates metrics and PNGs; `audit` regenerates provenance. Revisions: genai-native `7eefc851071b64eac602c9beb20cf92b19a3df68`, ComfyUI `b0b743566f65daafc423b4fea8a2fbda94b3384a`.

Only the investigation script and this report are added outside ignored build/output artifacts. Crate sources are unchanged; nothing was committed. Cargo updated the lockfile during the build; its original bytes were restored afterward.
