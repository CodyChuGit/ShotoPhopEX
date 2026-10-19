# Exporting SAM 3.1 for `gn-perception`

`gn-perception` runs SAM 3.1 through ONNX Runtime from graphs exported on the user's own
machine, from the user's own checkpoint. Nothing here ships weights: the application ships the
export script, the runtime loader and this note.

## Source

The graphs are traced from **ComfyUI's SAM 3.1 modules** (`comfy/ldm/sam3/`, the dependency-free
port ComfyUI uses for its `SAM3_Detect` node), loaded from `checkpoints/sam3.1_multiplex_fp16.safetensors`
(`Comfy-Org/sam3.1` on Hugging Face, ungated; the same file ComfyUI uses). That is deliberate: the
references the editor is measured against were made with that node, and a graph traced from the
same modules reproduces them to the pixel. The Hugging Face `transformers` port is a different
implementation (it feeds the trunk ImageNet-normalised pixels, has no point path in its detector)
and is not what we match.

## Licence

The checkpoint is Meta's, under the **SAM License** (research and commercial use allowed; no
military, nuclear, espionage or weapons end uses; the licence travels with any redistribution;
Meta may change the terms). The exported graphs are a derivative of those weights and carry the
same licence. Do not redistribute them; export them where they are used. The export script and
`gn-perception` are MIT OR Apache-2.0.

## Command

Run with ComfyUI's own Python (it has torch; add `onnx` and `onnxscript` for the exporter):

```powershell
$py = "<ComfyUI portable>\python_embeded\python.exe"
& $py -m pip install onnx onnxscript
& $py -s tools\export_sam31.py --comfy <ComfyUI dir> --out <models root>\sam3.1 --fp16 --mha --image <any png>
```

`--fp16 --mha` is the production set (half precision, ORT's fused `MultiHeadAttention` for the
trunk). Without them the graphs are fp32 with the exporter's own attention: slower, bit-closer to
the fp32 modules, useful as a reference. `--verify` (fp32 only) compares each small graph with
torch through `onnxruntime` on the CPU. `--parts` picks graphs; `--opset` defaults to 18 (23's
`Attention` op ran 40× slower on ORT 1.31's CUDA provider).

What comes out (sizes with `--fp16 --mha`):

| File | Size | Role |
|---|---:|---|
| `vision.onnx` + `.data` | 0.9 GB | trunk (ViTDet-H, 446 M parameters), detector FPN, tracker FPN, the geometry CLS token; one pass per picture |
| `text.onnx` + `.data` | 1.35 GB | CLIP text encoder + the language resizer (fp32 always) |
| `geometry.onnx` + `.data` | 17 MB | box prompts for the detector |
| `decoder.onnx` + `.data` | 55 MB | DETR encoder/decoder, scoring, segmentation head |
| `heads.onnx` + `.data` | 9 MB | the tracker's prompt encoder + mask decoder (points, boxes, mask refinement) |
| `tokenizer.json` | 3.6 MB | the CLIP byte-pair vocabulary (copy it in; any CLIP ViT-L tokenizer file) |

`--parts tracker` adds `vision-tracker.onnx` (trunk + tracker FPN only, for refinement crops). On an
RTX 5090 it saved 5 ms per crop for 0.9 GB more on disk and in VRAM; the default set leaves it out
and `gn-perception` uses it only when present.

## Reproducibility

`torch.onnx.export(dynamo=True)` is deterministic for a given torch / onnxscript version; the
weights inside are the checkpoint's, cast to fp16 where asked. Record the versions used
(`torch 2.14.0+cu130`, `onnxscript 0.7.2`, `onnx 1.23.2` for the 2026-10-19 set) rather than hashes:
a byte-identical re-export across versions is not guaranteed, a numerically equivalent one is.
The parity harness (`gn-bench run --engine sam-ort`, then `compare` against the ComfyUI references)
is the acceptance test of any export: 16 of 16 SAM cases at bounds IoU ≥ 0.99.

## Why `in_training = True`

ComfyUI routes RoPE through its `comfy_kitchen` custom op outside training. That op has no ONNX
translation; the training flag selects the plain-torch implementation of the same arithmetic.
