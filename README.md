# genai-native

The native generative engine for [PhotoshopEX](https://github.com/CodyChuGit/comfy-photocraft)
(branch `photoshopex`): one executable, no Python, the models the editor cares about running
in-process on the GPU, built to
[`docs/comfy/plan-native-engine.md`](https://github.com/CodyChuGit/comfy-photocraft/blob/photoshopex/docs/comfy/plan-native-engine.md)
in that repository. Windows and Linux first, macOS as a later kernel backend.

**Bring your own models.** This repository ships no weights and reads whatever you have: official
safetensors, ComfyUI's repacks, GGUF.

## Layers

| Crate | Owns | Status |
|---|---|---|
| `gn-harness` | fixtures, engines, metrics (PSNR, SSIM, mask IoU, boundary F), reports; the `gn-bench` tool | Phase 0: the ComfyUI route through PhotoshopEX's CLI runs; sd.cpp and the native worker follow |
| `gn-kernels` | the GPU: a `Kernels` trait, CUDA first (sm_120, then sm_89) | planned |
| `gn-blocks` | attention, MMDiT streams, AdaLN, RoPE, flow-match schedulers, VAE and encoder families, LoRA | planned |
| `gn-models` | thin definitions: Krea 2, Qwen-Image-2.1, … | planned |
| `gn-weights` | the BYOM loader: hardened safetensors and GGUF readers (headers only on open, ranges on demand), ComfyUI's repack conventions (`comfy_quant`: fp8-scaled, INT8 convrot, NVFP4), a folder index that identifies families from the tensors; the `gn-inspect` tool | Phase 0: readers and the index identify all 16 files of the owner's install ([`docs/MODELS-ON-DISK.md`](docs/MODELS-ON-DISK.md)); key maps and dequantisation come with the first model definition |
| `gn-perception` | SAM 3.1 through ONNX Runtime (our own export of ComfyUI's modules, `tools/export_sam31.py`); the worker-side engine behind `engine`; the `gn-sam` tool | Phase 1: 16/16 at parity with the ComfyUI route, serving PhotoshopEX's native worker since 2026-10-19 |
| `qir` (`qwen-image-rs`) | the imported Qwen-Image-2.1 port (candle + CUDA kernels: INT8 ConvRot GEMMs, SageAttention, fused norms); `comfy` (ComfyUI's noise, schedule and timestep conventions), `pipeline` (the resident library pipeline), the CLI; since Phase 3 the VAE encoder, the Qwen3-VL vision tower, the multimodal text encoder and the DiT's reference sequence with its prefix cache (`edit`) | Phase 2: 0.116 s/step on the 5090; at equal seed the ComfyUI route's picture (24 of 28 t2i cases at the gate, as many as ComfyUI's INT8 and bf16 routes agree with each other). Phase 3: edits with reference pictures at the same parity (the fox's winter 40 dB), 3.8 s of sampling for a 1024² edit with the prefix cache |
| `gn-diffusion` | the worker-side engine for `qwen-2.1/image`, `qwen-2.1/edit` and `qwen-2.1/fill` over `qir::pipeline`: the snapshot under `<models>/qwen-image-2.1/`, resident after the first request | Phase 2: serving PhotoshopEX's worker (feature `native-diffusion`), routed under `auto` since 2026-10-20; Phase 3: the edit and fill templates the same day |
| `gn-engine` | the worker process: the protocol (JSON messages, binary pixel frames), the server loop (one client, progress streamed, cancel mid-run, a second client turned away at once), the client (spawn or connect, resumable frame reads, respawn-ready), `Multi` (several engines behind one worker), a fake engine, the `gn-engine` binary | Phase 0: round-trips as a real child process (progress, cancel, kill noticed, 4 MB pictures both ways, token checked); PhotoshopEX's `NativeBackend` drives it |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/PARITY.md`](docs/PARITY.md) and
[`docs/PORTING.md`](docs/PORTING.md).

## The harness

```powershell
cargo build --release -p gn-harness
# references from the product as it is today (ComfyUI running or app-managed):
target\release\gn-bench run --engine comfy --suite t2i --fixtures crates\gn-harness\fixtures `
  --images C:\Users\5090\Documents\PhotoshopEX\docs\comfy\bench\images `
  --cli C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe `
  --out references\2026-10-18-comfy-t2i --template krea2-turbo/image
target\release\gn-bench compare references\2026-10-18-comfy-t2i runs\<engine>-t2i
```

Suites: `t2i` (28 cases: 12 prompts × 2 seeds at 1024², 4 at other aspects), `edit` (8, two with
references), `fill` (8, rectangles as fractions), `sam` (16: text, point, negative point, box).
Thresholds are in `gn_harness::metrics::Verdict` and in `docs/PARITY.md`. The native route is
measured two ways: `--engine qir` drives `qwen-image-rs` directly (t2i and edit cases), and
`--engine comfy --pref integrations.engine=native --pref integrations.runtimePath=… --pref
integrations.modelsPath=…` drives the app's own CLI with its native worker, which is what a fill
needs (the app builds the request: the composite, the feathered mask, the prompt wrapper).

## Licence

MIT OR Apache-2.0. Ported code keeps its origin in `NOTICE`.
