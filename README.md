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
| `gn-perception` | SAM 3.1 through ONNX Runtime | planned |
| `gn-engine` | the worker process: the protocol (JSON messages, binary pixel frames), the server loop (one client, progress streamed, cancel mid-run, a second client turned away at once), the client (spawn or connect, resumable frame reads, respawn-ready), a fake engine, the `gn-engine` binary | Phase 0: round-trips as a real child process (progress, cancel, kill noticed, 4 MB pictures both ways, token checked); the editor side (`NativeBackend`) follows once the repository is a dependency |

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
Thresholds are in `gn_harness::metrics::Verdict` and in `docs/PARITY.md`.

## Licence

MIT OR Apache-2.0. Ported code keeps its origin in `NOTICE`.
