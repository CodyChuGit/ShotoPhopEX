# genai-native

The native generative engine of [ShotoPhopEX](../README.md): the models the editor uses, running
in-process on an NVIDIA GPU from Rust (candle plus our own CUDA kernels), with no Python and no
ComfyUI. The editor starts it as a worker process (`photocraft --engine`) and talks to it through
the `gn-engine` protocol.

**Bring your own models.** This repository ships no weights. It reads official safetensors,
ComfyUI's repacks (fp8-scaled, INT8 ConvRot) and GGUF from a folder in ComfyUI's `models` layout.

## Crates

| Crate | Owns |
|---|---|
| `gn-engine` | The worker process: the protocol (JSON messages, binary pixel frames, token-authenticated greeting), the server loop (one client, streamed progress, cancel mid-run), the client, `Multi` (several engines behind one worker) and a fake engine for tests. |
| `gn-weights` | The bring-your-own-model loader: bounds-checked safetensors and GGUF readers (headers on open, tensor ranges on demand), ComfyUI's repack conventions, and a folder index that identifies model families from their tensors. |
| `qir` (`qwen-image-rs`) | Model code and kernels: Qwen-Image-2.1, Qwen-Image-Edit-2511, Krea 2, MOSS-Image, the Wan 2.1 VAE, Qwen text and vision encoders, LoRA merging; INT8 ConvRot GEMMs, SageAttention, fp8, flash attention and fused norms in CUDA. |
| `gn-diffusion` | The worker-side diffusion engine over `qir`: image, edit, fill, expand and matte requests, one model family resident at a time. |
| `gn-segment` | SAM 3.1 (text, point and box prompts) and ViTMatte on candle. |
| `gn-perception` | The worker-side perception engine over `gn-segment`. |
| `gn-harness` | The parity and speed harness (`gn-bench`): fixtures, engines, metrics (PSNR, SSIM, mask IoU) and reports. |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the contracts between them and
[`docs/MODELS-ON-DISK.md`](docs/MODELS-ON-DISK.md) for how model folders are recognised.

## Building

The CUDA build needs nvcc (CUDA 13), the CUTLASS headers and, on Windows, the MSVC environment;
the root [README](../README.md#-install) has the exact steps. Without the `cuda` features the
crates build and test on the CPU.

## Licence

MIT OR Apache-2.0. Ported code keeps its origin in [`NOTICE`](NOTICE).
