# qwen-image-rs (`qir`)

Rust and CUDA inference for the diffusion models behind ShotoPhopEX's generative tools, built on
[candle](https://github.com/huggingface/candle): Qwen-Image-2.1, Qwen-Image-Edit-2511, Krea 2,
MOSS-Image and the Wan 2.1 VAE, with their text and vision encoders, LoRA merging, and custom
CUDA kernels chosen from profiles and checked against reference outputs.

This crate ships **no weights**. Each model keeps its own licence (Qwen-Image-2.1 is under the
Qwen Research License); see [docs/WEIGHTS.md](docs/WEIGHTS.md).

## Cargo features

| Feature | What it adds |
|---|---|
| (none) | CPU-only build, for tests |
| `cuda` | candle's CUDA backend |
| `convrot` | ConvRot W8A8 INT8 linears: CUTLASS INT8 GEMM with a fused dequant epilogue, fused rotate+quant, merged GEMMs |
| `sage` | vendored SageAttention v1 (INT8 Q·K / FP16 P·V) with RoPE fused into the quant |
| `sage2` | SageAttention2 for sm89+ (INT8 Q·K per-thread / FP8 P·V) |
| `fusednorm` | fused LayerNorm+AdaLN, RMSNorms and VAE norms |
| `fp8` | ComfyUI's `fp8_scaled` files as they are, on cuBLASLt FP8 GEMMs |
| `flash` | exact bf16 attention in one pass |

`CUDA_COMPUTE_CAP` picks the GPU architecture the kernels are built for (`86` RTX 30, `89`
RTX 40, `120` RTX 50); the kernels carry PTX, so a build also runs on newer cards.

## Licence and credits

Apache-2.0 (see [`LICENSE`](LICENSE)). Attributions are in [`NOTICE`](NOTICE): an independent port
of Qwen-Image-2.1 (weights not included); thu-ml SageAttention v1 and v2 (Apache-2.0), vendored
under `kernels/sage/vendor/` with one marked local patch; NVIDIA CUTLASS for the INT8 GEMM;
candle (Hugging Face); the ConvRot scheme follows pytorch/ao's Regular Hadamard rotation.
