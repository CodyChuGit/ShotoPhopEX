# Hardware tiers: long-term goals

The owner's direction (2026-10-29): make every generative feature fast on the RTX 5090 first,
then bring it to smaller machines in tiers. **The current work is the 5090 tier only**; the rest
is recorded here so the 5090 work does not close doors on it.

## The tiers, in order

| Tier | Machine | Memory for models | Weights | Status |
|---|---|---|---|---|
| 1 | RTX 5090 (Blackwell, 32 GB) | ~30 GB | int8 ConvRot everywhere; NVFP4 where it passes the quality check | **in progress** (speed pass, `benchmarks.md` §0) |
| 2 | RTX 3090 (Ampere, 24 GB) | ~22 GB | the same int8 files and kernels; no fp8 or NVFP4 (Ampere has neither) | later |
| 3 | 16 GB NVIDIA cards | ~14 GB | 4-bit: NVFP4 on 50-series, a 4-bit weight-only format on older cards; one model resident at a time | later |
| 4a | Mac, 16 GB unified memory | ~10 GB (macOS and the app share the memory) | MLX grouped 4-bit, fp16 compute; one model resident; tiled VAE | last |
| 4b | Mac, 8 GB unified memory | ~3–4 GB | a "super MOSS" built to fit (below) | last |

## Weight format

- **int8 (ConvRot W8A8) is the common format** for tiers 1 and 2: on the 5090 int8 and fp8
  tensor cores run at about the same rate, and the 3090 has int8 tensor cores but no fp8 ones
  (fp8 weights there only save memory and run at bf16 speed). The worker keeps reading the files
  people have (bf16 or fp8, in ComfyUI's `models` layout) and builds an int8 copy once, cached,
  as it already does for Qwen-Image-2.1 and MOSS. Krea 2, 2511 and Layered still run fp8 today.
- **NVFP4** is an extra speed mode on Blackwell (about twice int8's matmul rate, half the
  memory), never the base format. Each model passes a same-seed check against its int8 output
  before it is turned on.
- The int8 GEMM is moving from CUTLASS to cuBLASLt (the MOSS thread's change), which serves both
  the 5090 and the 3090.
- **Mac** has no int8 or fp8 tensor cores: MLX's own 4/8-bit grouped weights with fp16 math, a
  separate conversion and a separate backend (MLX or oMLX), not the CUDA kernels.

## What fits where

Parameter counts read from the safetensors headers in the models folder (2026-10-29); 4-bit sizes
estimated at about 4.5 bits per parameter (scales included).

| Model | DiT | Text encoder | VAE | 4-bit DiT / encoder | 16 GB Mac | 8 GB Mac |
|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | 7.12B (13.3 GB bf16) | Qwen3-VL 8B, 8.77B (16.3 GB) | 0.34B | 4.0 / 4.9 GB | yes | no |
| Krea 2 Turbo | 12.82B (23.9 GB) | Qwen3-VL 4B, 4.44B (8.3 GB) | 0.13B | 7.2 / 2.5 GB | yes, tight | no |
| MOSS-Image | 5.08B (9.5 GB) | Qwen3.5 9B, 9.65B (18.0 GB) | flux2 small, 0.06B | 2.9 / 5.4 GB | yes | needs a smaller variant |
| Qwen-Image-Edit-2511 | 20.43B (38.1 GB) | Qwen2.5-VL 7B, 8.29B (15.4 GB) | 0.13B | 11.5 / 4.7 GB | no | no |
| Qwen-Image-Layered | 20.43B (38.1 GB) | Qwen2.5-VL 7B | 0.13B | 11.5 / 4.7 GB | no | no |
| SAM 3.1 | 0.87B (1.6 GB f16) | none | none | stays f16 | yes | yes |

Measured working memory: the 2.1 VAE decoding 992×992 in one piece needs 3.8 GB; tiled decode
(already in the code) keeps it under about 1 GB. The DiT working sets are estimates until the
precision experiment records peak VRAM: about 0.5–1 GB for text to image at 1 MP, 1–2 GB with a
reference, 3–5 GB for Layered and 2511 (about 19k image tokens with batched CFG).

## The 8 GB Mac: a "super MOSS"

About 3–4 GB is left for the model's peak after macOS and the app. That needs a DiT of about
2.5 GB or less at 4-bit (about 4–5B parameters, or MOSS at about 3-bit) and a text encoder of
about 3B parameters or fewer: Qwen3.5 9B is 5.4 GB at 4-bit on its own. A model-design question
for the MOSS work.

## What the 5090 work should keep in mind

- One model resident at a time, a tiled VAE and streamed loading are the shape of tiers 3 and 4:
  build them as switches, not forks.
- Nothing Blackwell-only in the common path: NVFP4 and fp8 kernels stay optional, int8 is the
  path every NVIDIA tier shares.
- Record peak VRAM next to every speed number from now on, so the smaller tiers can be planned
  from measurements.
