# Models on disk: what the files actually look like

Read off the real files in the owner's ComfyUI install with `gn-inspect` on 2026-10-19 (headers
only). These are the conventions `gn-weights` identifies by (`index::identify`) and the
conventions the loader's key maps will translate. Every claim below is a tensor name or shape
seen in a file; nothing is inferred from a file name.

## ComfyUI's quantised repacks (`comfy_quant`)

Each quantised linear layer carries a `<layer>.comfy_quant` tensor: a `U8` array holding a JSON
descriptor, and the file header's `__metadata__` carries `_quantization_metadata` =
`{"format_version": "1.0", "layers": {"<layer>": {"format": …, …}}}` naming every layer.

| Format (`comfy_quant`) | Weight | Scales | Seen in |
|---|---|---|---|
| `{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}` | `weight` F8E4M3 `[out, in]` | `weight_scale` F32 `[]` (per tensor); the "fp8mixed" files add `input_scale` F32 `[]` (a static activation scale) and keep `bias` BF16 | `qwen_image_edit_2511_fp8mixed`, `qwen_image_layered_fp8mixed`, `krea2_turbo_fp8_scaled`, `qwen3vl_4b_fp8_scaled`, `qwen_2.5_vl_7b_fp8_scaled` (that one has no `comfy_quant` tensors, a `scaled_fp8` marker tensor instead) |
| `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` | `weight` I8 `[out, in]` | `weight_scale` F32 `[out, 1]` (per row); the Hadamard rotation is implied by the format (group size 256), no rotation tensor is stored | `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot` (the embedding too: `model.embed_tokens.weight` I8 `[151936, 4096]` with `[151936, 1]` scales) |
| `{"format": "nvfp4", …}` | `weight` U8 `[out, in/2]` (two FP4 e2m1 values per byte) | `weight_scale` F8E4M3 `[out, in/16]` (block scales, 16 elements per block), `weight_scale_2` F32 `[]` (global scale; 0.00126 on `blocks.0.attn.wk`) ; the 2511 encoder's NVFP4 layers add `input_scale` F32 `[]` | `krea2_turbo_nvfp4` (the fp8 scale tensors of the `_scaled` file stay for the non-NVFP4 layers), `qwen_2.5_vl_7b_nvfp4` (NVFP4 linears, fp8 embeddings and head) |

`__index_timestep_zero__` (F32, shape `[0]`) is a sentinel ComfyUI writes into its 2511 repacks.

## Families, by tensor names and shapes

| Family | Markers | Hidden / blocks | Files here |
|---|---|---|---|
| Krea 2 transformer | prefixes `blocks`, `txtfusion` (`layerwise_blocks`, `refiner_blocks`), `first`, `last`, `tmlp`, `tproj`, `txtmlp`; `blocks.N.attn.{wq,wk,wv,wo,gate}`, `attn.qknorm.{qnorm,knorm}.scale` | `attn.wk` `[1536, 6144]` → model width 6144, kv width 1536; text fusion width 2560 (= Qwen3-VL-4B's hidden) | `krea2_turbo_fp8_scaled` (686 tensors), `krea2_turbo_nvfp4` (942) |
| Qwen-Image 20B (2509 / 2511 / Layered) | prefixes `transformer_blocks`, `img_in`, `txt_in`, `txt_norm`, `time_text_embed`, `norm_out`, `proj_out`; `attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out,to_add_out}` with biases | `img_in.weight` `[3072, 64]`; 60 blocks | `qwen_image_edit_2511_fp8mixed` (4 451 tensors), `qwen_image_layered_fp8mixed` (3 612) |
| Qwen-Image-2.1 | as above plus prefix `modulation` (`modulation.1.weight` `[16384, 4096]`), no biases | `img_in.weight` `[4096, 64]`; 32 blocks (640 block tensors) | `qwen_image_2.1_int8_convrot` (649) |
| Qwen3-VL 4B text encoder | prefix `model` only; `model.embed_tokens.weight` `[151936, 2560]`; `model.layers.N.mlp.{gate,up,down}_proj`; `model.visual.*` present (vision tower under `model`) | 2560 | `qwen3vl_4b_fp8_scaled` (1 217) |
| Qwen3-VL 8B text encoder | prefixes `model`, `lm_head`; `model.embed_tokens.weight` `[151936, 4096]` | 4096 | `qwen3vl_8b_int8_convrot` (1 258) |
| Qwen2.5-VL 7B text encoder | prefixes `model`, `visual` (vision tower at top level), `lm_head`; `model.embed_tokens.weight` `[152064, 3584]` | 3584 | `qwen_2.5_vl_7b_fp8_scaled` (1 446), `qwen_2.5_vl_7b_nvfp4` (1 517) |
| Wan 2.1 VAE (Qwen-Image's and Krea 2's) | prefixes `encoder`, `decoder`, `conv1`, `conv2`; 3-D convs `[…, k, k, k]`; `decoder.head.2.weight` `[3, 96, 3, 3, 3]`; `encoder.head.2.weight` `[32, 384, 3, 3, 3]` (16 latent channels × 2) | 194 tensors | `qwen_image_vae` |
| Layered VAE (RGBA) | the same architecture; `decoder.head.2.weight` `[4, 96, 3, 3, 3]` | 194 tensors | `qwen_image_layered_vae` |
| Qwen-Image-2.1 VAE | `modelspec.architecture = qwen_image_2.1_vae` in the metadata; `decoder.head.2.weight` `[4, 144, 1, 3, 3]`; `encoder.head.2.weight` `[128, 768, 1, 3, 3]` (64 latent channels × 2); `conv1.weight` `[128, 128, 1, 1, 1]`, `conv2.weight` `[64, 64, 1, 1, 1]` | 238 tensors | `qwen_image_2.1_vae_bf16` |
| SAM 3 / 3.1 | prefixes `detector` (1 133: `backbone.language_backbone.encoder.*` is a CLIP-style text tower, 1024 wide, 49 408 tokens) and `tracker` (457) | F16 throughout | `sam3.1_multiplex_fp16` (1 590) |
| Lightning LoRA (2511) | `transformer_blocks.N.attn.<proj>.{lora_down.weight [64, 3072], lora_up.weight [3072, 64], alpha []}` | rank 64 | the two `Qwen-Image-Edit-2511-Lightning-*steps-V1.0-bf16` files (2 160 tensors each) |

## What this means for the loader

- The ComfyUI repacks are self-describing: `_quantization_metadata` says the format per layer,
  `comfy_quant` repeats it per tensor. The loader reads the header's description and never
  guesses from a file name.
- The three quantisations map to three kernel paths: fp8 per-tensor scale (one GEMM with a
  scalar scale), INT8 per-row scale with a Hadamard rotation of the activations (group 256), and
  NVFP4 block-scaled (16-element blocks, F8 scales, one global F32). All three keep norms, biases
  and embeddings (except the 8B encoder's) in BF16.
- Official diffusers layouts differ in naming (and split into shards); the key maps per family
  come with the first native model definition.
