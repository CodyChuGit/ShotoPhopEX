# R3 porting spec: Qwen-Image-Layered on candle

> Written by Codex (`codex exec`, read-only, 2026-10-27) for the Rust rebuild (`docs/comfy/goals/rust-rebuild.md` R3), from ComfyUI's model code, the installed checkpoints' headers and genai-native's `qir`. Checked by Claude before any code: the block counts and the fp8mixed layout match the headers; line estimates are guesses.

**This is feasible, but it needs a separate Qwen-Image DiT implementation.** The biggest reusable component is qir’s existing `model/wan_vae.rs`: the layered VAE can use its single-frame machinery after adding RGBA support. The FP8 tensor reader is also reusable, with qualifications below.

This was a read-only source and checkpoint inspection. No files were modified, and no inference parity runs were performed. All line-count estimates below are **engineering guesses**.

For compact citations, paths use these roots:

- `Q/` = `C:\Users\5090\Documents\genai-native`
- `C/` = `C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI`
- `P/` = `C:\Users\5090\Documents\PhotoshopEX`

**1. Architectural and workflow differences**

| Area | Qwen-Image-Layered | Qwen-Image-2.1 in qir |
|---|---|---|
| DiT | **60 dual-stream blocks**, width **3072**, **24 × 128** heads | **32 single-stream blocks**, width **4096**, **32 × 128** heads |
| Text conditioning width | 3584 | 4096 |
| Latent | `[B,16,L+1,H/8,W/8]` | `[B,64,H/16,W/16]` |
| Image tokenization | Spatial **2×2 patches**, producing 64 values/token | One 64-channel latent cell/token |
| Attention | Joint bidirectional attention over text, generated slices and reference tokens | Segment-based causal conditioning; target image attends to its prefix |
| FFN | Separate image/text GELU FFNs, expansion 4 | Shared-stream SwiGLU, expansion 3 |
| Modulation | Separate per-block image/text shift, scale and gate projections | Shared timestep modulation producing four scale/gate rows |
| VAE | Wan 2.1-style, 16 latent channels, 8× spatial compression, RGBA | Wan 2.2-style residual architecture, 64 latent channels, 16× compression, RGBA |
| Encoder | Qwen2.5-VL 7B | Qwen3-VL 8B |

Sources: Layered defaults and construction at `C/comfy/ldm/qwen_image/model.py:346`, FFNs at `:27`, dual-stream blocks at `:211`; qir configuration at [model/config.rs:48](C:/Users/5090/Documents/genai-native/crates/qir/src/model/config.rs:48), block at `Q/crates/qir/src/model/dit.rs:894`, and projections at `:1076`.

I also inspected the **installed checkpoint headers**, independently confirming 60 blocks, `img_in.weight=[3072,64]`, `proj_out.weight=[64,3072]`, `txt_norm.weight=[3584]`, and an additional timestep embedding of `[2,3072]`. Binary-header observations have no source line numbers.

**Multiple-layer positions and conditioning**

The generated latent has **L+1 temporal slices**. The slices are jointly denoised; layers are not additional latent channels. `EmptyQwenImageLayeredLatentImage` explicitly allocates `[B,16,layers+1,H/8,W/8]`. (`C/comfy_extras/nodes_qwen.py:232`.)

The DiT packs each slice’s spatial 2×2 patches into tokens. Its three RoPE axes have dimensions `[16,56,56]` and theta 10,000, as in qir, but the **position assignment differs**:

- Generated slices receive temporal positions `0…L`.
- Their spatial coordinates are centered using `row-floor(grid_height/2)` and `column-floor(grid_width/2)`.
- Layered checkpoint detection selects `default_ref_method="negative_index"`: the first single-frame reference gets temporal coordinate **−1**, then −2 for another reference.
- Text positions start at half the larger target token-grid dimension and increase equally on all three axes.

See [Comfy model.py:408](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:408), `:469`, `:504`, and `C/comfy/model_detection.py:998`. qir’s existing layout instead interleaves text/reference segments and advances a running position: `Q/crates/qir/src/model/dit.rs:1282`.

The template uses **plain `CLIPTextEncode`**, then attaches the same VAE-encoded picture to **both positive and negative conditioning** through `ReferenceLatent`. It does not send the picture through the text encoder’s vision tower. References are normalized into Wan21 sampler space before reaching the DiT. (`P/crates/genai/workflows/qwen-layered-split.json:37`; `C/comfy_extras/nodes_edit_model.py:24`; `C/comfy/model_base.py:2643`.)

Layered adds a learned two-entry timestep-condition embedding. With this template, no explicit `additional_t_cond` is supplied, so ComfyUI selects **entry zero**. References use the ordinary timestep modulation in this route, rather than the special zero-timestep reference path. (`C/comfy/ldm/qwen_image/model.py:78`, `:469`, `:497`.)

**VAE and production of RGBA layers**

The installed layered VAE header shows:

```text
encoder.conv1.weight       [96,4,3,3,3]
decoder.conv1.weight        [384,16,3,3,3]
decoder.head.2.weight       [4,96,3,3,3]
```

Thus it really encodes and decodes **four image channels**. ComfyUI selects its Wan 2.1 implementation with width 96, multipliers `[1,2,4,4]`, 16 latent channels and temporal-downsampling flags `[false,true,true]`. It pads RGB input with opaque alpha. (`C/comfy/sd.py:863`, `:1154`.)

The model’s temporal axis represents the generated slice stack, but the template deliberately avoids decoding that stack as a video:

```text
sample [B,16,L+1,h,w]
    → discard temporal slice 0
    → [B,16,L,h,w]
    → move slices into batch
    → [B×L,16,1,h,w]
    → decode each as an independent RGBA image
```

The “whole picture” interpretation of slice zero is documented in the template; the code directly establishes that it is discarded. See [template:43](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-layered-split.json:43), `C/comfy_extras/nodes_latent.py:195`, `:235`.

Consequently, **full temporal VAE convolution/caching is unnecessary for parity with this template**. At `T=1`, ComfyUI uses causal-zero padding and skips temporal resampling state; qir’s Wan VAE already implements the corresponding last-temporal-tap simplification. (`C/comfy/ldm/wan/vae.py:31`, `:102`, `:491`; `Q/crates/qir/src/model/wan_vae.rs:8`.)

Normalization is Wan21’s fixed 16-channel mean/std, already present in qir’s Wan VAE—not the 2.1 VAE’s 64-channel statistics. (`C/comfy/latent_formats.py:755`; `Q/crates/qir/src/model/wan_vae.rs:20`, `:331`.)

**Text encoder**

| Parameter | Qwen2.5-VL 7B | qir Qwen3-VL 8B |
|---|---:|---:|
| Decoder layers | 28 | 36 |
| Hidden width | 3584 | 4096 |
| FFN width | 18944 | 12288 |
| Query / KV heads | 28 / 4 | 32 / 8 |
| Head dimension | 128 | 128 |
| RoPE theta | 1,000,000 | 5,000,000 |
| Q/K/V biases | Present | Absent |
| Per-head Q/K normalization | Absent | Present |

Sources: `C/comfy/text_encoders/llama.py:400`; [qir text_encoder.rs:73](C:/Users/5090/Documents/genai-native/crates/qir/src/model/text_encoder.rs:73), `:185`.

There is also an important **output-selection difference**: ComfyUI’s Qwen2.5 route uses the final normalized decoder output; qir’s Qwen3 encoder intentionally returns the output **before** final normalization. The existing behavior cannot simply be retained. (`C/comfy/sd1_clip.py:279`; `C/comfy/text_encoders/llama.py:1049`; `Q/crates/qir/src/model/text_encoder.rs:8`.)

Port the Qwen2.5 system template and token-dependent system-prefix removal exactly. ComfyUI locates the second `<|im_start|>` and conditionally skips the following user-role tokens. (`C/comfy/text_encoders/qwen_image.py:17`, `:61`.)

**Sampler and application contract**

The template specifies:

- `ModelSamplingAuraFlow`, **shift 1.0**.
- Euler / `simple`, **20 default steps**, **CFG 2.5**, denoise 1.0.
- Batch size one; one returned PNG per retained slice.

See `P/crates/genai/workflows/qwen-layered-split.json:15`, `:40`.

AuraFlow replaces the original sampling object with `ModelSamplingDiscreteFlow + CONST`, multiplier **1**, using a **1000-entry** sigma table. At shift one, sigma is the unshifted timestep fraction. For 20 simple steps, the schedule is `1,.95,…,.05,0`. qir already supports this schedule family through `Shift::Snr`, but its Qwen2.1 pipeline hardcodes Flux shift 0.69 and its own timestep-rounding convention. Those pipeline choices must change. (`C/comfy_extras/nodes_model_advanced.py:132`, `:163`; `C/comfy/model_sampling.py:308`; `Q/crates/qir/src/comfy.rs:154`, `:194`; `Q/crates/qir/src/pipeline.rs:623`.)

The actual engine command is **`generate.splitLayers`**. It defaults to three layers, allows 1–8, defaults to sampling the document composite, caps the longer side at 640 and rounds dimensions to multiples of 16. Returned images are resized to the original document dimensions and inserted in order, background first, above the active layer, preserving model alpha. (`P/crates/engine/src/generate_cmds.rs:421`, `:4123`, `:4137`, `:4178`, `:4224`.)

**2. Reuse versus new implementation**

| Component | Assessment |
|---|---|
| Safetensors loading and FP8 dequantization | Reuse existing infrastructure; add a bias-capable linear wrapper and legacy encoder key handling. |
| CPU PyTorch-compatible random generator | Reuse `torch_randn`; replace the shape-specific packing wrapper. |
| Flow schedule implementation | Reuse `FlowSchedule::simple` with `Shift::Snr { shift:1, multiplier:1, timesteps:1000 }`. |
| Euler update / ordinary CFG arithmetic | Reuse the arithmetic; adapt tensor geometry and conditioning. |
| RGBA tensor/byte conversion | Reuse qir’s four-channel helpers per image, subject to PNG quantization parity below. |
| Wan VAE internals | Reuse residual blocks, norms, attention, spatial resampling, last-tap convolutions and normalization. Parameterize image input/output channels. |
| Qwen2.5 text backbone | Add a variant: configuration, biased QKV, absent Q/K norms, final norm, tokenizer/template and checkpoint prefixes. |
| Layered DiT | **New model/block implementation**: dual streams, separate projections and FFNs, per-block modulation, additional timestep embedding, output head and positions. |
| Layered pipeline | New orchestration for reference encoding, multi-slice noise, patch packing, sampling, slice removal and ordered outputs. |
| Existing prefix cache | **Cannot be reused unchanged.** Layered text/reference states depend on generated tokens through joint attention. |
| Engine integration | Add layered native capability/routing and multi-image response support where required. `run_split` currently calls `backend(s)` directly. |

Evidence for reusable Wan internals and current RGB restrictions: [wan_vae.rs:193](C:/Users/5090/Documents/genai-native/crates/qir/src/model/wan_vae.rs:193), `:269`, `:322`, `:365`. Sampling primitives: `Q/crates/qir/src/comfy.rs:90`, `:194`; current CFG arithmetic: `Q/crates/qir/src/pipeline.rs:705`.

Two easy parity traps:

- `noise_packed` assumes 64 channels and no temporal dimension. Draw noise in ComfyUI’s `[B,16,L+1,h,w]` storage order **before** 2×2 patch packing. (`Q/crates/qir/src/comfy.rs:124`.)
- qir’s RGBA converter rounds to u8; ComfyUI `SaveImage` clips and casts, effectively truncating nonnegative values. Compare float decode outputs separately, then match the PNG conversion explicitly. qir’s helper also exports only the first batch item. (`Q/crates/qir/src/model/vae.rs:1154`; `C/nodes.py:1705`.)

**3. Port sequence and parity gates**

The estimates assume adapting existing code, implementing only the template’s text-only encoder path and single-frame VAE path, and adding no new CUDA kernels.

| Step | Work | Parity test against ComfyUI | Estimated production Rust |
|---|---|---|---:|
| 1 | Explicit component paths, checkpoint validation, FP8/bias wrapper and legacy scale mapping | Inventory shapes/dtypes; compare selected biased linear outputs, including full-precision FP8 dequantization | 150–250 |
| 2 | RGBA variant of `WanVae` | Same opaque-alpha input: compare posterior mean and normalized reference latent. Decode identical individual layer latents; compare RGB and alpha separately | 120–220 |
| 3 | Qwen2.5 text encoder variant and tokenizer wiring | Exact token IDs and prefix cut for positive/empty negative prompts; compare intermediate blocks and final 3584-wide conditioning | 350–650 |
| 4 | Layered dual-stream DiT | Compare patch tokens, position IDs, RoPE, timestep embedding, block 0 outputs, selected later blocks and final velocity on identical inputs | 650–1,100 |
| 5 | Multi-slice noise/packing and layered schedule | Compare the complete initial noise tensor, pack/unpack inversion, all 21 sigmas, timestep inputs and one Euler update | 150–250 |
| 6 | Complete layered pipeline and ordered RGBA outputs | Same input, prompt, seed and layer count: compare both CFG branches, each step’s latent, retained slices and decoded outputs | 230–430 |
| 7 | Native worker / engine routing | Compare `generate.splitLayers` dimensions, output count, order, alpha, insertion positions and resampling with the ComfyUI route | 150–300 |

**Estimated total: approximately 1,800–3,200 production lines, plus 600–1,000 Rust test/harness lines.** This excludes a full Qwen2.5 vision tower, general video VAE support, and substantial new offloading infrastructure.

For the parity harness, capture ComfyUI tensors at the listed boundaries using the same installed checkpoints. First compare float tensors with matching dtype/backend settings; report maximum error and relative RMS error. Set numerical acceptance limits from observed baseline variation rather than promising bitwise equality. Only after those gates pass should image-quality comparisons and attention optimizations become acceptance criteria.

Memory is a material implementation constraint: the installed DiT is **20,533,591,821 bytes**, the encoder **9,384,670,680 bytes**, and the VAE **253,816,616 bytes**. Their combined file footprint is about **28.1 GiB before activations**. **Inference:** a 24 GiB GPU will require staged residency/offloading; qir’s existing encoder eviction provides a useful starting point (`Q/crates/qir/src/pipeline.rs:388`).

**4. Can qir read `fp8mixed` as-is?**

**At the FP8 weight-reader level: yes, for this installed DiT. As a working qir model: no.**

The installed `qwen_image_layered_fp8mixed.safetensors` contains:

- **839 E4M3 weight tensors**.
- Corresponding F32 `weight_scale` tensors, all shape `[1]`.
- **839 U8 `comfy_quant` records**, all containing:
  ```json
  {"format":"float8_e4m3fn","full_precision_matrix_mult":true}
  ```
- Remaining BF16 tensors, including biases and selected full-precision weights.
- **No file-level `_quantization_metadata`**.

These are direct binary-header/payload observations.

qir already detects `weight_scale`, loads E4M3 bytes, reads per-layer `comfy_quant`, and honors `full_precision_matrix_mult`. Therefore **no DiT checkpoint conversion is needed**. Its existing dequantization path is appropriate; forcing FP8 activation GEMMs would disregard this checkpoint’s requested execution mode. See [fp8.rs:101](C:/Users/5090/Documents/genai-native/crates/qir/src/fp8.rs:101), `:149`, `:171`.

However:

1. **`Fp8Linear` has no bias.** The new layered linear wrapper must add each BF16 bias after the multiplication. ComfyUI explicitly supports biased mixed-precision linears. (`Q/crates/qir/src/fp8.rs:127`; `C/comfy/ops.py:1370`.)
2. **The existing Qwen2.1 DiT loader does not use this FP8 wrapper** and expects a different architecture and naming scheme. (`Q/crates/qir/src/model/dit.rs:117`, `:127`, `:1076`.)
3. **The installed text encoder is a separate compatibility problem:** it uses legacy `scale_weight` / `scale_input`, not `weight_scale` / `input_scale`. qir’s current reader will miss those scales. Implement an in-memory mapping equivalent to ComfyUI’s `convert_old_quants`, including its `scaled_fp8` marker semantics. (`C/comfy/utils.py:1441`; `Q/crates/qir/src/fp8.rs:102`.)
4. For other repacks, the filename `fp8mixed` alone is insufficient evidence. qir currently assumes scalar E4M3 scales and does not implement ComfyUI’s full range of quantization formats or activation-scale options. (`C/comfy/ops.py:1215`, `:1435`, `:1462`; `Q/crates/qir/src/fp8.rs:153`.)

The practical path is therefore: **reuse the FP8 reader, extend WanVae to RGBA, add a Qwen2.5 text variant, and implement the older dual-stream DiT plus a layered pipeline.**
