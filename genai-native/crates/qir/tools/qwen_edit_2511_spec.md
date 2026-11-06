# Qwen-Image-Edit-2511 natively: porting spec (Codex, read-only, 2026-10-28; checked by Claude)

2511 can reuse most of `LayeredDit` and the RGB Wan VAE. The main work is **Qwen2.5-VL image conditioning, split timestep modulation for reference tokens, and the template sampler**. Reusing Qwen-Image-2.1’s complete vision/DiT path would change the architecture.

No files were changed. Source links below reference the installed copies. Safetensors findings come from directly reading their JSON headers and small metadata tensors; binary files have byte offsets, not meaningful source line numbers.

**1. DiT compatibility**

The architectures share 60 dual-stream blocks, width 3072, 24 heads × 128, text width 3584, 2×2 latent patches, 16 latent channels, 4× GELU-tanh MLPs, non-affine LayerNorm, learned per-head Q/K RMSNorm, and RoPE axes `(16,56,56)` with theta 10000. The same ComfyUI class implements both models. [Comfy model.py:211](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:211), [model.py:347](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:347), [qir constants:37](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:37)

I compared **every header key, shape and dtype** in the installed diffusion files:

| Comparison | Result |
|---|---|
| 2511 / Layered tensor counts, excluding `__metadata__` | 4451 / 3612 |
| Shared names | 3611 |
| Shared names with identical shape and dtype | 2772 |
| Shared names differing in shape | All 839 `*.weight_scale`: 2511 scalar `[]`, Layered `[1]`; both F32 |
| Only in 2511 | 839 scalar F32 `*.input_scale`; empty F32 `__index_timestep_zero__` |
| Only in Layered | BF16 `time_text_embed.addition_t_embedding.weight`, `[2,3072]` |
| Other shape/dtype differences | None |

Evidence: [2511 checkpoint](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors), JSON header bytes `[8,595408)`; [Layered checkpoint](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/qwen_image_layered_fp8mixed.safetensors), header bytes `[8,415480)`. Matching inventories establish layout compatibility, not matching weight values.

All 839 2511 input scales contain `1.0`. Both files’ 839 `comfy_quant` records specify `float8_e4m3fn` and `full_precision_matrix_mult:true`. Thus the existing dequantize-to-BF16 multiplication path is appropriate; retain the FP8 storage. The current scale loader already reshapes either scalar representation to `()`. [qir linear:54](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:54)

Required differences:

- **Additional timestep condition:** make `LayeredInput.addition` optional. Layered adds embedding entry zero even when no condition is supplied; 2511 must add nothing. Its absence also means timestep computation cannot derive the compute dtype from that tensor. [qir input:267](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:267)
- **Model detection:** ComfyUI selects `index_timestep_zero` from the empty marker, and selects Layered’s additional embedding plus `negative_index` from the other key. [model_detection.py:991](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_detection.py:991)
- **Reference modulation:** with references, concatenate timestep batches `[t,0]`. In every block, target image tokens use the modulation and residual gates from `t`; reference image tokens use those from zero. Text uses only `t`. Both image modulation halves—attention and MLP—require this split. The installed implementation has no separate argument named `zero_cond_t`; this is its equivalent behavior. [model.py:255](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:255), [model.py:284](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:284), [model.py:497](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:497)
- **Packing:** independently patchify each `[B,16,1,H/8,W/8]` reference into `[B,N,64]`; append them after target tokens. Different reference sizes are allowed. References participate in joint attention and continue changing through the blocks. They are not cached frozen tokens.
- **Positions:** target frame index is zero; 2511 references use **`+1,+2,+3`**, versus Layered’s `−1,−2,…`. Each reference has its own centered spatial grid, without target-placement offsets. Text positions start at the larger target patch-grid half-size and repeat that scalar across all three axes. RoPE concatenation order is text, target, references. [model.py:408](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:408), [model.py:469](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:469), [model.py:504](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:504)
- **Output:** final normalization uses `t`; retain only target velocities before unpatchifying and cropping padding. The current `LayeredDit` leaves reference slicing to its caller. [model.py:565](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image/model.py:565), [qir forward:485](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:485)

`TextEncodeQwenImageEditPlus` exposes three image slots; the DiT itself iterates a reference list without a three-image architectural limit. Handle odd latent dimensions with ComfyUI’s patch padding; current Layered helpers assume even dimensions. [nodes_qwen.py:65](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_qwen.py:65), [qir patchify:154](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:154)

**2. Text and vision conditioning**

For every present image slot, `TextEncodeQwenImageEditPlus` creates two independently resized inputs:

1. **VL input:** preserve aspect ratio at approximately `384²` pixels, round each dimension, resize with `area`.
2. **VAE reference:** preserve aspect ratio at approximately `1024²` pixels, round dimensions to multiples of eight, resize with `area`, encode RGB.

Slot labels preserve their original numbers: `Picture 1: …`, `Picture 2: …`, `Picture 3: …`. Each contains `<|vision_start|><|image_pad|><|vision_end|>`, immediately followed by the next picture label or prompt. [nodes_qwen.py:75](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_qwen.py:75)

The exact chat envelope is:

```text
<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>
<|im_start|>user
{picture-labels-and-image-tokens}{prompt}<|im_end|>
<|im_start|>assistant
```

Use the literal template at [nodes_qwen.py:79](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_qwen.py:79). Token weighting is disabled; token `151655` becomes an image embedding request. ComfyUI replaces that placeholder with the complete merged vision sequence and marks those tokens attended. [qwen_image.py:39](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen_image.py:39), [sd1_clip.py:212](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/sd1_clip.py:212)

The tower’s actual raster dimensions are then rounded to multiples of **28**. Preprocessing uses bilinear interpolation with `align_corners=False`, CLIP mean/std, two repeated temporal frames, 14×14 patches and 2×2 spatial merging. A square 384 input therefore becomes **392×392**, yielding 784 patches and 196 merged image tokens. [qwen_vl.py:9](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen_vl.py:9)

The installed `qwen_2.5_vl_7b_fp8_scaled.safetensors` contains the **complete tower**, not merely a projector:

- `visual.patch_embed.proj.weight`: BF16 `[1280,3,2,14,14]`.
- `visual.blocks.0..31`: width 1280, 16 heads, MLP width 3420, QKV/projection, gated MLP and RMSNorm weights.
- `visual.merger.ln_q` and two merger linears: `5120→5120→3584`.
- FP8 vision matrices with `scale_weight`/`scale_input`, BF16 biases/norms.
- Counts: 704 `visual.blocks.*` entries, nine merger entries, one patch embedding.

Evidence: [encoder binary header](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors); architecture at [llama.py:1392](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/llama.py:1392), [qwen_vl.py:290](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen_vl.py:290). Window size is 112; blocks 7, 15, 23 and 31 use full attention. Restore original token order after window processing and merging. [qwen_vl.py:413](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen_vl.py:413)

Required encoder changes:

- Add a dedicated Qwen2.5 vision module. Qwen3’s existing tower requires learned position embeddings and deepstack mergers absent here; reuse preprocessing/attention primitives selectively. [qir vision.rs:343](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:343)
- Extend `Qwen25TextEncoder` to accept spliced embeddings, masks and three-axis position IDs; its current API accepts IDs and builds sequential one-axis RoPE. [qwen25_text.rs:162](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen25_text.rs:162)
- Match the installed **Qwen25-specific** image-position algorithm, including accumulated offsets and post-image text positions. Use M-RoPE sections `[16,24,24]`; do not substitute Qwen3’s interleaved scheme. [llama.py:415](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/llama.py:415), [llama.py:1405](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/llama.py:1405), [llama.py:522](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/llama.py:522)
- After the LM’s final norm, drop through the second `<|im_start|>user\n` prefix. Retain picture labels, image embeddings, instruction and assistant suffix. This is a token-derived cut, not a fixed “64 tokens” rule. Slice the attention mask identically. [qwen_image.py:61](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen_image.py:61)

Positive and negative branches receive the same images and reference latents.

**3. VAE and dimensions**

`qwen_image_vae.safetensors` is the RGB Wan 2.1 architecture already supported by `WanVae`: 16 latent channels, spatial compression eight, posterior-mean encoding and Wan21 normalization. ComfyUI also selects `latent_formats.Wan21`. [sd.py:863](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/sd.py:863), [supported_models.py:2058](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/supported_models.py:2058), [wan_vae.rs:334](C:/Users/5090/Documents/genai-native/crates/qir/src/model/wan_vae.rs:334)

Both installed VAE headers contain 194 BF16 tensors. Their only key-shape differences are:

| Key | RGB Qwen VAE | Layered VAE |
|---|---|---|
| `encoder.conv1.weight` | `[96,3,3,3,3]` | `[96,4,3,3,3]` |
| `decoder.head.2.weight` | `[3,96,3,3,3]` | `[4,96,3,3,3]` |
| `decoder.head.2.bias` | `[3]` | `[4]` |

Use the RGB file itself; matching remaining shapes does not establish identical weights. `WanVae::load` already infers three versus four channels. [wan_vae.rs:270](C:/Users/5090/Documents/genai-native/crates/qir/src/model/wan_vae.rs:270)

Preserve three distinct resizing operations:

- **Template `FluxKontextImageScale`:** choose the nearest aspect ratio from 17 fixed resolutions, then Lanczos resize with center crop; it is not a continuous megapixel resize. [nodes_flux.py:105](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_flux.py:105)
- **EditPlus internal resizing:** the separate 384² VL and 1024² VAE paths above.
- **Target VAEEncode:** uses the graph’s explicitly connected image. Most templates connect the original canvas; `angles` connects the Kontext-scaled image.

No listed graph uses `ImageScaleToTotalPixels`. If added later, reproduce its continuous `megapixels×1024²` area calculation, resolution-step rounding and disabled crop. [nodes_post_processing.py:249](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_post_processing.py:249)

ComfyUI center-crops target pixels to VAE compression multiples before encoding. Do not silently resize them instead. Also avoid normalizing references twice: ComfyUI normalizes in `QwenImage.extra_conds`; `qir::WanVae::encode` already returns normalized latents. [sd.py:1143](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/sd.py:1143), [model_base.py:2643](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_base.py:2643)

**4. Sampling contract per template**

All graphs set **AuraFlow shift 3.1, multiplier 1, CFGNorm strength 1**, and `index_timestep_zero` on both conditionings.

| Template suffix | Default steps / CFG | Model LoRAs, strength 1 | Conditioning / target |
|---|---:|---|---|
| `edit` | 40 / 4 | None | Up to three references; original image latent |
| `edit-lightning-8` | 8 / 1 | Lightning 8 | Same |
| `fill` | 40 / 4 | None | One reference; original latent + noise mask |
| `fill-lightning-4` | 4 / 1 | Lightning 4 | Same |
| `fill-lightning-8` | 8 / 1 | Lightning 8 | Same |
| `fill-guided` | 8 / 1 | Lightning 8 | Image plus mask **as a second reference**, and latent noise mask |
| `fill-ref` | 40 / 4 | None | Up to three references; original latent + noise mask |
| `fill-ref-lightning-8` | 8 / 1 | Lightning 8 | Same |
| `image-ref` | 8 / 1 | Lightning 8 | Up to three references; requested-size empty SD3 latent |
| `expand` | 40 / 4 | None | Cropped original reference; full padded canvas latent + mask |
| `expand-lightning-8` | 8 / 1 | Lightning 8 | Same |
| `remove-lightning-8` | 8 / 1 | Lightning 8, then Remover | One reference; original latent + mask |
| `angles` | 4 / 1 | Multiple Angles, then Lightning 4 | Kontext-scaled image supplies reference and target |

Graph evidence: [edit:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-edit.json:35), [edit-lightning-8:36](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-edit-lightning-8.json:36), [fill:34](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill.json:34), [fill-lightning-4:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill-lightning-4.json:35), [fill-lightning-8:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill-lightning-8.json:35), [fill-guided:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill-guided.json:35), [fill-ref:36](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill-ref.json:36), [fill-ref-lightning-8:37](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-fill-ref-lightning-8.json:37), [image-ref:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-image-ref.json:35), [expand:35](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-expand.json:35), [expand-lightning-8:36](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-expand-lightning-8.json:36), [remove:37](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-remove-lightning-8.json:37), [angles:34](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-angles.json:34). Defaults are in each file’s `meta.defaults`, line 13.

Most graphs hard-code **Euler/simple, denoise 1**. `fill`, `fill-lightning-{4,8}` expose sampler/scheduler/denoise placeholders; `remove-lightning-8` exposes sampler/scheduler while fixing denoise 1. Preserve overrides or explicitly reject unsupported combinations; do not advertise arbitrary KSampler parity.

Implementation details:

- Use `FlowSchedule::simple(steps, Shift::Snr { shift:3.1, multiplier:1.0, timesteps:1000 })`. The existing Layered schedule uses shift 1. [comfy.rs:181](C:/Users/5090/Documents/genai-native/crates/qir/src/comfy.rs:181), [layered_pipeline.rs:30](C:/Users/5090/Documents/genai-native/crates/qir/src/layered_pipeline.rs:30)
- Pass sigma as the DiT timestep; its sinusoidal projection supplies the internal ×1000. AuraFlow changes the external multiplier to one. [nodes_model_advanced.py:163](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_model_advanced.py:163)
- Convert velocity to denoised prediction: `p=x−σv`. Combine `p_cfg=p_neg+CFG·(p_pos−p_neg)`. CFGNorm then multiplies by `clamp(||p_pos||channel/(||p_cfg||channel+1e−8),0,1)`, separately at each spatial position. It operates in **denoised space**, after CFG. [model_sampling.py:86](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_sampling.py:86), [nodes_cfg.py:99](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_cfg.py:99)
- Initialize with `σ·noise+(1−σ)·target`; use existing Torch-compatible noise generation. Lightning’s steps/CFG come from template settings, not automatic behavior inside the LoRA.
- Fill uses `SetLatentNoiseMask`, **not** `InpaintModelConditioning` or masked-image VAE encoding. Read the PNG’s red channel; interpolate the mask to latent dimensions and repeat channels. Before each model call, replace protected input with the appropriately noised original latent; after prediction, replace protected denoised output with the original latent. Preserve fractional masks. [nodes.py:1575](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/nodes.py:1575), [utils.py:1350](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/utils.py:1350), [samplers.py:630](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/samplers.py:630)
- Expand’s graph performs no padding. PhotoshopEX supplies a canvas prefilled by nearest-edge pixel replication, plus the mask and original-picture rectangle. Crop that rectangle only for reference conditioning. [generate_cmds.rs:3454](C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3454), [reference rectangle:2798](C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2798)

**5. LoRA compatibility**

Header inventories:

| File | Layout | Coverage |
|---|---|---|
| Lightning 4 and 8 | 2160 BF16 tensors each; 720 modules; rank 64; **all alpha values 8** | Every block: eight attention projections plus image/text MLP up and down |
| Remover | 1440 BF16 tensors; 720 modules; rank 16; no alpha tensors | Every block: eight attention projections, image/text MLP down, image/text modulation |

Exact attention stems are `attn.{add_k_proj,add_q_proj,add_v_proj,to_add_out,to_k,to_out.0,to_q,to_v}`. Lightning MLP stems are `{img,txt}_mlp.net.{0.proj,2}`. Remover adds `{img,txt}_mod.1` and uses only MLP `net.2`.

Files: [Lightning 4](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors), [Lightning 8](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors), [Remover](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models/loras/Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors).

Mapping is direct:

```text
{stem}.lora_down.weight       → A
{stem}.lora_up.weight         → B
{stem}.lora_A.default.weight  → A  # add this suffix
{stem}.lora_B.default.weight  → B  # add this suffix
{stem}                       → DiT {stem}.weight
```

`lora.rs` already parses Lightning’s layout and alpha correctly: its effective multiplier at strength one is **8/64 = 0.125**. Remover currently produces no recognized pairs; add `.default` suffix support, with absent alpha giving multiplier one. ComfyUI explicitly supports that spelling. [qir lora.rs:149](C:/Users/5090/Documents/genai-native/crates/qir/src/lora.rs:149), [lora.rs:199](C:/Users/5090/Documents/genai-native/crates/qir/src/lora.rs:199), [Comfy adapter:193](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/weight_adapter/lora.py:193)

**Parsing alone is insufficient:** `qwen_image_dit.rs`’s private `LayeredLinear` has no LoRA slot/application hook. Attach stacks to every addressed projection and modulation linear; retain module names unchanged. [qir linear:43](C:/Users/5090/Documents/genai-native/crates/qir/src/model/qwen_image_dit.rs:43), [Comfy direct mapping:326](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/lora.py:326)

For strict numerical parity, capture ComfyUI’s actual patched-weight route. Its patcher can dequantize, merge and **requantize with recalculated scales and seeded stochastic rounding**. Runtime `base(x)+B(A(x))` is algebraically equivalent before rounding, but need not match that route numerically. [model_patcher.py:912](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_patcher.py:912), [ops.py:1499](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ops.py:1499)

The angles template additionally requires `qwen-image-edit-2511-multiple-angles-lora.safetensors`; that file is **not installed**, so its header compatibility remains unverified. The template’s prompt vocabulary is `<sks> [azimuth] [elevation] [distance]`. [angles.json:23](C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/qwen-edit-2511-angles.json:23)

**6. Porting sequence and parity gates**

The following are proposed acceptance limits, not measured 2511 results. Define relative RMS as `RMS(actual−oracle)/max(RMS(oracle),1e−12)`. Compare complete tensors, matching dtype/backend and identical boundary inputs.

| Order | Work and estimated changed/new lines | ComfyUI dumps and gate |
|---|---|---|
| 1 | Add `edit2511_files.rs`; validate original filenames, header variants, quant records and RGB VAE. **120–180** | Exact key/shape/dtype inventory; input scales exactly one. Compare representative dequantized weights exactly after matched BF16 casting. |
| 2 | Generalize `qwen_image_dit.rs`: optional addition, positive reference indices, zero-timestep split, odd-grid padding, target slicing. **180–280** | Dump packed tokens, IDs, RoPE, `temb(t/0)`, modulation/gates, independent blocks 0/29/59 and full velocity. IDs/packing exact; CPU independent blocks max abs ≤0.005 and relative RMS ≤1e−4; full velocity ≤0.002 and ≤5e−4. These extend existing Layered gates. |
| 3 | Reuse RGB VAE; add EditPlus area resize and Kontext crop/resize preprocessing in a new module. **180–300** | Dump both resized images, VAE posterior mean, normalized references and target latent. Dimensions/crops exact; float resize max abs ≤1e−6; isolated CPU VAE relative RMS ≤1e−4. Include portrait, landscape and non-multiple-of-eight target inputs. |
| 4 | Add `model/qwen25_vision.rs`, using existing FP8 reader and attention primitives. **400–600** | Dump normalized patches/grid, window permutation and lengths, patch projection, blocks 0/7/31, merger and restored sequence. Integer data exact; isolated CPU stages relative RMS ≤1e−4; whole tower ≤5e−4. |
| 5 | Extend `qwen25_text.rs` for embeddings, masks, exact edit template and M-RoPE. **220–350** | Dump token IDs, image spans, spliced embeddings, three-axis positions, LM taps and final sliced conditioning/mask. IDs/spans/positions exact; CPU conditioning max abs ≤0.02 and relative RMS ≤5e−5, matching the existing text gate. |
| 6 | Add Remover suffixes in `lora.rs`; wire DiT slots. Establish effective-weight parity for quantized patches. **120–220**, plus **250–500** if requantization support is required | Exactly 720 matched modules per installed adapter; alpha/rank exact. Dump effective weights and identical-input linear outputs for attention, MLP and modulation. FP32 delta relative RMS ≤1e−5; require exact quantized bytes/scales if reproducing the requantized route. |
| 7 | Add `edit2511_pipeline.rs`: staged residency, Euler/simple, CFGNorm, masked sampling, reference branches and empty target. **350–550** | Dump noise, sigmas, initial latent, branch velocities, denoised predictions, CFGNorm scales, protected-region replacements and every Euler latent. Sigma max abs ≤1e−7; noise ≤2e−6; isolated FP32 sampler primitives ≤2e−6. Use oracle velocities first to separate sampler errors from model errors. |
| 8 | Extend worker family/template routing, default LoRAs, image-slot pruning, expand rectangle and availability checks. **180–300** | For every table row, compare resolved arguments and preprocessed tensors exactly. Run end-to-end 4/8/40-step fixtures, three unequal references, soft masks, expand and stacked Remover. Do not expose angles until its adapter is available and validated. |

Existing gate evidence: [DiT test:58](C:/Users/5090/Documents/genai-native/crates/qir/tests/qwen_image_dit.rs:58), [text test:169](C:/Users/5090/Documents/genai-native/crates/qir/tests/qwen25_text.rs:169). Budget another **350–600 lines** for fixture capture and tests.

For CUDA BF16, start with relative RMS limits of **1% per isolated block and 2% for complete encoder/DiT outputs**, plus finite-value and maximum-error inspection. Require final normalized-latent relative RMS ≤3% and decoded float-RGB PSNR ≥35 dB against the fixed ComfyUI route. Calibrate these proposed limits from repeated oracle/backend comparisons before accepting them; CPU success alone is insufficient.

Parallel ownership can avoid shared edits:

- Worker A: `qwen_image_dit.rs` and its tests, steps 2 and DiT LoRA hooks.
- Worker B: new `qwen25_vision.rs` and vision fixtures.
- Worker C: `qwen25_text.rs` and text fixtures.
- Worker D: new file-resolution/preprocessing/pipeline modules, then worker routing.
- Assign `lora.rs` to A; one integrator owns `model/mod.rs`, `lib.rs` and export changes. Agree on vision output, embedding-span and DiT reference-boundary interfaces first.

**Memory constraint:** installed DiT storage is 20.534 GB and the complete encoder file 9.385 GB—about **27.86 GiB combined**, before activations, dequantization scratch, LoRAs, VAE and allocator slack. Stage VAE reference encoding, vision, LM conditioning, DiT sampling and VAE decoding; release encoder allocations before loading the DiT. Existing Layered code already follows this lifecycle and trims retained CUDA memory. [layered_pipeline.rs:153](C:/Users/5090/Documents/genai-native/crates/qir/src/layered_pipeline.rs:153)

At 1024², each reference contributes 4096 DiT tokens; target plus three references reaches 16,384 image tokens before text. Require fused/chunked attention and retain streamed-block fallback. Avoid materializing the whole DiT as BF16 or keeping a second merged-weight copy; either would defeat the 32 GB residency budget.
