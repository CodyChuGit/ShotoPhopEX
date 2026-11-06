# R4 spec: Qwen-Image-2.1 from ComfyUI's single files

> Codex (read-only, 2026-10-27), from Claude's brief; checked by Claude against the headers' counts before filing. Headers only: structural compatibility, not numerical equality.

Qwen-Image-2.1 can load directly from the three ComfyUI files without a converted copy. The BF16 port requires a fused-MLP adapter, a text-encoder prefix change, and a VAE name/shape adapter. The INT8 port requires additional work: signed-byte loading, per-row scales, ConvRot basis compatibility, and rotated embedding lookup.

All research was read-only. For safetensors, I read exactly the 8-byte length followed by the JSON header; no tensor payload was read. Consequently, the findings establish structural compatibility, not numerical equality.

Paths below use these roots:

- **N** = `C:\Users\5090\Documents\genai-native`
- **C** = `C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI`
- **D** = `C:\Users\5090\Documents\_eval\native-models\qwen-image-2.1`
- **K** = `C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\comfy_kitchen`

**1. Header inventory and exact weight mapping**

Counts exclude `__metadata__`. Header lengths exclude the initial 8-byte length field.

| File relative to root | Header bytes | Tensor count | Stored dtypes |
|---|---:|---:|---|
| C `models/diffusion_models/qwen_image_2.1_bf16.safetensors` | 31,136 | 265 | 265 BF16 |
| C `models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 74,448 | 649 | 192 I8, 192 F32, 192 U8, 73 BF16 |
| C `models/text_encoders/qwen3vl_8b_bf16.safetensors` | 87,216 | 750 | 750 BF16 |
| C `models/text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 140,992 | 1,258 | 254 I8, 254 F32, 254 U8, 496 BF16 |
| C `models/vae/qwen_image_2.1_vae_bf16.safetensors` | 28,872 | 238 | 238 BF16 |
| D `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 24,784 | 211 | BF16 |
| D `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 10,136 | 86 | BF16 |
| D `text_encoder/model-00001-of-00004.safetensors` | 49,792 | 426 | BF16 |
| D `text_encoder/model-00002-of-00004.safetensors` | 18,520 | 142 | BF16 |
| D `text_encoder/model-00003-of-00004.safetensors` | 18,552 | 142 | BF16 |
| D `text_encoder/model-00004-of-00004.safetensors` | 5,200 | 40 | BF16 |
| D `vae/diffusion_pytorch_model.safetensors` | 27,888 | 238 | F32 |

Both snapshot indexes exactly match the union of their shard-header keys:

| Index | Keys | Metadata |
|---|---:|---|
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 297 | `{"total_size":14230249472}` |
| `text_encoder/model.safetensors.index.json` | 750 | `{"total_parameters":8767123696,"total_size":17534247392}` |

Actual ComfyUI header metadata:

```json
// Both DiTs and both text encoders:
{"format":"pt"}

// VAE:
{"modelspec.architecture":"qwen_image_2.1_vae","format":"pt"}
```

There is **no `_quantization_metadata` entry in either inspected INT8 header**. The snapshot transformer/text shards have `{"format":"pt"}`; the snapshot VAE has no metadata entry. This corrects the broader claim in `N/docs/MODELS-ON-DISK.md:10` that quantization metadata is always present.

**DiT: 297 snapshot tensors → 265 ComfyUI tensors**

For every `i = 0..31`:

```text
ComfyUI:
transformer_blocks.{i}.img_mlp.gate_up.weight [24576,4096]

equals structurally:
cat(
  snapshot transformer_blocks.{i}.img_mlp.gate_layer.weight [12288,4096],
  snapshot transformer_blocks.{i}.img_mlp.proj.weight       [12288,4096],
  dim=0
)
```

The order is **gate first, up/proj second**. ComfyUI explicitly documents and implements this in [C/comfy/ldm/qwen_image21/model.py:51](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/qwen_image21/model.py:51).

Therefore, to preserve qir’s existing split interface:

```text
gate_layer.weight = gate_up.weight.narrow(0, 0,     12288)
proj.weight       = gate_up.weight.narrow(0, 12288, 12288)
```

All other **233 keys** are identical in name, shape and dtype:

```text
# Nine global tensors:
img_in.weight
modulation.1.weight
norm_out.linear.weight
proj_out.weight
time_text_embed.timestep_embedder.linear_1.weight
time_text_embed.timestep_embedder.linear_2.weight
txt_in.in_layer.weight
txt_in.out_layer.weight
txt_in.text_norm.weight

# Seven tensors per block, i=0..31:
transformer_blocks.{i}.attn.norm_k.weight
transformer_blocks.{i}.attn.norm_q.weight
transformer_blocks.{i}.attn.to_k.weight
transformer_blocks.{i}.attn.to_out.0.weight
transformer_blocks.{i}.attn.to_q.weight
transformer_blocks.{i}.attn.to_v.weight
transformer_blocks.{i}.img_mlp.out.weight
```

There is no DiT QKV fusion, no prefix stripping, and no matrix transpose. Both layouts store linear weights as `[out,in]`. There are no biases.

The only unmatched names before adaptation are:

- Snapshot only: 32 `img_mlp.gate_layer.weight` plus 32 `img_mlp.proj.weight`.
- ComfyUI only: 32 `img_mlp.gate_up.weight`.

After splitting, there are no unexplained tensors.

qir currently requires the split names: [N/crates/qir/src/model/dit.rs:17](/C:/Users/5090/Documents/genai-native/crates/qir/src/model/dit.rs:17), with MLP loading at `dit.rs:815`. Its LoRA loader already splits fused `gate_up` LoRAs in the same order: `N/crates/qir/src/lora.rs:208`.

**Text encoder: 750 → 750 tensors**

Apply exactly this prefix substitution to **398 tensors**:

```text
snapshot model.language_model. → ComfyUI model.
```

The affected names are:

```text
model.language_model.embed_tokens.weight
model.language_model.norm.weight

# i=0..35:
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.mlp.down_proj.weight
model.language_model.layers.{i}.mlp.gate_proj.weight
model.language_model.layers.{i}.mlp.up_proj.weight
model.language_model.layers.{i}.self_attn.k_norm.weight
model.language_model.layers.{i}.self_attn.k_proj.weight
model.language_model.layers.{i}.self_attn.o_proj.weight
model.language_model.layers.{i}.self_attn.q_norm.weight
model.language_model.layers.{i}.self_attn.q_proj.weight
model.language_model.layers.{i}.self_attn.v_proj.weight
```

The remaining **352 tensors** are unchanged:

- `lm_head.weight`: one tensor, `[151936,4096]`.
- `model.visual.*`: 351 tensors.

All mapped shapes and dtypes match. There are no unmatched tensors after substitution and no transposes.

Language-model Q/K/V remain separate. Vision QKV is already fused on **both** sides:

```text
model.visual.blocks.{i}.attn.qkv.weight [3456,1152]
model.visual.blocks.{i}.attn.qkv.bias   [3456]
# i=0..26
```

The existing loader already accepts an alternate language-model root through `QwenTextEncoder::load_at`; use `vb.pp("model")` for ComfyUI instead of `model.language_model`. See [N/crates/qir/src/model/text_encoder.rs:349](/C:/Users/5090/Documents/genai-native/crates/qir/src/model/text_encoder.rs:349). Also change the lazy chat norm lookup at `N/crates/qir/src/pipeline.rs:478`; vision and `lm_head` names need no prefix change.

**VAE: 238 → 238 tensors**

All 238 names change. The following rules form a complete bijection; they are scoped substitutions, not unrestricted string replacements.

Let `s ∈ {encoder,decoder}`, `p ∈ {weight,bias}`, and apply the residual-member substitutions only inside residual blocks.

| Snapshot name/prefix | ComfyUI name/prefix |
|---|---|
| `quant_conv.{p}` | `conv1.{p}` |
| `post_quant_conv.{p}` | `conv2.{p}` |
| `{s}.conv_in.{p}` | `{s}.conv1.{p}` |
| `{s}.conv_out.{p}` | `{s}.head.2.{p}` |
| `{s}.norm_out.gamma` | `{s}.head.0.gamma` |
| `{s}.mid_block.resnets.0.` | `{s}.middle.0.` |
| `{s}.mid_block.resnets.1.` | `{s}.middle.2.` |
| `{s}.mid_block.attentions.0.` | `{s}.middle.1.` |
| `encoder.down_blocks.{i}.resnets.{j}.` | `encoder.downsamples.{i}.downsamples.{j}.` |
| `encoder.down_blocks.{i}.downsampler.` | `encoder.downsamples.{i}.downsamples.2.` |
| `decoder.up_blocks.{i}.resnets.{j}.` | `decoder.upsamples.{i}.upsamples.{j}.` |
| `decoder.up_blocks.{i}.upsampler.` | `decoder.upsamples.{i}.upsamples.3.` |

Stage ranges:

```text
encoder residuals:  i=0..4, j=0..1
encoder downsample: i=0..3

decoder residuals:  i=0..4, j=0..2
decoder upsample:   i=0..3

time_conv exists:
encoder downsample i=1..3
decoder upsample   i=0..2
```

Residual-member mapping:

| Snapshot suffix | ComfyUI suffix |
|---|---|
| `norm1.gamma` | `residual.0.gamma` |
| `conv1.{p}` | `residual.2.{p}` |
| `norm2.gamma` | `residual.3.gamma` |
| `conv2.{p}` | `residual.6.{p}` |
| `conv_shortcut.{p}` | `shortcut.{p}` |

The shortcut tensors occur only at:

```text
encoder.down_blocks.{1,2,3}.resnets.0
decoder.up_blocks.{2,3,4}.resnets.0
```

Attention and resampler suffixes remain unchanged:

```text
norm.gamma
proj.weight
proj.bias
to_qkv.weight
to_qkv.bias

resample.1.weight
resample.1.bias
time_conv.weight
time_conv.bias
```

Accounting:

| Group | Tensors |
|---|---:|
| Encoder stages | 80 |
| Decoder stages | 110 |
| Encoder middle | 17 |
| Decoder middle | 17 |
| Encoder input/output/norm | 5 |
| Decoder input/output/norm | 5 |
| Quant/post-quant convolutions | 4 |
| **Total** | **238** |

Shape/dtype adaptation:

- **76 weights:** ComfyUI `[out,in,1,h,w]` → snapshot/qir `[out,in,h,w]`, by `squeeze(2)`.
- **162 tensors:** mapped shapes already match.
- All ComfyUI values are BF16; all snapshot VAE values are F32.
- No transpose is required.
- Eight spatial `resample.1.weight` tensors remain 4D.
- Four middle-attention projection/QKV weights remain 4D.
- Middle attention QKV stays fused on both sides.
- No missing/extra learned tensors remain after mapping.

Examples:

```text
decoder.conv1.weight:
  ComfyUI [1152,64,1,3,3]
  qir     [1152,64,3,3]

conv1.weight:
  ComfyUI [128,128,1,1,1]
  qir     [128,128,1,1]

decoder.middle.1.to_qkv.weight:
  both    [3456,1152,1,1]
```

qir currently constructs `Conv2d` directly, so name remapping alone will fail on the singleton temporal dimensions: [N/crates/qir/src/model/vae.rs:164](/C:/Users/5090/Documents/genai-native/crates/qir/src/model/vae.rs:164).

**INT8 ConvRot format and qir compatibility**

Each quantized module has this structural triple:

```text
<module>.weight       I8  [out,in]
<module>.weight_scale F32 [out,1]
<module>.comfy_quant   U8  [72]
```

There are no stored rotation matrices, zero points, `input_scale`, or `weight_scale_2` tensors in these two files.

| Component | Quantized modules | BF16 tensors retained |
|---|---:|---|
| DiT | 192 = 32 × 6 | 73 |
| Text encoder | 254 = 36 × 7 + embedding + head | 496 |

DiT quantized modules, for `i=0..31`:

```text
transformer_blocks.{i}.attn.to_q
transformer_blocks.{i}.attn.to_k
transformer_blocks.{i}.attn.to_v
transformer_blocks.{i}.attn.to_out.0
transformer_blocks.{i}.img_mlp.gate_up
transformer_blocks.{i}.img_mlp.out
```

All nine global DiT tensors and 64 Q/K norm tensors remain BF16.

Text quantized modules, for `i=0..35`:

```text
model.layers.{i}.self_attn.q_proj
model.layers.{i}.self_attn.k_proj
model.layers.{i}.self_attn.v_proj
model.layers.{i}.self_attn.o_proj
model.layers.{i}.mlp.gate_proj
model.layers.{i}.mlp.up_proj
model.layers.{i}.mlp.down_proj

model.embed_tokens
lm_head
```

The 351 vision tensors and 145 language-model norm tensors remain BF16.

ComfyUI’s loader decodes `.comfy_quant` as JSON, reads `.weight_scale`, and recognizes `format`, `convrot`, and `convrot_groupsize`: [C/comfy/ops.py:1195](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ops.py:1195), especially `ops.py:1229`. `int8_tensorwise` selects `TensorWiseINT8Layout` at `C/comfy/quant_ops.py:238`.

The repository’s existing inventory records this descriptor:

```json
{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

Source: `N/docs/MODELS-ON-DISK.md:17`. **I did not independently decode those 72-byte tensor payloads**, because this task restricted safetensors reads to headers. The header proves their presence, size, and dtype; production loading must decode and validate them rather than infer the descriptor from filenames.

“ConvRot” means an orthogonal, group-wise regular Hadamard rotation applied before quantization:

```text
Wrot = W Hᵀ
Xrot = X H
Y ≈ dequantized_INT8_matmul(Xrot, Wrotᵀ)
```

Weights use per-output-row scales; activations are dynamically quantized per row. Despite the format name `int8_tensorwise`, these files’ weight scales are per row. ComfyUI delegates the implementation to comfy-kitchen:

- `K/tensor/int8.py:98`: quantization contract.
- `K/tensor/int8.py:160`: dequantization.
- `K/tensor/int8_utils.py:11`: Hadamard construction.
- `K/backends/eager/quantization.py:899`: rotate then row-wise quantize.

**qir cannot consume these files unchanged today.** Its prequantized format expects:

```text
<module>.weight_i8 U8 [out,in]  # signed bytes carried as U8
<module>.col_scale F32 [out]
```

See [N/crates/qir/src/model/dit.rs:150](/C:/Users/5090/Documents/genai-native/crates/qir/src/model/dit.rs:150) and `N/crates/qir/src/convrot.rs:678`. Candle’s lack of an I8 tensor dtype is explicitly documented at `convrot.rs:1`; preserve the signed byte patterns when importing I8 as U8, rather than numerically converting negative values.

There is also a **rotation-basis mismatch**:

```text
ComfyUI/comfy-kitchen H4:
  [ 1, 1, 1,-1]
  [ 1, 1,-1, 1]
  [ 1,-1, 1, 1]
  [-1, 1, 1, 1]

qir H4:
  [-1, 1, 1, 1]
  [ 1,-1, 1, 1]
  [ 1, 1,-1, 1]
  [ 1, 1, 1,-1]
```

ComfyUI source: `K/tensor/int8_utils.py:24`. qir source: [N/crates/qir/src/model/rotation.rs:24](/C:/Users/5090/Documents/genai-native/crates/qir/src/model/rotation.rs:24), and its compiled activation kernel at `N/crates/qir/kernels/convrot/quant_ops.cu:76`; `N/crates/qir/build.rs:51` builds that kernel.

These bases differ by reversal of the rotated coordinates within each 256-wide group. A mathematically valid adapter can reverse each group’s stored INT8 columns to use qir’s existing basis, leaving row scales unchanged. Alternatively, add an explicit ComfyUI basis to the activation kernels. This relation needs numerical tests before using real weights; sharing the name “regular Hadamard” does not establish compatibility.

`convrot_int8_gemm.cu:113` also contains a ComfyUI-compatible matrix, but the current `ConvRotLinear` activation path uses `quant_ops.cu`. Do not assume that source file changes the active basis.

Finally, the INT8 embedding requires **gather, scale, then inverse rotation**. ComfyUI handles this explicitly at [C/comfy/ops.py:1702](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ops.py:1702) and `ops.py:1743`; see also `K/tensor/int8.py:179`. Returning scaled rotated rows as ordinary embeddings would be incorrect.

`gn-weights` already provides a useful raw reader: `N/crates/gn-weights/src/safetensors.rs:58` reads bounded headers and `:107` reads tensor ranges; its dtype model supports I8 at `N/crates/gn-weights/src/dtype.rs:27`. That capability is not yet wired into qir’s model loaders.

**2. What the worker reads from the snapshot today**

The worker resolves:

```text
models_root / "qwen-image-2.1"
```

at [N/crates/gn-diffusion/src/lib.rs:291](/C:/Users/5090/Documents/genai-native/crates/gn-diffusion/src/lib.rs:291). Constants are at `lib.rs:94–100`. Availability still requires snapshot structure through `qir::pipeline::is_snapshot`: `N/crates/qir/src/pipeline.rs:206`.

Load sequence:

| Dependency | Code |
|---|---|
| `processor/tokenizer.json` | `pipeline.rs:324` |
| Optional root `text_encoder.q8_0.gguf` selection | `gn-diffusion/src/lib.rs:343` |
| GGUF text loader | `pipeline.rs:731–738`; `model/text_encoder.rs:371` |
| Transformer files/cache | `pipeline.rs:335`; `convrot_cache.rs:352` |
| `vae/config.json` | `pipeline.rs:342` |
| VAE safetensors | `pipeline.rs:345` |
| Text safetensors for vision | `pipeline.rs:353` |
| Text safetensors for lazy chat head/final norm | `pipeline.rs:475` |

[Pipeline load source](/C:/Users/5090/Documents/genai-native/crates/qir/src/pipeline.rs:318).

Across supported worker operations, the actual snapshot file dependencies are:

```text
processor/tokenizer.json

text_encoder.q8_0.gguf             # when present; selected automatically

text_encoder/model-00001-of-00004.safetensors
text_encoder/model-00002-of-00004.safetensors
text_encoder/model-00003-of-00004.safetensors
text_encoder/model-00004-of-00004.safetensors

transformer/diffusion_pytorch_model-00001-of-00002.safetensors
transformer/diffusion_pytorch_model-00002-of-00002.safetensors

vae/config.json
vae/diffusion_pytorch_model.safetensors
```

Important qualifications:

- With GGUF enabled, the language decoder loads from GGUF, but **the default vision-enabled pipeline still opens the text safetensors**. `Options::default()` sets `vision: true` at `pipeline.rs:66`.
- Chat also loads `lm_head.weight` and the final norm from those safetensors.
- A valid external `.qir-cache` can supply transformer tensor data instead of the original transformer shards. Cache resolution still examines the source set. See `N/crates/qir/src/convrot_cache.rs:357`.
- The loader enumerates component files by extension; **it does not read either index JSON**: [N/crates/qir/src/loader/mod.rs:45](/C:/Users/5090/Documents/genai-native/crates/qir/src/loader/mod.rs:45).
- The worker does **not** read `transformer/config.json`, `text_encoder/config.json`, `scheduler/scheduler_config.json`, `model_index.json`, generation config, processor configs, chat template, separate vocab/merges, added-token JSON, or tokenizer config.
- Of snapshot config values, the worker currently reads only `vae/config.json`’s `latents_mean` and `latents_std`. Architecture and scheduler values are already Rust constants.

Standalone qir CLI paths repeat these dependencies: tokenizer reads at `main.rs:780,978`; VAE config reads at `main.rs:885,985,1228,1495`. CLI help mentioning transformer/text `config.json` does not mean those files are parsed.

**3. Tokenizer source and recommendation**

Krea 2 still reads a standalone tokenizer from:

```text
models_root/krea2-turbo/tokenizer/tokenizer.json
```

The directory is resolved at `N/crates/gn-diffusion/src/lib.rs:295`; loading is at [N/crates/qir/src/krea2_pipeline.rs:76](/C:/Users/5090/Documents/genai-native/crates/qir/src/krea2_pipeline.rs:76), and availability requires that file at `krea2_pipeline.rs:51`.

On this machine, Krea’s tokenizer and Qwen-Image-2.1’s tokenizer are **byte-identical**:

```text
Size:   11,422,654 bytes each
SHA256: aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

Thus the Krea weight path already reads ComfyUI’s fp8 files, but its tokenizer dependency still lives outside the ComfyUI models layout.

ComfyUI ships compatible Qwen tokenizer assets inside its code tree:

```text
C/comfy/text_encoders/qwen25_tokenizer/
  vocab.json              2,776,833 bytes
  merges.txt              1,823,241 bytes
  tokenizer_config.json       8,259 bytes
                         -------------
  Total                   4,608,333 bytes
```

There is no `tokenizer.json` or separate licence file in that directory. `merges.txt` begins `#version: 0.2`; the JSON files have no licence/copyright header. This describes the local files, not a redistribution licence conclusion.

ComfyUI intentionally uses this directory for Qwen3-VL: [C/comfy/text_encoders/qwen3vl.py:147](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/qwen3vl.py:147). It instantiates Transformers’ `Qwen2Tokenizer`, with no automatic start/end tokens and pad ID `151643`.

I compared the assets:

- `vocab.json` equals the snapshot tokenizer’s complete BPE vocabulary.
- All **151,387 merges** match in order, excluding the merges file’s version header.
- Full tokenizer behavior still includes normalization, pretokenization, added tokens and decoding; vocabulary/merge equality alone is insufficient.

**Recommendation:** embed the existing, exact shared Krea/2.1 `tokenizer.json` at build time and load it with `Tokenizer::from_bytes`. This removes both snapshot dependencies and avoids requiring a ComfyUI installation.

Size cost:

- Raw embedded bytes: **11,422,654 bytes**, approximately **10.89 MiB**, shared once between both models.
- In-memory gzip compression measured **2,062,020 bytes**, approximately **1.97 MiB**, if build/package size warrants decompression support.
- Reading ComfyUI’s three assets would avoid executable growth but introduce a code-tree dependency and require constructing the full tokenizer behavior.
- Requiring a user-folder tokenizer would add an **11.42 MB** asset and another installation requirement. It can remain an explicit override, but should not be required for normal R4 operation.

Pin the exact tokenizer hash and preserve its upstream attribution/licence notices. Keep prompt templates in Rust: ComfyUI’s 2.1 template and pre-final-norm behavior are at `C/comfy/text_encoders/qwen_image21.py:8,33,52`.

**4. Config values absent from single-file metadata**

The single files encode tensor dimensions, but not all architectural, preprocessing, or sampling choices. Their header metadata does not contain the snapshot configs.

| Component | Snapshot values | ComfyUI source / port action |
|---|---|---|
| DiT dimensions | layers `32`; heads `32`; head dim `128`; input/output `64`; context `4096`; MLP ratio `3` | Inferred from tensors/block count at `C/comfy/model_detection.py:970–988` |
| DiT non-dimensional choices | RoPE axes `[16,56,56]`; epsilon `1e-6`; patch size `1`; causal conditioning `true` | Constructor defaults at `C/comfy/ldm/qwen_image21/model.py:234`; causal-prefix behavior at `:273` and attention/block implementation |
| Text architecture | width `4096`; intermediate `12288`; layers `36`; heads `32`; KV heads `8`; head dim `128`; vocabulary `151936`; RMS epsilon `1e-6`; SiLU; no attention bias | `C/comfy/text_encoders/llama.py:323–347` |
| Text position encoding | maximum positions `262144`; theta `5000000`; interleaved MRoPE sections `[24,20,20]`; rope type `default` | `C/comfy/text_encoders/llama.py:350–357` |
| Text output head | untied embeddings, `tie_word_embeddings:false` | Separate `lm_head.weight`; `llama.py:343`; contrast 4B’s tied behavior at `:363` |
| Vision | width `1152`; intermediate `4304`; depth `27`; DeepStack `[8,16,24]`; heads `16`; input channels `3`; patch `16`; temporal patch `2`; spatial merge `2`; positions `2304`; output width `4096` | `C/comfy/text_encoders/qwen3vl.py:14–20,58` |
| Vision activation | `gelu_pytorch_tanh` | `C/comfy/text_encoders/qwen35.py:475–483` |
| VAE dimensions | encoder base `96`; decoder base `144`; latent `64`; input/output `4`; multipliers `[1,2,4,8,8]` | Detected/defaulted at `C/comfy/sd.py:837–847` |
| VAE behavior | residual architecture; resblocks `2`; attention scales `[]`; dropout `0`; temporal downsamples `[false,true,true,true]`; spatial factor `16`; temporal factor `8`; snapshot `patch_size:null` | Same selection at `sd.py:846`; Wan implementation at `C/comfy/ldm/wan/vae2_2.py:661`. ComfyUI uses `patch_size:1`, `temporal_kernel:1` for this image VAE |
| VAE normalization | 64 means and 64 standard deviations | Hard-coded at `C/comfy/latent_formats.py:984–1003` |

Relevant references: [DiT detection](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_detection.py:970), [text configuration](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/llama.py:323), [VAE detection](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/sd.py:837), [latent normalization](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/latent_formats.py:983).

The 128 VAE normalization constants in ComfyUI match the snapshot values. Move them into a named Rust 2.1 profile and test all entries, replacing the current JSON read. The formulas are:

```text
encode normalization: (latent - mean) / std
decode normalization: latent * std + mean
```

Other text config fields:

```text
image_token_id        151655
video_token_id        151656
vision_start_token_id 151652
vision_end_token_id   151653
bos_token_id          151643
eos_token_id          151645

attention_dropout    0.0
initializer_range    0.02
use_cache            true
dtype                bfloat16
```

Token IDs belong in the tokenizer/model profile; initialization values and version/class strings are not inference weight dependencies. Preserve zero dropout for inference and existing KV-cache behavior.

A pre-existing discrepancy should be explicit: qir’s `TextConfig::default()` uses `max_position_embeddings:8192`, while the snapshot and ComfyUI specify `262144`. See `N/crates/qir/src/model/text_encoder.rs:72–84`. Decide whether 8192 is an intentional worker cap; do not describe it as the checkpoint’s maximum.

The full snapshot scheduler settings are:

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.37.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": false,
  "max_image_seq_len": 8192,
  "max_shift": 0.9,
  "num_train_timesteps": 1000,
  "shift": 1.0,
  "shift_terminal": 0.02,
  "stochastic_sampling": false,
  "time_shift_type": "exponential",
  "use_beta_sigmas": false,
  "use_dynamic_shifting": true,
  "use_exponential_sigmas": false,
  "use_karras_sigmas": false
}
```

**ComfyUI does not reconstruct this entire diffusers scheduler configuration.** Its 2.1 model uses:

```text
ModelType.FLUX
shift = 0.69
ModelSamplingFlux timestep table = 10000 entries
```

Sources: `C/comfy/supported_models.py:2079–2083`, `C/comfy/model_base.py:2671`, and [C/comfy/model_sampling.py:416](/C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_sampling.py:416).

qir already supports both behaviors without reading scheduler JSON:

- Diffusers profile: `N/crates/qir/src/model/scheduler.rs:19`, including dynamic shift and terminal `0.02`.
- ComfyUI profile: `N/crates/qir/src/comfy.rs:174`.
- Selection: `N/crates/qir/src/pipeline.rs:622–630`.

Preserve that explicit selection during the weight-loader port. Changing file layout must not silently change the schedule.

**5. Testable implementation steps**

1. **Introduce explicit resolved component paths.** Replace snapshot-only discovery with a model descriptor containing DiT, text-encoder and VAE file paths plus layout/precision profiles. Resolve exact filenames under `diffusion_models/`, `text_encoders/`, and `vae/`; do not enumerate and merge an entire ComfyUI category. Define deterministic precedence when BF16 and INT8 variants both exist.  
   **Tests:** all-BF16, all-INT8, mixed precision, both variants present, missing component, unrelated files, malformed headers, wrong family/shape, and filenames resolved relative to the single selected models root.

2. **Add complete header validation and lazy key adaptation.** Implement the mappings above in a backend that translates requested names and performs row slices/singleton squeezes on demand. Reject incompatible shapes before GPU allocation.  
   **Tests:** 297→265 DiT accounting, 750→750 text bijection, 238→238 VAE bijection, exactly 76 VAE squeezes, no unexpected tensors, and malformed fused dimensions. Use small synthetic tensors for value/order assertions.

3. **Port the BF16 path first.** Load the text decoder through `load_at(..., vb.pp("model"))`; adapt lazy chat norm loading; keep vision/head names unchanged. Split DiT `gate_up` in memory, and squeeze VAE temporal axes.  
   **Tests:** gate-first/up-second forward equivalence; every text tensor resolves; VAE encode/decode with synthetic 5D weights; chat head and vision loading; existing fused-MLP LoRA behavior remains correct.

4. **Remove tokenizer/config file requirements.** Embed the shared tokenizer and introduce a fixed `QwenImage21` profile containing latent normalization and non-inferable architecture settings. Retain explicit scheduler selection.  
   **Tests:** tokenizer SHA256, existing Krea/2.1 token parity, multilingual and combining-character prompts, special tokens, image/reference templates, all 128 latent constants, and exact scheduler regressions. Resolve the 8192-versus-262144 position limit explicitly.

5. **Implement ComfyUI INT8 module loading.** Decode bounded `.comfy_quant` payloads at runtime; validate the format/group size and F32 scale shapes. Import I8 as signed bytes carried in U8. Split fused MLP weights and scales together. Honor the file’s mixed precision instead of applying qir’s current “rotate all eligible tails” policy automatically.  
   **Tests:** negative-byte preservation, row-scale alignment, malformed/unknown descriptors, missing scales, unsupported group sizes, and 192/254 quantized-module accounting. Direct files must bypass converted-cache creation.

6. **Make ConvRot basis handling explicit.** Either implement the ComfyUI activation basis or validate the per-256-column reversal adapter. Add rotated embedding gather and INT8 chat-head support; extend text linears beyond their current BF16/Q8_0 loading.  
   **Tests:** basis matrices and permutation identity, orthogonality, synthetic ComfyUI-quantized linear parity, fused/unfused activation paths, embedding inverse rotation, output head, and accidental double-rotation detection. Compare against ComfyUI’s actual selected backend, including rounding behavior.

7. **Run component parity on installed weights.** These future tests may read tensor data; this research did not. Compare text pre-final-norm states, vision/DeepStack outputs, a DiT block and full forward, VAE encode/decode, and chat logits. Compare BF16 VAE against the snapshot rounded to BF16 as well as its existing F32 baseline. Compare INT8 primarily against ComfyUI loading the same INT8 file.  
   **Gate:** record maximum error/cosine and full-pipeline image metrics; establish tolerances from the existing parity harness rather than assuming byte equality across precision changes.

8. **Run the R4 worker acceptance test.** Start with only the ComfyUI models root configured, embedded tokenizer available, and snapshot/GGUF/cache paths unavailable. Exercise generation, reference editing, fill/expand/matte, and chat; test eviction/reload and cancellation.  
   **Gate:** filesystem tracing shows model reads only from the selected files, no snapshot/config/tokenizer dependency, no converted model/cache creation, and parity/performance meet the rebuild’s existing gates.
