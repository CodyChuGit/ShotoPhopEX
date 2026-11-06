# R1 porting spec: SAM 3.1 on candle

> Written by Codex (`codex exec`, read-only, 2026-10-27) for the Rust rebuild (`docs/comfy/goals/rust-rebuild.md` R1), from ComfyUI's SAM 3.1 code, the checkpoint header, `export_sam31.py` and `gn-perception`; checked by Claude before any code. Line estimates are guesses.

A direct safetensors → Candle CUDA port is feasible. The main engineering work is the model implementation, efficient attention at SAM’s head sizes, and exact resampling/ROI behavior. cuDNN is not mathematically required.

For parity with today’s worker, port the **detector plus the interactive tracker heads**. Video memory attention and multiplex propagation are outside the current ONNX path. Estimated implementation: **4,000–6,000 lines of Rust**, plus optional CUDA performance work.

No files were modified. I inspected source and read only the checkpoint’s length-prefixed JSON header; I did not run inference or benchmarks. Estimates and proposed acceptance tolerances below are explicitly recommendations.

For compact citations, these ComfyUI filenames refer to:

- `sam.py`: [comfy/ldm/sam3/sam.py](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py)
- `detector.py`: [comfy/ldm/sam3/detector.py](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py)
- `tracker.py`: [comfy/ldm/sam3/tracker.py](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/tracker.py)

**1. Required module tree and operations**

Recommended Rust organization:

```text
sam31/
  mod.rs                 public API, Embedding, orchestration
  weights.rs             direct checkpoint loading and shape validation
  attention.rs           scaled attention, padding masks, additive bias
  position.rs            axial RoPE, sine positions, random Fourier positions
  vision.rs              ViTDet trunk
  fpn.rs                 detector and interactive FPNs
  text.rs                CLIP text encoder and language resizer
  geometry.rs            detector box encoder and image-conditioned CLS
  detector_transformer.rs
  detector_heads.rs      scoring and instance mask prediction
  prompt.rs              interactive point/box/mask encoder
  mask_decoder.rs        interactive two-way transformer and mask heads
  ops.rs                 ROIAlign, antialiased resize, channel LayerNorm
```

Keep the existing tokenizer, CPU resampling, crop arithmetic, selection rules, and worker protocol where practical.

| Module | Architecture and exact operations |
|---|---|
| Vision patch embedding | Input `[1,3,1008,1008]`; bias-free `Conv2d(3→1024,k14,s14)` produces `72×72`. Learned absolute position table `[1,577,1024]`: discard its CLS entry, tile the original `24×24` spatial table across `72×72`; **do not interpolate it**. Apply affine LayerNorm before blocks. |
| Vision trunk | **32 blocks**, width **1024**, **16 heads**, head dimension **64**, FFN **4736**. Pre-norm residual attention, then pre-norm residual `Linear→exact GELU→Linear`. Fused QKV with bias; output projection with bias. |
| Vision attention | Blocks **7,15,23,31**, zero-based, attend globally over 5,184 patches. Other blocks use **24×24 windows**, nine windows per image. Both use ordinary noncausal scaled dot-product attention plus **2D axial interleaved RoPE**. Global coordinates are scaled by `24/72=1/3`; local window coordinates are unscaled. Absolute positions and RoPE are both present. |
| Detector FPN | Three independent branches from the final trunk, scales **4,2,1**. Scale 4: transpose conv `1024→512,k2,s2`, exact GELU, transpose conv `512→256,k2,s2`; scale 2: transpose conv `1024→512,k2,s2`; scale 1: unchanged spatial size. Each then uses `1×1` projection to 256 and `3×3,p1` convolution. No FPN normalization or top-down addition here. |
| Interactive FPN | Same structure as detector FPN, **different weights**. Outputs `[1,256,288,288]`, `[1,256,144,144]`, `[1,256,72,72]`. Add `interactivity_no_mem_embed [1,1,256]` only to the flattened lowest-resolution feature. |
| Detector image positions | Fixed 256-channel DETR sine positions, concatenated **y then x**, temperature 10,000, scale `2π`; image coordinates use `index/(size−1+1e−6)`. |
| Text encoder | **24 CLIP blocks**, width **1024**, **16 heads**, head dimension **64**, FFN **4096**, context **32**, vocabulary **49,408**. Learned token and absolute position embeddings. Pre-norm attention residual and pre-norm FFN residual. **Causal attention plus padding-key mask**. Activation is **QuickGELU**, `x*sigmoid(1.702*x)`. Final affine LayerNorm, then language `Linear(1024→256)`. Use all token features; no pooled/text-projection output. |
| Detector box geometry | Width **256**, **3 layers**, **8 heads**, head dimension **32**, FFN **2048**. Normalize image features for pooling. Sum direct box projection, ROI feature projection, sine center/size projection, and positive-label embedding. Then LayerNorm → three encoder layers → LayerNorm → final linear. |
| Geometry CLS | Shares geometry weights but has a **different ordering and cross-attention path**: `norm(final_proj(cls_embed))`; three special geo layers whose cross-attention keys are `image+image_pos`; final `encode_norm`. This token is computed during vision and appended to every detector prompt. |
| Detector encoder | **6 layers**, width **256**, **8 heads**, FFN **2048**. Global image self-attention with sine positions added to normalized Q/K, V without position. Then cross-attention from image to valid prompt tokens, then ReLU FFN. All three sublayers are pre-norm residuals. No deformable attention. |
| Detector decoder | **6 layers**, width **256**, **8 heads**, FFN **2048**. **200 learned queries plus one presence token**. Per layer: query self-attention → prompt cross-attention → image cross-attention with learned box-dependent additive bias → ReLU FFN. These use post-residual norms. Learned reference boxes are sigmoid-transformed and iteratively updated by a shared box MLP. |
| Detector scoring | Prompt MLP `256→2048→256`, ReLU, residual, output LayerNorm. Mean-pool only valid prompt tokens. Project queries and pooled prompt separately; dot product scaled by `1/√256`; clamp logits to `[-12,12]`. |
| Detector segmentation | Normalize encoder memory, cross-attend it to prompts, add residual. Replace lowest FPN level with this memory. Two active top-down stages: nearest upsample + next FPN feature → `3×3 Conv256` → **GroupNorm(8)** → ReLU. Then `1×1 Conv256`; three-linear ReLU mask MLP `256→256→256→256`; matrix product produces **200 masks at 288×288**. |
| Interactive sparse prompt encoder | Width **256**. Random Fourier coordinate encoding using checkpoint matrix `[2,128]`: `concat(sin,cos)(2π*(2*normalized_xy−1)@matrix)`. Compute projection/trig in fp32, cast back. Add point-type embeddings; invalid points use only `not_a_point_embed`. Coordinates receive **+0.5 pixels** first. |
| Interactive dense prompt encoder | Antialiased bilinear resize mask logits `1008→288`; `Conv(1→4,k2,s2)→channel-LN→GELU→Conv(4→16,k2,s2)→channel-LN→GELU→Conv(16→256,k1)`, yielding `72×72`. No mask uses broadcast `no_mask_embed`. Dense image PE uses the same Fourier matrix at pixel centers. |
| Interactive two-way transformer | **2 blocks**, width **256**, **8 heads**, FFN **2048**. Token self-attention internal width 256/head dimension **32**; token→image and image→token cross-attention internal width **128**/head dimension **16**. ReLU FFN; affine LayerNorm after each sublayer. First block’s token self-attention omits positional addition and its residual. Final token→image cross-attention plus LayerNorm. No RoPE. |
| Interactive output heads | Tokens ordered **object-score, IoU, four mask tokens, sparse prompts**. Upscale image output: transpose conv `256→64,k2,s2` + projected `144²` FPN → channel-LN → GELU; transpose conv `64→32,k2,s2` + projected `288²` FPN → GELU. Four separate MLPs `256→256→256→32` generate hypernetwork vectors; multiply by image features for four masks. IoU MLP `256→256→256→4`; object-score MLP `256→256→256→1`. |

Architecture evidence: [sam.py:165](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:165), [sam.py:223](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:223), [sam.py:261](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:261), [sam.py:301](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:301), [sam.py:364](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:364), [sam.py:377](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:377).

Text evidence: [sam3_clip.py:4](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/text_encoders/sam3_clip.py:4), [clip_model.py:58](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/clip_model.py:58), [clip_model.py:95](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/clip_model.py:95), [clip_model.py:163](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/clip_model.py:163), [export_sam31.py:127](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:127).

Detector evidence: [detector.py:84](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:84), [detector.py:122](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:122), [detector.py:145](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:145), [detector.py:225](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:225), [detector.py:314](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:314), [detector.py:367](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:367), [detector.py:415](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:415).

Interactive evidence: [sam.py:28](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:28), [sam.py:93](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:93), [tracker.py:436](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/tracker.py:436), [tracker.py:518](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/tracker.py:518), [export_sam31.py:153](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:153).

Important implementation details:

- Ordinary LayerNorms use PyTorch’s default epsilon **1e−5** here. `LayerNorm2d_op` simply moves channels last, applies that LayerNorm, and moves them back; it is not GroupNorm or RMSNorm. [common.py:63](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/cascade/common.py:63)
- Detector box positions concatenate **sin(y), sin(x), h, w** before the `258→256` projection. ROIAlign uses normalized-box corners scaled by 72. The installed torchvision defaults are **output 7×7, spatial_scale=1, sampling_ratio=−1, aligned=false**, meaning adaptive averaged bilinear samples and no half-pixel coordinate offset. [sam.py:359](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:359), [detector.py:264](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:264), [roi_align.py:129](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torchvision/ops/roi_align.py:129), [roi_align.py:204](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torchvision/ops/roi_align.py:204).
- “Box rotary position bias” is the source’s wording, but the decoder implementation is an **additive learned relative bias**, not a Q/K rotation. It encodes signed-log distances from image coordinates to both box edges through separate x/y MLPs `2→256→8`, then adds their outputs. Presence gets a zero bias row. [detector.py:173](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:173)
- The checkpoint contains three pixel-decoder conv/norm stages, but this three-level forward executes only stages **0 and 1**. The semantic segmentation head is also unused. [detector.py:314](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:314)

**2. Raw checkpoint weight layout**

Found:

```text
C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\
checkpoints\sam3.1_multiplex_fp16.safetensors
```

Header inspection: **223,360-byte JSON header; 1,590 tensors; all F16; 872,661,740 stored scalar values**. These are direct binary-header observations, so there are no text line numbers. The loader independently recognizes SAM31 through its propagation-FPN key. [model_detection.py:1157](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_detection.py:1157)

In the following table, `{weight,bias}` means both suffixes; index ranges are inclusive.

| Raw prefix | Exact suffix layout |
|---|---|
| `detector.backbone.vision_backbone.trunk` | `patch_embed.proj.weight [1024,3,14,14]`; `pos_embed [1,577,1024]`; `ln_pre.{weight,bias}` |
| Same, `.blocks.i`, `i=0..31` | `norm1`, `norm2`; `attn.qkv.{weight,bias}` with weight `[3072,1024]`; `attn.proj`; `mlp.fc1 [4736,1024]`; `mlp.fc2 [1024,4736]`, each linear with bias |
| `detector.backbone.vision_backbone.{convs,interactive_convs}.i`, `i=0..2` | All have `conv_1x1`, `conv_3x3`. Level 0 additionally `dconv_2x2_0`, `dconv_2x2_1`; level 1 additionally `dconv_2x2`; level 2 has no transpose conv. All have weight/bias. |
| `detector.backbone.language_backbone.encoder` | `token_embedding.weight [49408,1024]`; `positional_embedding [32,1024]`; `ln_final.{weight,bias}` |
| Same, `.transformer.resblocks.i`, `i=0..23` | `attn.in_proj_weight [3072,1024]`, `attn.in_proj_bias [3072]`, `attn.out_proj.{weight,bias}`; `ln_1`, `ln_2`; `mlp.c_fc`, `mlp.c_proj`, all with weight/bias |
| `detector.backbone.language_backbone.resizer` | `weight [256,1024]`, `bias [256]` |
| `detector.geometry_encoder` | `boxes_direct_project`, `boxes_pool_project [256,256,7,7]`, `boxes_pos_enc_project [256,258]`; corresponding `points_direct_project`, `points_pool_project`, `points_pos_enc_project`; `label_embed.weight [2,256]`, `cls_embed.weight [1,256]`; `norm`, `img_pre_norm`, `encode_norm`, `final_proj` |
| Same, `.encode.i`, `i=0..2` | `self_attn`, `cross_attn_image`: each has fused `in_proj_weight [768,256]`, `in_proj_bias [768]`, `out_proj.{weight,bias}`. Also `linear1`, `linear2`, `norm1`, `norm2`, `norm3`. |
| `detector.transformer.encoder.layers.i`, `i=0..5` | Same layer key pattern as geometry’s `encode.i` |
| `detector.transformer.decoder.layers.i`, `i=0..5` | Attention prefixes `self_attn`, `cross_attn`, `ca_text`, each with fused `in_proj_*` and `out_proj.*`; `linear1`, `linear2`; `norm1`, `norm2`, `norm3`, `catext_norm` |
| `detector.transformer.decoder` | `query_embed.weight [200,256]`; `reference_points.weight [200,4]`; `norm`; `ref_point_head.layers.0..1`; `bbox_embed.layers.0..2`; `boxRPB_embed_{x,y}.layers.0..1`; `presence_token.weight`; `presence_token_head.layers.0..2`; `presence_token_out_norm` |
| `detector.dot_prod_scoring` | `hs_proj`, `prompt_proj`; `prompt_mlp.layers.0..1`, `prompt_mlp.out_norm` |
| `detector.segmentation_head` | `cross_attend_prompt.in_proj_{weight,bias}`, `cross_attend_prompt.out_proj.*`; `cross_attn_norm`; `pixel_decoder.conv_layers.0..2`; `pixel_decoder.norms.0..2`; `instance_seg_head`; `semantic_seg_head`; `mask_predictor.mask_embed.layers.0..2` |
| `tracker.model.interactive_sam_prompt_encoder` | `pe_layer.positional_encoding_gaussian_matrix [2,128]`; `point_embeddings.0..3.weight`; `not_a_point_embed.weight`; `no_mask_embed.weight`; `mask_downscaling.{0,1,3,4,6}.{weight,bias}` |
| `tracker.model.interactive_sam_mask_decoder` | `obj_score_token.weight`, `iou_token.weight`, `mask_tokens.weight [4,256]`; `conv_s0`, `conv_s1`; `output_upscaling.{0,1,3}.{weight,bias}`; `output_hypernetworks_mlps.0..3.layers.0..2`; `iou_prediction_head.layers.0..2`; `pred_obj_score_head.layers.0..2` |
| Same, `.transformer` | `layers.0..1.{self_attn,cross_attn_token_to_image,cross_attn_image_to_token}.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}`; per layer `norm1..4`, `mlp.lin1`, `mlp.lin2`; final `final_attn_token_to_image.{q_proj,k_proj,v_proj,out_proj}.*`, `norm_final_attn.*` |
| `tracker.model` | `interactivity_no_mem_embed [1,1,256]` |

The three FPN `conv_1x1.weight` shapes are respectively `[256,256,1,1]`, `[256,512,1,1]`, `[256,1024,1,1]`. Transpose-convolution weights use **input-channel, output-channel, kernel-y, kernel-x** ordering.

**Load raw names directly in Rust.** ComfyUI changes names before loading:

1. Removes language encoder weights into a separate CLIP model.
2. Maps `tracker.model.* → tracker.*`.
3. Splits `in_proj_weight/bias` along dimension 0 in **Q, K, V order**.
4. Maps interactive decoder `mlp.lin1/lin2 → mlp.0/2`, and `norm_final_attn → norm_final`.

These are loader adaptations, not checkpoint names. [supported_models.py:2462](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/supported_models.py:2462)

For this port, omit propagation FPN, memory backbone/transformer, multiplex decoder, object-pointer projections and related video parameters. The exporter selects only interactive prompt/mask modules and the no-memory embedding. [export_sam31.py:84](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:84), [export_sam31.py:153](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:153)

**3. Preserve today’s pre/post-processing exactly**

**Image input**

- Read RGB from RGBA8; ignore alpha.
- Convert channels to fp32 by `/255`.
- Independently stretch width and height to **1008×1008**, CHW, batch 1.
- Bilinear interpolation uses half-pixel coordinates: `max((dst+0.5)*src_size/dst_size−0.5,0)`, with edge-clamped taps.
- **No mean/std normalization, aspect-ratio preservation, letterboxing, or image padding.**
- Apply the same procedure to refinement crops.

Evidence: [sam31.rs:164](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:164), [resample.rs:7](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/resample.rs:7). ComfyUI’s node likewise uses square bilinear stretching. [nodes_sam3.py:116](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_sam3.py:116)

**Text and detector prompts**

Reuse the existing `tokenizer.json` and `tokenizers` implementation. Encoding disables the tokenizer file’s special-token postprocessor, takes at most **30 pieces**, prepends **49406**, appends **49407**, then zero-pads to 32. Attention mask is 1 through EOS, then 0. Preserve normalization/BPE behavior from the actual JSON rather than recreating it. [tokenizer.rs:26](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/tokenizer.rs:26)

Detector prompt sequence:

```text
text_features[32] || optional_box_features[M] || geo_cls[1]
```

Text validity comes from its attention mask; all geometry/CLS entries are valid. Detector boxes are normalized **cx,cy,w,h**, derived from original-image pixel corners. [sam31.rs:487](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:487), [sam31.rs:499](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:499)

**Point and interactive-box prompts**

- Scale coordinates independently: `x/W*1008`, `y/H*1008`.
- Positive label **1**, negative **0**, box top-left **2**, bottom-right **3**, padding **−1**.
- Absent points contribute one `[0,0],−1` entry.
- No box contributes one additional `[0,0],−1` entry.
- A box contributes its two corners instead of that additional padding entry.
- Exactly **one actual point**, including a single negative point, selects multimask mode. Choose the highest predicted IoU among tokens **1–3**, using strict `>` updates; otherwise choose token **0**.
- If object-score logit is **not greater than zero**, replace the selected high-resolution mask with **−1024**.
- Upsample selected `288²` logits to `1008²`; subsequent refinement calls supply only the mask, with no original point/box prompts.
- Perform at least one pass, then `max(refine_iterations,1)−1` mask-feedback passes.
- Finally resize `1008²→original size` and threshold strictly **logit > 0**.
- Returned point/box instance score is **1.0**, bbox `[0;4]`; predicted IoU only selects the mask.

Evidence: [sam31.rs:368](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:368), [sam31.rs:399](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:399), [sam31.rs:445](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:445).

**Text detection and refinement**

1. Compute `sigmoid(pred_logits)`.
2. Keep scores **strictly greater** than threshold; descending sort; truncate to `max_detections`. Defaults: **0.5, 1, 2 refinement iterations**.
3. **No box NMS, mask NMS, or presence-score multiplication.** `presence` is produced by the graph but ignored by this runtime.
4. Resize each selected `288²` mask **directly to original-image size**. Scale normalized predicted xyxy boxes to image pixels.
5. With zero refinement iterations, threshold that coarse mask at zero.
6. Otherwise pad the predicted box by **10% of its width/height on each side**. Use f64 arithmetic, truncate toward zero, then clamp to image bounds. Invalid crops return the coarse binary mask.
7. Run trunk + interactive FPN once on the stretched crop.
8. Crop the original-resolution coarse **logits**, resize to `1008²`, and run the interactive mask-feedback path `iterations` times.
9. Resize final crop logits to crop dimensions and threshold at zero.
10. **OR** refined foreground into the coarse binary mask. Refinement can add foreground; it cannot remove coarse foreground.

Evidence: [sam31.rs:538](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:538), [sam31.rs:569](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:569), [sam31.rs:628](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:628).

At the worker boundary, identical image content reuses an embedding. If both points and a box are supplied on the interactive route, they produce **separate instances**, not one combined prompt. Empty masks are dropped. Bounds are foreground-inclusive pixel bounds expressed as `x,y,width,height`; pixel count tests coverage `>0.5`. [engine.rs:209](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:209), [engine.rs:224](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:224), [engine.rs:256](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:256), [graph.rs:92](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/graph.rs:92).

**4. Reuse versus new work**

| Existing code | Reuse decision |
|---|---|
| qir safetensors `VarBuilder` loading | Reuse approach, creating F16 model and F32 text builders over the same file. No conversion/export required. Existing example: [krea2_pipeline.rs:98](C:/Users/5090/Documents/genai-native/crates/qir/src/krea2_pipeline.rs:98). |
| qir vision block | Reuse its linear/norm/reshape structure as a starting point. Its block is private and Qwen-specific, so it is not a drop-in module. Change position handling, patchification, windows, epsilon, FFN width, names and GELU. [vision.rs:243](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:243) |
| qir Qwen vision preprocessing/positions | Do not reuse: normalization, temporal patch duplication, merge order, split-half RoPE, and interpolated absolute positions differ. [vision.rs:85](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:85), [vision.rs:160](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:160), [vision.rs:193](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:193) |
| qir chunked attention | Useful reference implementation: fp32 scores, bounded query chunks. Extend to padding masks and arbitrary additive box bias; reduce chunk size if necessary. It is currently private. [dit.rs:760](C:/Users/5090/Documents/genai-native/crates/qir/src/model/dit.rs:760) |
| `crate::flash` | **Not directly reusable**: BF16 only, fixed head dimension **128**, noncausal, with only a restricted constant key-range bias. SAM needs dimensions **64/32/16**, causal/padding masks and spatially varying decoder bias. [flash.rs:16](C:/Users/5090/Documents/genai-native/crates/qir/src/flash.rs:16), [flash.rs:51](C:/Users/5090/Documents/genai-native/crates/qir/src/flash.rs:51), [flash.rs:128](C:/Users/5090/Documents/genai-native/crates/qir/src/flash.rs:128) |
| qir RoPE | Reuse interleaved rotation math through Candle `rope_i`; generate SAM-specific x-then-y tables and scale. qir’s custom BSHD bridge is BF16-only and gated behind `sage`; adapting it is optional. [rope.rs:1](C:/Users/5090/Documents/genai-native/crates/qir/src/rope.rs:1), [rope.rs:75](C:/Users/5090/Documents/genai-native/crates/qir/src/rope.rs:75) |
| qir fused norms | Not a direct match: AdaLN/no-affine and RMSNorm paths differ from SAM’s affine LayerNorm/GroupNorm. Start with `candle_nn` norms. [fusednorm.rs:1](C:/Users/5090/Documents/genai-native/crates/qir/src/fusednorm.rs:1), [fusednorm.rs:138](C:/Users/5090/Documents/genai-native/crates/qir/src/fusednorm.rs:138) |
| qir conv helpers | Reuse bias/layout patterns and the non-cuDNN `conv2d` path. Existing VAE wrappers are private and largely stride-1-specific; SAM needs a small general wrapper. Do not enable `cudnn`. [vae.rs:152](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vae.rs:152), [vae.rs:192](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vae.rs:192) |
| gn-perception tokenizer/postprocessing | Reuse essentially unchanged. This is the safest way to preserve PhotoshopEX behavior. |

Candle 0.11 already exposes linear, embedding, convolution, transpose convolution, affine LayerNorm and GroupNorm. Its CUDA backend includes a native transpose-convolution implementation; this does **not** require cuDNN. [Candle NN 0.11 API](https://docs.rs/candle-nn/0.11.0/candle_nn/), [Candle CUDA implementation](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/cuda_backend/mod.rs#L3789).

Plan new implementations for:

- Exact torchvision-style ROIAlign.
- PyTorch-compatible **antialiased bilinear** mask downsampling.
- SAM position generation, window partitioning and box relative bias.
- The model-specific transformer and heads.
- Efficient F16 attention dispatch at the required head dimensions.

I did not verify a suitable stock Candle ROIAlign or antialiased-resize implementation; treat them as required new operations until demonstrated otherwise. Ordinary outer resampling can remain the current Rust CPU function.

For transpose convolutions, a useful optimization is available: every required transpose conv has **kernel=stride=2, padding=0**, so output patches do not overlap. It can be expressed exactly as GEMM producing `4*out_channels`, followed by reshape/permutation into spatial pixels and bias addition. This is a mathematical implementation option, not a measured speed claim.

Use **exact GELU / `gelu_erf`** for PyTorch `nn.GELU()`. Do not blindly copy qir vision’s `.gelu()` call; qir itself distinguishes `.gelu()` and `.gelu_erf()`. [vision.rs:279](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:279), [vision.rs:309](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vision.rs:309).

**5. Port order, parity tests and Rust size estimate**

The five graph boundaries are already useful reference interfaces. Vision returns detector/interactive FPNs, image position and geometry CLS; text returns features/mask; geometry returns box tokens; decoder returns boxes/logits/masks/presence; heads returns all four masks/IoUs/object score. [export_sam31.py:84](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:84), [export_sam31.py:127](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:127), [export_sam31.py:140](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:140).

**Proposed diagnostic dumping**, to implement during the port—not performed in this study:

- Add an opt-in dump hook at each graph invocation, writing contiguous arrays with name, shape, dtype and stage metadata to safetensors or raw little-endian data plus JSON.
- For device-bound ORT tensors, copy to CPU explicitly, then extract. The current geometry-CLS extraction demonstrates this sequence. Dump all eight vision outputs before constructing `Frame`. [sam31.rs:308](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:308)
- Dump exact graph inputs too: resized pixels, token IDs/mask, normalized boxes, assembled prompts, point labels/coordinates, mask feedback and `has_mask`.
- Have Candle consume those **same reference inputs** at each boundary. This isolates each module from upstream drift.
- Then test the complete chain with Candle-generated intermediates.
- To compare internal layers, use a diagnostic copy of the ONNX graph with selected existing node outputs added as graph outputs. Preserve external-data references and reduce optimization for diagnostic runs. Original graph exports do not expose every layer.
- Dump crop rectangles, selected query/token indices, rankings, resized logits and final masks. Tensor similarity alone will miss decision changes near thresholds.

| Step | Work and parity gate | Estimated Rust lines |
|---|---|---:|
| 1 | Add backend-neutral tensor/embedding interfaces, checkpoint key/shape validation, and diagnostic serialization. Verify all required keys, fused QKV slice order, dtype casts and saved fixture round-trips. Keep ORT available as a development-only reference. | 350–500 |
| 2 | Reuse tokenizer and preprocessing. Exact-match token IDs/masks, resized pixel planes, normalized boxes and sparse-label lists against current Rust behavior. Test non-square images, alpha changes, long/empty text and crop edges. | 100–200 |
| 3 | Implement ordinary/chunked attention, norms, exact GELU, positions and transpose-conv wrapper. Compare primitive outputs against exposed ONNX intermediates on representative SAM shapes; test boolean-mask polarity and additive-bias placement. | 450–650 |
| 4 | Implement CLIP and resizer first. Feed identical IDs/masks; compare `[1,32,256]` and exact boolean mask against `text.onnx`. Keep CLIP F32. Test padding and full-context prompts. | 300–450 |
| 5 | Implement ViTDet and both FPNs. Compare patch output, blocks 0/7/15/23/31 where diagnostic outputs permit, then detector/interactive FPN outputs and position tensor. Test full-frame and crop inputs; verify tracker-only execution equals full vision’s tracker outputs. | 650–900 |
| 6 | Implement geometry CLS and box encoder, including ROIAlign. Compare CLS to `vision.onnx`, box tokens to `geometry.onnx` using ORT’s FPN input. Test tiny, full-frame, edge/outside and multiple boxes. | 350–550 |
| 7 | Implement interactive prompt encoder, two-way transformer and raw output heads. Compare all four `288²` mask planes, four IoUs and object score against `heads.onnx`. Test positive/negative/padding labels, boxes, absent mask and nontrivial mask feedback. Then test mask choice and NO_OBJ behavior. | 550–800 |
| 8 | Implement detector encoder/decoder, box bias, scoring and segmentation. Feed saved ORT FPN/prompt tensors; compare **all 200** boxes, logits, masks and presence. Compare reference-box updates per decoder layer when needed. | 750–1,050 |
| 9 | Integrate current selection, iterative refinement, image cache and worker lifecycle. Run current SAM fixtures through both backends; compare instance ordering, empty removal, masks, scores, bounds and pixel counts. Remove ORT from the production dependency graph and configuration. | 250–450 |
| 10 | Optimize only after parity: F16 flash attention, transpose-conv GEMM, convolution tiling, position caches and buffer reuse. Rerun affected tensor gates plus the complete suite after each optimization. | 250–500 |

**Total estimate: approximately 4,000–6,000 Rust lines**, including parity support; these are engineering estimates. Custom CUDA attention/resampling/conv optimizations could add roughly **600–1,500 CUDA lines**, excluded from that Rust estimate.

Suggested numerical acceptance:

- Integer IDs, masks, labels, crop rectangles and discrete choices: exact.
- F32 isolated outputs: begin with `atol=1e−4, rtol=1e−4`, then establish per-tensor tolerances from actual errors.
- F16 outputs: report max error, RMS error, relative error and cosine similarity; do not use one permissive tolerance for every tensor.
- Explicitly count threshold crossings, changed best-mask tokens and changed ranked queries.
- Require final full-mask IoU and boundary agreement as well as bounds agreement.

These tolerances are **proposals**, not measured guarantees. The existing export verifier only reports maximum absolute differences; the repository documents a 16-case bounds-IoU acceptance suite. Extend that with full-mask comparison because matching boxes can conceal mask drift. [export_sam31.py:310](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:310), [EXPORT.md:62](C:/Users/5090/Documents/genai-native/crates/gn-perception/docs/EXPORT.md:62)

The existing harness launches `gn-sam` with text/point/box prompts and saves masks, so a Candle backend can share that surface. [engines.rs:217](C:/Users/5090/Documents/genai-native/crates/gn-harness/src/engines.rs:217)

**6. Risks and practical constraints**

**Numerics**

Production exports use **F16 vision/geometry/detector/interactive modules**, while CLIP stays **F32**. External graph inputs/outputs remain F32. Preserve these conversion boundaries initially—including the F32 CLIP output cast before the F16 language resizer. Loading everything as BF16 would introduce an avoidable precision change. [export_sam31.py:46](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:46), [export_sam31.py:69](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:69), [export_sam31.py:127](C:/Users/5090/Documents/genai-native/tools/export_sam31.py:127)

The largest parity traps are:

- Tiled absolute positions versus interpolated positions.
- Interleaved versus split-half RoPE, axis order and global coordinate scale.
- LayerNorm epsilon **1e−5** versus qir vision’s **1e−6**.
- Exact GELU versus tanh approximation; QuickGELU in CLIP.
- Geometry CLS’s special execution order.
- ROIAlign adaptive sampling and boundary behavior.
- Antialiased internal mask resize versus ordinary outer resize.
- Softmax/reduction precision, iterative box updates and logits near zero.
- Score ties or tiny score changes altering which detection receives refinement.

The existing qir flash kernel also rounds its probability/value product through F16 despite BF16 inputs; its “exact” label should not be interpreted as bit-identical to ORT attention. [flash_fwd.cu:1](C:/Users/5090/Documents/genai-native/crates/qir/kernels/flash/flash_fwd.cu:1)

**Speed**

A naïve Candle implementation is likely slower than optimized ORT CUDA initially—**prediction, not benchmark evidence**. Risks include materialized attention matrices, extra contiguous copies, kernel-launch overhead and im2col convolution bandwidth. ORT currently enables level-3 graph optimization and CUDA execution, with CPU fallback for unsupported operators. [runtime.rs:159](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:159)

Prioritize:

1. F16 fused attention for trunk **D=64**.
2. Bounded-memory detector self-attention **D=32**.
3. Efficient transpose convolutions and high-resolution `3×3` convs.
4. Avoiding host round trips and unnecessary detector-FPN work for crops.

Keep small causal text attention and bias-bearing decoder attention straightforward until profiling justifies specialization. Extending the current flash kernel to all masks/biases at once is a substantially larger task than adding an unmasked D=64 path.

A native model can share one trunk across full-image and crop execution. The optional ONNX tracker-only graph currently duplicates the trunk in another session; repository documentation reports approximately **0.9 GB** extra disk/VRAM for that graph. [EXPORT.md:52](C:/Users/5090/Documents/genai-native/crates/gn-perception/docs/EXPORT.md:52)

Do not assume CUDA graph capture is needed for acceptance: the current implementation defaults to ordinary execution because capture has failed on this model. [sam31.rs:202](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:202)

**Memory**

Calculated from inspected header shapes and source-defined tensor dimensions:

| Allocation | Approximate size |
|---|---:|
| Required selected weights, all F16 | **1.70 GB / 1.58 GiB** |
| Same, keeping CLIP F32 as today | **2.41 GB / 2.24 GiB** |
| Detector + interactive FPN caches plus lowest image position, F32 | **228 MB / 218 MiB** |
| All 200 detector masks at `288²`, F32 | **66 MB / 63 MiB** |
| Dense global trunk attention scores, `[1,16,5184,5184]`, F32 | **1.72 GB / 1.60 GiB** |
| Dense detector-encoder scores, `[1,8,5184,5184]`, F32 | **860 MB / 820 MiB** |
| One `256×288×288` input’s `3×3` im2col workspace, F16 | **382 MB / 365 MiB** |

These are individual allocations, **not peak-memory measurements**. Softmax copies, FFN intermediates, convolution workspaces and allocator retention can add considerably more. Shapes are defined in [sam31.rs:83](C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:83), [sam.py:224](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/sam.py:224), and [detector.py:314](C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/ldm/sam3/detector.py:314).

Chunked/fused attention is therefore a production requirement. For convolution, qir already documents im2col’s ninefold input expansion; avoiding cuDNN exchanges deployment simplicity for potential workspace and speed costs. [vae.rs:436](C:/Users/5090/Documents/genai-native/crates/qir/src/model/vae.rs:436)

Build with Candle CUDA and keep **both** `candle-core/cudnn` and qir’s optional `cudnn` feature disabled. CUDA/cuBLAS remain dependencies; ONNX Runtime, cuDNN and Python export can disappear from the SAM production path. qir already separates CUDA from cuDNN features. [Cargo.toml:44](C:/Users/5090/Documents/genai-native/crates/qir/Cargo.toml:44)
