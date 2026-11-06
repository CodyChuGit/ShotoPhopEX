# AusBoss's "Krea 2 Clothing From Reference" workflow, read for PhotoshopEX

Read on 2026-10-21 from `krea2ClothingFrom_v12.json` (the owner's download; ComfyUI workflow
JSON with four subgraphs), the AusBoss node help pages (`js/docs/*.md` in
[ausboss/ComfyUI-AusBoss](https://github.com/ausboss/ComfyUI-AusBoss), MIT) and the
[yijunwang2/krea2-anypaint](https://huggingface.co/yijunwang2/krea2-anypaint) model card. The
workflow: "Click the clothes in your picture. Krea 2 repaints them to match the clothes in a
second picture." Nothing of it was run here: the AusBoss pack is not installed and the LoRAs are
Krea 2 Community License (never a default for us). What follows is what it does, why it works,
and what PhotoshopEX took from it the same day.

## 1. Anatomy

Top level (reading order of the graph):

1. **Your picture** → `ImageCropRotatePad` (an optional crop with handles; feather 24,
   fill `#808080`, `stitch_blend 32`).
2. **Mask by Touch** (`AUSBOSS_NODES_MaskByTouch`): SAM 3.1 (`sam3.1_multiplex_fp16`, the
   Comfy-Org checkpoint) on a click; hover previews what a click would pick; a second click
   adds a piece; arrows cycle "the next thing it could mean"; **grow / soften / smooth /
   fill holes** on the mask; `ask` = a text search; `fine_edges` (BiRefNet matting) for hair
   and fur; `if_nothing: stop the run`.
3. **The clothing picture** → the same crop node. **"What to take from it"**: a short phrase
   ("the cardigan and the top under it").
4. Subgraph **Cut out the area**: `RefineMask` (expand 12, blur 4, fill holes) →
   `CropForInpaint` (context factor 2, 192 px of extra context, 1.0 MP target, multiple of
   16, blend 24, `keep_inside`) → `ImageCompositeMasked` paints the picked area **flat
   mid-grey** → `GrowMask` 24 px tapered = the **noise mask**. Every one of those pixel
   numbers is scaled by `min(1, sqrt(w·h) / 1500)` (`MathExpression` nodes), so a 600-px
   picture gets a 5-px grow and a 10-px blend, a 3000-px one the full 12 and 24.
5. Subgraph **Put them side by side**: the clothing resized to the crop's height (Lanczos,
   multiple of 16) and stitched to the **right** of the grey-holed crop; a second pair with the
   un-greyed crop; the noise mask and the picked area each composited onto an empty mask of the
   pair's size, **left half only**.
6. Subgraph **Write the prompt** (switchable, on by default): `TextGenerate` on the **Krea 2
   text encoder itself** (Qwen3-VL-4B) sees the side-by-side pair and gets this instruction
   plus the user's phrase:

   > This picture has two photos side by side. The left photo is the one being finished: a
   > flat mid-gray area on it will be painted. The right photo is a reference. Write one
   > text-to-image prompt for the finished picture, as one paragraph of 60 to 110 words that
   > describes a side-by-side pair of photos. Describe the left photo first: the same person,
   > pose, framing, setting and lighting, now wearing the clothing from the right photo where
   > the gray area is. Describe that clothing exactly as the right photo shows it: garment
   > type, colour, material, knit or weave, pattern, collar, buttons, length and fit. Then say
   > briefly that the right photo shows the same clothing on another person. Never call
   > anything gray, blank, masked or painted. Output only the prompt. What to take from the
   > right photo: *{the user's words}*

   (temperature 0.7, top-k 64, top-p 0.95, `thinking: true`, 300 tokens.) A "Show Text" node
   displays "the prompt Krea 2 sees".
7. Subgraph **Paint with Krea 2**: `Krea2Encode` (prompt; `vae`; **reference = the grey-holed
   crop**; **mask = the picked area**; `vlm_reference: true`) → `Krea2OutpaintModelPatch`
   (`kv_cache: true`, `placement: whole canvas`) on the model with the **AnyPaint LoRA**
   (`krea2_anypaint_rank32`, strength 1); `VAEEncode` of the **plain side-by-side pair** →
   `SetLatentNoiseMask` with the left-half noise mask → `KSampler` **er_sde, 8 steps, CFG 1,
   simple, denoise 1** → decode → `ImageCrop` the left half back out → `StitchInpaint`
   (`color_match 0.5`, classic seam) → an A/B compare and Save.

## 2. Why each piece is there (from the node docs and the model card)

- **Krea 2 is a text-to-image model**: it has no mask input and no reference slot of its
  own. Three tricks give it both:
  - **In-context pair.** The reference sits in the same canvas (right half) and the noise
    mask keeps that half fixed, so the sampler "sees" the clothing through self-attention
    while repainting only the left hole. This is the oldest trick in the book (IC-LoRA style)
    and needs no model change.
  - **Registered reference tokens.** `Krea2Encode` attaches the grey-holed crop as reference
    latents (longest side 384 px, multiple of 16) and, with `vlm_reference`, also shows it to
    the vision tower so the text encoding is grounded in the picture. `Krea2OutpaintModelPatch`
    modifies the attention layers so those tokens are **registered onto the canvas grid**
    (`whole canvas` spreads them over the frame) instead of acting as a loose style hint, and
    computes their keys and values **once per run** (`kv_cache`). Without a matching LoRA the
    base model is "inconsistent"; **AnyPaint** was trained exactly this way: the masked canvas
    as one 384-px reference, generated pixels in that reference replaced by the **median colour
    of the known pixels** (not grey: on dark or saturated pictures the model paints grey
    back), unmasked latents restored after every flow step, a **32-px boundary band** left for
    the model to redraw.
  - **The prompt describes the finished canvas**, not an instruction: "the model generates
    everything and matches the reference where needed". Hence the prompt writer, and hence
    "never call anything gray": a description of grey would be obeyed.
- **The mask stack**: grow (so the old garment's edge is inside the repainted area and a
  bulkier piece has room), fill holes, blur the sampling mask only a little; **the paste-back
  mask is feathered separately** from what the sampler sees (`CropForInpaint`'s "Blend"
  widens and blurs only the stitch mask). Everything scaled by picture size.
- **The crop**: mask bounds × 2 plus 192 px, inside the picture, to 1 MP: the model works
  at its native size on the part that matters, and the stitch puts it back bit-exactly outside
  the blend; `color_match 0.5` removes the colour step a model leaves across the seam.
- **Speed**: 8 steps at CFG 1 (Turbo), the encoder shared with the prompt writer, the
  reference's KV computed once.

## 3. What PhotoshopEX already had, and what it took

| Workflow piece | PhotoshopEX before 2026-10-21 | Taken on 2026-10-21 |
|---|---|---|
| SAM 3.1 click/box/text mask, grow, fill holes, "next meaning" | Object Selection tool: click, box, Refine mode, refine edge, text field; no grow / fill holes on the tool | `generate.fill` `grow` (px); fill holes and the mask options stay on the list (§4) |
| Crop with context, 1 MP, multiple of 16, inside the picture | `margin` 25 % of the selection's longer side, 0.75 MP cap, 16-px grid, inside the canvas | unchanged (equivalent) |
| Sampling mask: hard, grown 24 px; paste-back mask feathered separately | one feathered mask serves as the noise mask and (dithered) as the layer mask | unchanged; a separate hard sampling mask is a measurable A/B for later |
| **Grey out the picked area** so the model paints a hole instead of copying the old object | the picture went out intact: the model saw the old garment it was told to replace | `erase` (and `eraseSelection` on templates): the selection flattened to mid-grey in the request image |
| **A reference image for a fill** | fills took none (edits and text-to-image did) | `qwen-edit-2511/fill-ref[-lightning-8]`: one or two references on the 2511 fill graph, the prompt naming the hole and "exactly as the references show it"; Auto switches to it when references are given; the Generate panel's fill page has the deck |
| **Prompt writer sees the pair side by side** | the enhancer saw the picture alone | the fill-from-reference enhancer sees the grey-holed picture and the first reference side by side (`side_by_side`) and names the thing exactly as the reference shows it |
| Stitch with tone match | the dithered soft layer mask | not yet (§4) |
| Krea 2 inpainting (AnyPaint, registered references, KV cache) | Krea 2 is text-to-image only, on both routes | not yet: the recipe is written into the native plan (§4) |

The 2511 route is a different model from Krea 2 (an instruction-driven editor with native
multi-image slots, Apache-2.0), so the pair trick and the AnyPaint patch are not needed there:
`image2` *is* the reference slot. What carries over is the preparation (the hole, the grow, the
prompt that describes the thing as the reference shows it) and the prompt writer's view.

## 4. Still to take (ordered by value per hour)

1. **Measure more.** `docs/comfy/bench/scripts/fill-ref-live.ps1` ran the swap on two Krea 2
   pictures the same day (`benchmarks.md` §5h): the grey hole gave the reference's cable knit,
   the plain fill recoloured the old cardigan and kept its buttons; Lightning 28 s, the 40-step
   tier 150 s. Still to judge over more seeds: `grow` 0 vs 8, `erase: false` on the reference
   template, typed vs enhanced.
2. ~~**Mask options on the Object Selection tool**: Widen, Fill Holes, Soften~~ (done later on
   2026-10-21: `grow` / `fillHoles` / `soften` on the `select.*` detector commands, the options
   bar's three controls, Widen on the fill page). Still open: a "next meaning" cycle from the
   detector's other candidates (`select.byPoint` returns one mask today).
3. **Stitch tone match**: measure the colour step across the seam on the bench fills; if it is
   there, a per-line offset estimated along the seam (AusBoss's "classic" tone match) is a
   small engine change on the result layer.
4. **Separate sampling and paste-back masks**: hard (grown) noise mask, feathered layer mask;
   A/B against today's one feathered mask.
5. **Krea 2 inpainting natively** (plan Phase 4 follow-up): the AnyPaint recipe is a LoRA
   loader (planned anyway), a reference-token path in the DiT's attention with a per-run KV
   cache (the prefix cache of the 2.1 pipeline is the same idea), median-colour hole fill, a
   32-px redraw band, and the pair trick needs nothing at all. Licence: the LoRAs are Krea 2
   Community License, so BYOM, never a default; the base model is the same Krea 2 Turbo the
   engine already serves.
6. **Size-scaled pixel numbers**: the fork's feather is already 4 % of the request; grow and
   blend should scale the same way when they get UI.
