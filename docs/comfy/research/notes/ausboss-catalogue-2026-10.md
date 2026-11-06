# AusBoss's catalogue (31 models on Civitai, read 2026-10-26), what PhotoshopEX takes

Read from [civitai.com/user/AusBoss/models](https://civitai.com/user/AusBoss/models) through
the site's public API (`/api/v1/models?username=AusBoss`): every page's description, files
and measurements. Nothing was downloaded or run here; the workflow JSONs stay the owner's to
fetch. The earlier note [`ausboss-krea2-clothing-workflow.md`](ausboss-krea2-clothing-workflow.md)
read one workflow's graph in detail; this one reads the whole catalogue for the **concepts**,
with his numbers, and maps them onto the fork. His numbers all come from one RTX 5090, like ours.

## 1. The catalogue

**LoRAs for Qwen-Image-2.1** (his own, ai-toolkit on the Comfy-Org INT8 ConvRot base, rank 32,
alpha 32, 1 000–2 000 steps; all under the Qwen Research License as derivatives):

| LoRA | What it fixes | His measurement (held-out pictures, same seed) |
|---|---|---|
| **Consistency** (152 MB) | 2.1 edits come back a few percent taller or nudged, different per seed; local edits repaint hair, signs, texture | restyles: worst corner off, median 24.3 px → 1.6 px; local edits on people: pixels changed outside the edit 12.8 % → 7.5 %, PSNR outside the edit 27.7 → 31.6 dB; trained on 950 pairs whose drift he measured and removed, many run backwards (the edit as source, the untouched original as target) |
| **Outpaint** v1/v2 (152 MB) | pad with flat `#808080`, the LoRA fills the grey and leaves the picture where it was | plain 2.1 moved or rescaled the picture in 13 of 16 renders (up to 284 px), 5 with a seam; with the LoRA 32 of 32 in place, no seam; kept-area PSNR 16.4 → 34.6 dB; tilted pictures (17°, grey wedges): plain moved all 12, LoRA 24 of 24 in place |
| **Restore** (152 MB) | 2.1's own upscale over-sharpens (gritty skin, invented pores, older faces, a few px shift) | 37 small pictures vs their full-size originals: shift > 2 px → 0.3 px; face match 0.79 → 0.88; colour shift 1.9 % → 0.9 %; draws slightly less grain than the original (soft); strength 0.5 for blurry inputs |
| **Outfit Swap consistency** (152 MB) | a two-picture swap zooms out to fit the outfit, redraws the body, softens the face, repaints the background | 16 swaps: shift 3.6 → 0.1 px; face pixels changed 14.1 → 1.7 %; background repainted 10.8 → 0.9 %; old outfit cleared 0 of 3 → 2 of 3 seeds |

Plus a **FLUX 3 Image Klein Outpaint LoRA** (same recipe on FLUX 3 Klein Base: plain Klein
zoomed or slid the picture in 36 of 40 runs; with the LoRA 0.1–1.5 px, strength 1.5) and a
Klein **Restore** LoRA.

**Workflows** (all on his MIT node pack, `ComfyUI-AusBoss`):

| Family | Workflows (his time on a 5090) |
|---|---|
| Qwen-Image-2.1 | Text to Image starter (13 s at 1536², 25 steps); Generate + Edit (edit 15 s at 2 MP, 6 s Fast); Turbo Generate + Edit (official 2.1 Turbo checkpoint, 8 built-in steps: new picture 4 s at 2 MP, edit 9 s); Turbo Multi-Image Edit (up to 6 pictures, 10–40 s); Edit + Realign (18 s, 11 s Fast); Outpaint and Rotate + Outpaint (8 s); Inpaint, click to mask (12–15 s); **Cut Out + Remove** (12 s, 6 s with Turbo); Outfit Swap (23 s, 13 s Fast); Person Swap (BFS LoRAs by Alissonerdx, MIT; 20 s); Tiled Upscale to 8 MP (9 s one pass, 43 s for four tiles) |
| Krea 2 Turbo | Inpaint click-to-mask and Clothing from reference (AnyPaint LoRA, 15–20 s); Outpaint and Rotate + Outpaint (14 s) |
| FLUX 3 Image Klein (Oct 2026, Kommunity licence) | Edit (6–8 s, 16 steps, INT8 5.8 GB); Multi-Image Edit (12–24 s); Outpaint (8 s); Upscale with SeedVR2 + Restore (13–24 s for 4 MP) |
| Others | FIBO 2.0 Generate + Edit (CC BY-NC; 3 s an edit at 4 steps); MOSS-Image text to image (quoted words rendered right, layout boxes, "Change words"); FLUX.2 Klein 9B Outpaint; LTX 2.3 video outpaint; MiniMax H3 orbit; **Simple Video Watermark Remover** (SAM 3 "watermark" + Mask Refine + **LaMa**) |

## 2. The concepts, and what they are worth to us

Ordered by how much of the fork's open list they close, with the licence of what they need.

### 2.1 Realign to source (model-agnostic, no model file)

Every edit model he uses drifts: 2.1 by 24–35 px median on restyles, FLUX 3 Klein "up to
202 px" when the prompt describes the whole picture, FIBO "a third of a pixel". His *Edit +
Realign* workflow measures the drift between the original and the edit and moves the edit
back on top ("style edits sat 35 px off at the median; after Realign under 5 px"), and pads
the picture with a **mirrored margin** first so the edit can move without leaving an empty
strip at an edge. For us: a post-process in the result compositing of `generate.edit` and of
the whole-picture routes (2.1 fill/edit native, Generate Similar) that estimates the
translation (and a small uniform scale) between the request picture and the result on luma
(phase correlation, then a refinement on edges) and resamples the result back before the layer
mask is applied. Pure Rust, no model. The 2511 fill under a noise mask does not drift (the
latent is pinned), so it is for the routes that regenerate the whole picture.

### 2.2 The removal takes the shadow with it (the differential composite)

*Cut Out + Remove*: "only the thing, a band around it, and **what changed with it** are taken
from Qwen's picture. Every other pixel is your own." 2.1 regenerates the whole picture, so the
shadow and the reflection go; the composite then takes the pixels that changed (connected to
the hole, within a search band) and leaves the rest. His caveats are measurements for us:
a long hard shadow can leave a faint streak; a hand holding the thing goes with it (the second
box, "what to leave in its place", is our fill prompt: "her empty hand resting near her chin"
kept a hand); naming two things ("the woman and her bag") took only the bag; see-through parts
keep what was behind them; a thing partly hidden comes out as the visible part. His *Inpaint*
page says the same from the other side: "Pick its shadow too, or the shadow stays."

For us: on a whole-picture route, grow the layer mask by a change-detection band (|result −
original| on luma and chroma above a threshold, connected to the hole, feathered); on the 2511
crop route, the rewriter can be asked to name the shadow/reflection and the detector's mask
grown to include it (SAM 3.1 segments shadows when asked "the dog's shadow"). Both measurable
on the bench sets (the dog on the lawn, the cone, the hydrant have shadows).

### 2.3 Two outputs from one name: cut-out and removal together

The same workflow saves the object as a transparent PNG (2.1's RGBA output, or the SAM mask)
and the picture without it, the same size, "so you can stack them, move the thing, or put
something else behind it". For PhotoshopEX that is a layer operation: **Remove to its own
layer** — the object lifted to a new layer (the detector's mask, or the matte route when
allowed) and the removal filled underneath, one command. The product already has both halves
(`generate.removeBackground` to a layer mask, the removal); the command that does both and
names the layers is small.

### 2.4 Outpaint: grey pad, no noise mask, stitch with a blend, re-roll by cropping

His outpaint recipe for 2.1 and FLUX 3: pad flat `#808080` to multiples of 32 (about 1–2 MP),
the padded canvas as the one reference at `resolution 0`, the trigger instruction word for
word ("Outpaint the image: replace the solid gray areas with a seamless continuation of the
scene, keeping the existing picture unchanged." on Klein), **no `SetLatentNoiseMask`** ("on
Qwen 2.1 that draws a visible rectangle at the seam"), the original pasted back with a 32-px
blend and a tone match; for Krea 2 "the new area starts from your picture's own colour" (not
grey: dark or colourful pictures came back with a grey frame). Rules of thumb measured: pull
out at most the picture's own size per side per run (more paints a second scene); crop off
noisy or framed edges first; a vignette carries on into the new area; **re-roll a strip by
cropping past it and padding back out**; tilted pictures leave grey wedges the same recipe
fills (*Rotate + Outpaint*: straighten a crooked photo and fill the corners).

For us: the native 2.1 expand already places the picture at its own position inside the
canvas (`RefImage::offset`) with the mask confining the result, which is the LoRA-free way of
"stays in place" (judged by eye, no drift seen on the contact sheet); the 2511 expand has
`expandMargin`. To take: the **tone match at the seam** (open since the first note), a
**Straighten and fill** command (rotate arbitrary → the wedges become the expand mask), the
"at most the picture's own size per side" guard as a warning in the Expand UI, and the re-roll
gesture (Expand on a selection that overlaps the old edge).

### 2.5 Tiled generative upscale with a restore prompt and per-tile colour matching

*Tiled Upscale*: resize to about 2 MP, both sides to multiples of 32, the picture as image 1,
the prompt "Restore the photo: remove blur, noise and compression artifacts and recover sharp,
natural detail, keeping the same composition, people and colors." (or the old-print variant),
25 steps CFG 1 (8 with Viggle Turbo), **tiles of about 2 MP sharing a strip and fading into
each other**, **each tile matched to the original's colours before joining** ("Keep colors",
so grain does not push saturation up), a small picture redrawn at up to 8× its own megapixels
then enlarged the plain way; the Restore LoRA keeps the redraw in place (0.3 px) and stops the
invented grit; blurry inputs at strength 0.5. The Texture-Fix VAE for 2.1 (madebyollin) removes
a fine grid visible at 400 %. The FLUX 3 variant: SeedVR2 3B (Apache-2.0, ByteDance; "twice as
big on each side and sharper, 3–4 s on a 5090") first, then the tiles; for blurry photos the
whole picture redrawn first.

For us this is Phase 5's **Generative Upscale**, specified: a `generate.upscale` command that
tiles the picture (2 MP tiles, a shared strip, feathered join), runs the edit route per tile
with the restore prompt, matches each tile's colour statistics to the source before the join,
and resamples back; SeedVR2 as a permissive, non-redrawing "Enlarge 2×" candidate for
`models.md`. The LoRA is a BYOM extra on the 2.1 route when LoRA loading exists.

### 2.6 Prompting rules that moved his numbers

- **Name each piece per reference; call pictures by number.** "The outfit from image 2" alone
  changed nothing in 7 of 18 swaps; with each piece named all 18 went through. On FLUX 3,
  pieces named without their image numbers changed the socks too in 4 of 4 runs; with the
  numbers 8 of 8 stayed. Our fill-from-reference rewriter names the thing "exactly as the
  reference shows it"; it should also number the references in the prompt it writes and name
  each piece when more than one is asked for.
- **Say what stays.** "Keep her face, hair, pose and the background exactly the same." moved a
  FLUX 3 edit about 1 px; "make it snow" without it turned night into day. Our edit prompt
  wrapper has the "change nothing else" clause; keep it, and add the explicit list of what
  stays when the rewriter can name it from the picture.
- **Ask for the change, not the whole picture.** ComfyUI's own FLUX 3 template suggests a full
  description; that redrew the picture up to 202 px off.
- **The fit check before a swap**: a vision model looks at the picture: no legs in view → only
  the top half goes on; no feet → no shoes. "Before I added it, about half of my close-up
  swaps came out wrong." A rewriter rule for our fill-from-reference when the reference is a
  garment.
- **Removal of a person: say what they hold goes too**, or a cup or a phone stays behind
  (our rewriter names one object; a second clause for held things is a measurable change).
- **Picture sizes**: the edited picture at 2 MP, references at 1 MP (0.5 MP on FLUX 3);
  swaps "came out best that way".

### 2.7 Speed on the 2.1 route

- The **official Qwen-Image-2.1 Turbo checkpoint** (`qwen_image_2.1_turbo_int8_convrot`, the
  same 7.26 GB) runs 8 steps with **the step list that ships with the checkpoint** ("a plain
  KSampler set to 8 steps came out gritty"): 4 s for a 2 MP picture, 9 s for an edit, on a 5090.
  Not to stack with the Viggle LoRA ("leathery skin and grit"). Same architecture as 2.1: the
  native engine would load it as a second weight set with its own sigma list.
- **Viggle Turbo v0.3** (6-step LoRA, rank 128, 680 MB, research licence) on the base 2.1: a
  run in about half the time; "a little softer"; plain fabric "a little grainy".
- The **auto prompt uses the text encoder the workflow already loads** (Qwen3-VL 8B for 2.1,
  4B for Krea 2 and FLUX 3): no second model. For the native route this is the lever against
  the re-staging we measured today (`benchmarks.md` §5i): the rewriter and the judge on the
  resident encoder, which needs a generation loop (the 8B repack carries `lm_head`,
  `MODELS-ON-DISK.md`).

### 2.8 LaMa for small marks (permissive, tiny, fast)

The watermark remover is SAM 3 ("watermark", confidence 0.29) → Mask Refine (grow 8 px,
soften) → **LaMa** (`big-lama.pt`, 206 MB, Apache-2.0, advimman/lama) frame by frame: a
124-frame clip in 42 s. LaMa is a convolutional inpainter: no prompt, no diffusion, about
50 ms a frame, and it does what Spot Healing is for (dust, wires, small blemishes, watermarks)
better than PatchMatch and a hundred times faster than a diffusion fill. For PhotoshopEX: a
**`heal/lama` route** in the native worker (ONNX through the `ort` runtime it already has,
BYOM: the user fetches an ONNX export such as Carve's) as the first tier of Spot Healing and
the Remove tool for small holes, the diffusion fill for big ones. Permissive, so it may be a
default.

### 2.9 Interaction patterns

Click to mask with hover preview and "the next thing it could mean" arrows (our Object
Selection tool has click and box; the cycle of candidates is still open from the first note);
**Grow** under the picture ("a little extra is better than too little"); a **before/after
slider** on every result; **Show Text** (the prompt the model got, which we store as
`enhanced` on the layer and could show in the Generations library); a **New seed** button;
LoRA rows with switches and strengths that the workflow locks to its own switches; "the run
stops with a clear message naming the missing file"; ratio buttons and orange handles for the
pad; blue crop handles on every loader; a Denoise box on inpaints (1 replaces the area, 0.7
keeps the old shape, 0.5 stays close: a `denoise` parameter our fill lacks).

### 2.10 Models to add to `models.md`

| Model | Licence | Why |
|---|---|---|
| Qwen-Image-2.1 **Turbo** (official, Comfy-Org INT8) | Qwen Research | 8 built-in steps, 4 s a picture, 9 s an edit: the fast native route for people who opt in |
| **FLUX 3 Image Klein** (Black Forest Labs, 2026-10, INT8 5.8 GB + Qwen3-VL 4B + FLUX.2 VAE) | FLUX Kommunity (free use for individuals and companies under US$5M revenue; outputs under its §2) | an edit model at 6–8 s a run, behind a licence click on Hugging Face; the second edit family candidate |
| **SeedVR2 3B** (ByteDance, INT8 3.46 GB + its VAE) | Apache-2.0 | a 2× enlarge-and-sharpen in 3–4 s that does not redraw: the permissive half of Generative Upscale |
| **LaMa** (advimman) | Apache-2.0 | §2.8 |
| FIBO 2.0 (Bria) | CC BY-NC 4.0 | 4-step edits in 3 s; non-commercial, so BYOM only, noted for the comparison |
| MOSS-Image (OpenMOSS) | (check) | text rendering from quoted words, layout boxes, "Change words"; a later Type-tool item |
| The BFS body/head swap LoRAs (Alissonerdx) | MIT | the only MIT LoRAs in the catalogue; person swap on 2.1 |

## 3. What PhotoshopEX already has (so nothing is taken twice)

SAM 3.1 natively (click, box, text, refine edge, grow/fill holes/soften); 2.1 edit, fill,
expand and matte natively and resident with the prefix cache; the grey hole (`erase`) and the
smoothed hole (`prefill`); fill from a reference with the side-by-side rewriter; the removal
pipeline (naming, whole-object take, the adapter, the judge); Expand with the placed reference
(2.1) and `expandMargin` (2511); the Generations library with the prompt as sent; Best of two;
`purgeModels`; the parity harness. Not yet: LoRA loading in the native engine, the tone match,
the candidate cycle, Generative Upscale, Harmonize, a denoise strength on fills.

## 4. Taken, in order (status in the dev log)

1. **Measure the 2.1 route on the removal sets** (today, `native21-run.ps1`): the native fill,
   one candidate, against the 2511 one-candidate run; then with the whole-picture edit and the
   differential composite (§2.2) if the fill falls short.
2. **Realign to source** (§2.1) in the whole-picture composite: pure Rust, measurable on the
   edit suite's PSNR outside the edit.
3. **Remove with its shadow** (§2.2): the change band on the whole-picture route; the
   rewriter's shadow clause on the crop route.
4. **Remove to its own layer** (§2.3): one command, two layers.
5. **The rewriter and the judge on the resident encoder** (§2.7): the generation loop in the
   native text encoder; removes ComfyUI from the native dab and the re-staging with it.
6. **LaMa as Spot Healing's first tier** (§2.8).
7. **Generative Upscale** (§2.5): tiles, the restore prompt, the colour match, SeedVR2 later.
8. **2.1 Turbo as a second weight set** (§2.7) and **LoRA loading** (Consistency, Outpaint,
   Restore as BYOM rows on the 2.1 route); the tone match at the seam; Straighten and fill.
9. The prompting rules (§2.6) as rewriter changes, each A/B'd on the bench sets.
