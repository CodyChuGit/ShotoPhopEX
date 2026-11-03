# AusBoss: Krea 2 Inpaint (click to mask) v1.3 and the Qwen-Image-2.1 Outpaint LoRA

Read 2026-11-03 at the owner's request ("learn from these … to do select + inpaint and image
extension properly"). Sources: the workflow JSON `krea2InpaintClickTo_v13.json` the owner
attached (Civitai 2971849, v1.3, 2026-10-14), the Civitai API records of 2971849 and 2967375
(the HTML pages refuse automated fetches), and the LoRA files on disk. Behaviour and numbers
only; nothing copied.

## 1. Select + inpaint: "Krea 2 Inpaint | Click to Mask" v1.3

Models: Krea 2 Turbo fp8 scaled, Qwen3-VL 4B fp8 (the Krea 2 encoder), `qwen_image_vae`,
`krea2_anypaint_rank32` (yijunwang2, strength 1), SAM 3.1 multiplex for the click; BiRefNet
only for the Background button. AusBoss nodes 2.11+.

**Mask (Mask by Touch, SAM 3.1).** A click picks the object (it lights up on hover); arrows
step to the next meaning; a second click adds a piece; Paint/Erase by hand; Background picks
everything around the main subject. Node defaults: grow 0, soften 0, smooth 0, fill holes on,
"stop the run" when nothing is found.

**"Cut out the area" (all sizes scale with the picture, `k = min(1, sqrt(w·h) / 1500)`, so the
values below are at 2.25 MP and up):**

| step | value |
|---|---|
| grow the mask | `max(1, round(12·k))` px, then blur `max(0.5, 4·k)` px, fill holes (≤ 2 px) |
| crop around the mask | context factor 2 (twice the mask's box) plus up to `max(32, round(192·k))` px, scaled to about 1.6 MP, sides on 16, kept inside the picture |
| blend band (stitch) | `max(4, round(24·k))` px |
| noise mask | the cropped mask grown a further 24 px (tapered corners) |
| prompt-writer copy | the crop with the grown area flat `#808080` |

**"Paint with Krea 2".** `Krea 2 Encode` gets the prompt, the **plain crop** (not grey: v1.3's
fix: "Krea 2 is now shown the picked area in your picture's own colour, the way its inpaint
LoRA was trained"; v1.1's grey left flat grey patches under denoise 0.85) and the picked area,
VLM reference on. `Krea 2 Outpaint Model Patch`: KV cache on, placement "whole canvas" (the
reference registered over the whole target grid). VAE-encode the crop, `SetLatentNoiseMask`
with the noise mask, KSampler 8 steps, CFG 1, `er_sde` / `simple`, denoise from the front
(default 1). Stitch back: classic seam, colour match 0.5, no halo fix.

**Auto prompt (on by default).** Qwen3-VL 4B through `TextGenerate` (thinking on, temperature
0.7, top-k 64, top-p 0.95, min-p 0.05, repetition 1.05, 180 tokens max) reads the grey-filled
crop. Its instruction, in substance: write one 40–80 word text-to-image prompt describing the
picture as it will look finished (subject, setting, materials, lighting, camera style); never
call the grey area grey, blank or masked, describe it as part of the scene; then the user's
words for what goes there. Off: the words go to Krea 2 as typed.

**Card lessons.** Say what replaces the area, not only what to remove (else a flat patch).
One change per run (two separate areas: one may not come out). Ask for something about the
same size as what was there. Denoise 1 replaces; ~0.7 keeps the old shape; ≤ 0.5 stays close.
Keep bare skin out of the pick (repainted skin shifts tone). Nothing more than ~100 px from the
mask moved in his tests. 16–20 s a run on a 5090 in ComfyUI.

**Where we stand.** The engine half (registered reference, masked latent, denoise) is on
genai-native `krea2-anypaint-fill` (`krea2-turbo/fill`, the coordinator session's). What the
app still lacks against this workflow: the scaled grow/soften/blend/context crop, the 24 px
noise-mask growth separate from the layer mask, the auto-prompt writer on the grey copy, and
the 0.5 colour match in the stitch. The app's fill already has grow/soften/fill holes on the
Object Selection bar and SAM 3.1 click; "next meaning" arrows are not there yet.

## 2. Image extension: "Qwen Image 2.1 Outpaint LoRA" (Civitai 2967375)

Files (both installed in `models/loras` since 2026-10-26, checked again 2026-11-03):
`qwen-image-2.1-outpaint-v2.safetensors` (159,436,576 bytes, SHA-256 `3611a5cc…eca28`, for 1–2 MP
everyday extensions) and `qwen-image-2.1-outpaint.safetensors` (v1, `201ab351…7eb177`, any size,
big zoom-outs at ~1 MP). Rank 32, ai-toolkit, trained on Comfy-Org's INT8 convrot 2.1.

**Recipe.** Pad flat `#808080`, the picture exactly where it stays, sides on 32, ~1–2 MP. The
padded canvas is `image_1` of `TextEncodeQwenImage21` at resolution 0 (the target takes its
size); sample the encoder's latent; **no** `SetLatentNoiseMask` (on 2.1 it draws a rectangle at
the seam; the LoRA keeps the picture by itself). 25 steps, CFG 1, euler / simple, denoise 1,
LoRA 1. Prompt: the trigger first, word for word ("Outpaint the image: replace the solid gray
areas with a seamless continuation of the scene, keeping the existing picture unchanged."),
then optionally `Scene:` and a description (75 % of training captions had a Qwen3-VL-8B one).
Decode (2.1's VAE is RGBA), split the alpha off, paste the original pixels back. Tilted
pictures: rotate + pad with feather 0 (a fade into grey becomes a dark band).

**His numbers.** Plain 2.1 moved or rescaled the picture in 13 of 16 renders (up to 284 px);
with v1 or v2, 32 of 32 in place. Kept-area PSNR 16.4 → 34.6 dB (subtle crops), 25.5 → 34.0
(big). Grey left unfilled 9.1 % → 0.9 %. Tilt 17°: plain moved 12 of 12, LoRA 24 of 24 in place.
Weak spots: a flat backdrop can get a soft vertical smudge on the new side; the kept picture
under ~15 % of the canvas varies by seed (two passes, or v1); text in new areas is not legible.

**What landed (2026-11-03).** genai-native `qwen21-outpaint-lora` serves `qwen-2.1/outpaint`
(the canvas as image 1, the LoRA added first, listed when the file is in `loras/`). The app's
`qwen-2.1/outpaint` template (`greyCanvas`): the request is the whole grown canvas prefilled
flat grey (not the edge-replicating prefill), no mask, sides on 32, the trigger + `Scene:`
around the prompt, the result realigned to the picture; `auto` expands with it when research
models are allowed, the worker runs, and the LoRA is in the models folder (else 2511 as
before). The two-stage blank expand (continue a cut-off subject, then the backdrop) is kept.
