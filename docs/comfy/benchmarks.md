# Generative benchmarks (comfy-photocraft)

Measured 2026-10-09 with a local bench script (`bench-fill.ps1`, kept out of the repository) on the development PC:
RTX 5090 (32 GB, driver 617.42), ComfyUI 0.39.0 portable, Python 3.13.14, torch 2.14.0+cu130.
Same image (the 1024² lighthouse, `q21-image.png`), a 384×320 selection with a 25 % margin (a
576×512 request), the same prompt. `ms` is `generate.fill`'s own wall time; the breakdown comes
from the command's `timings`: `encode` (PNG), `upload`, `queue` (queued → the server starts),
`sampling` (the server's execution, model loads included), `download` (fetch + decode). All
numbers are milliseconds. The raw CSVs and the output PNGs sit in
`C:\Users\5090\ComfyUI\photocraft-tests\bench\` (not in the repository).

## 1. Fill templates, default server flags, no cache (run 1)

Uploads had unique names, so every run paid the 7 B vision text encoder and the VAE encode;
"warm" = same pixels and seed again. The 2511 base was already resident from earlier work, so
its "cold" is not a disk load.

| Fill | cold | warm | encode | upload | queue | download |
|---|---|---|---|---|---|---|
| 2511 base, 40 steps CFG 4 | 166 703 | 162 129 | 2 | 5–6 | 0–1 | 294–372 |
| 2511 Lightning 8 steps CFG 1 | 37 406 (LoRA load) | 18 597 | 2 | 5–6 | 0–1 | 362–366 |
| 2511 Lightning 4 steps CFG 1 | 12 529 | 16 525 | 2 | 6–21 | 0–1 | 355–366 |
| Qwen-Image-2.1, 25 steps CFG 1 | 8 563 | 4 504 | 2 | 4–6 | 0–1 | 7–229 |

Reading: Qwen-Image-Edit-2511 is a 20 B model working at its 1 MP size whatever the request
(`FluxKontextImageScale` upsamples a 576×512 request to ~1 MP), and CFG 4 means two passes per
step: ~4 s per step, so 40 steps is 2½ minutes. The Lightning LoRA (CFG 1, one pass) makes a
step ~2 s and 8 steps a 16–19 s fill. The 4-step tier's "warm" being slower than its "cold" is
the text-encoder cost landing differently between runs: the encoder is several seconds of every
run in this table. Qwen-Image-2.1 (7 B, CFG 1, KV prefix cache) does the same fill in 4.5 s.
The client's own share is negligible: 2 ms encode, 5 ms upload, under 0.4 s download.

## 2. Fill templates, `--fast fp8_matrix_mult`, content-addressed uploads (run 3)

Uploads are now named by content hash, so ComfyUI's node cache keeps the loader, the text
encoder and the VAE encode for the same pixels. "cold" = a selection shifted by 16 px (new
pixels, encoder runs; the family's model already resident); "warm" = the same pixels with the
next seed (the variation / re-roll case: only sampling runs).

| Fill | cold | warm | sampling (warm) |
|---|---|---|---|
| 2511 base, 40 steps CFG 4 | 158 935 | 153 331 | 153 055 |
| 2511 Lightning 8 steps | 19 284 | 15 960 | 15 913 |
| 2511 Lightning 4 steps | 11 664 | 8 115 | 8 068 |
| Qwen-Image-2.1, 25 steps | 8 027 | 2 614 | 2 576 |

Reading: `fp8_matrix_mult` is worth about 5 % on the `fp8mixed` 2511 file (159 s vs 167 s). The
content-addressed cache is worth the encoder's 3–5 s on every re-roll: a 2.1 variation is 2.6 s,
a 2511 Lightning-4 variation 8 s, a Lightning-8 one 16 s.

Not adopted, measured:

- `--highvram` with 2511 fp8 (20 GB) + its 8.7 GB encoder resident left 2 GB of 32 free; the
  40-step run got slower (192 s) and the Lightning LoRA run thrashed on reload. The two runs
  that measured it also overlapped (two benchmarks on one GPU), so only the direction is
  reliable; the plain dynamic-VRAM default keeps the models resident anyway while they fit.
- lightx2v's `qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors` (the FP8-tensor-core
  layout) was 35 % faster per fill (10.6–12.6 s for Lightning 8) but produced **noise** in the
  masked area with this graph under `fp8_matrix_mult`; removed again. Worth a second look with
  the ComfyUI-specific `_lightning_comfyui_4steps` file or without the flag.
- `--fast fp16_accumulation`: only speeds up fp16 models; ours run bf16, fp8 or int8.

## 3. Quality: placement and seams (Lightning 8, same seeds)

Three cases on the lighthouse: the water/rock edge with seeds 7 and 11 ("a small red wooden
rowing boat floating on the water") and the sky ("a hot air balloon drifting in the evening
sky"). Judged by eye on the output PNGs.

| Variant | seed 7 boat | seed 11 boat | sky | cost |
|---|---|---|---|---|
| plain prompt, hard mask | on the rocks | cut off, visible rectangular seam | good | 16–18 s |
| mask as `image2` + instruction (`fill-guided`) | on the water | on the water | good, flatter palette | 28–51 s (another 1 MP of reference tokens) |
| "Add {prompt} to this image…" wrapper | on the rocks | on the water, no seam | good | same as plain |
| "Edit this image: {prompt}…" wrapper | on the water | on the water | rectangular seam | same as plain |
| **shipped**: "Add…" / imperative wrapper + outward-feathered request mask | on the rocks | on the water, no seam | good, no seam | same as plain |

What shipped: every 2511 fill template wraps the prompt ("Add {prompt} to this image, fitting
the scene's perspective, lighting and surroundings naturally. Change nothing else."; a prompt
that already starts with an imperative verb gets "{prompt}. Fit the result…" instead), and the
request mask is feathered outward so the latent noise mask blends the edge. `fill-guided`
remains in the picker for the cases where placement matters more than time. Qwen-Image-2.1 put
the boat on the water at every seed tried; it is the research-licensed option.

**Edges (later the same day).** With the layer mask equal to the selection, the model's blended
band was cut off at the selection's edge: a hard border wherever the re-rendered pixels differed
from the original in tone (the user's "the borders are harsh"). Measured on the right-side
expand's sky (mean luminance per column, y 200–420): the original stepped from 225 to 228 at
the old edge, with a dark line of 213 right on it. Three changes, in the order they were found:

1. The layer mask is by default the soft request band itself, its ramp **dithered**: per-pixel
   noise (a splitmix64 hash of pixel index and seed), amplitude `0.6 × (1 − |2m − 1|)`, zero
   inside and outside the band and strongest mid-ramp, clamped to 0..1. The transition reads as
   grain, like film, not as a gradient or a line, and 8-bit exports cannot band. Fills feather
   4 % of the longer side (6–48 px, was 2 %), a band of twice that; expands the same 4 %
   (8–80 px, was 8 %): at 8 % the model took the licence to move the horizon inside the band
   (a ghosted double horizon at one seed), at 2 % the old edge still read as a line.
   `"edge":"hard"` restores the exact selection (or exactly the added canvas) for compositing
   work that wants it.
2. The feather ramp now starts at **full** coverage on the edge and falls to nothing at twice
   the radius (ComfyUI's GrowMask + FeatherMask shape). The first version, two box blurs
   `max`ed with the selection, started at half coverage: as the noise mask it kept half of the
   original latent one pixel outside the edge, and as a layer mask it stepped from 1 to 0.5,
   so half the tone difference stayed as a line.
3. Expand's empty canvas is pre-filled by replicating the picture's edge pixels, not mid-grey.
   The model never sees it as content (its reference is the cropped picture and the noise mask
   is full over it), but the VAE's latents beside a wall of grey carried its tone into the edge
   band: that was the dark line of 213.

After the three: 229 → 226 across 160 px, no step and no dip in the profile, nothing visible in
a contrast-stretched crop; the hard variant keeps its step (as asked). Server cost: none, the
request's size and graph are unchanged (the hard/soft variants of one seed hit ComfyUI's cache
and return in 0.1 s); the dither is one pass over the mask. The grey pre-fill was A/B'd again
with the full-coverage ramp (`xab-f08-grey-*`, `xab-f04-grey-*`): still a 2–3 level step right
on the edge at both seeds, so the edge replication stays.

**Consistency of the new side (the thing no edge can hide).** With the user's prompt "more
open sea and evening sky", seeds 6 and 8 painted a crisper, darker sea with a sharp horizon
next to the hazy original: a different photograph, visible through any blend. Same widen,
Lightning 8 unless noted (`bench/xab2-*.png`, `xab3-*.png`):

| Prompt → what the model got | seed 6 | seed 8 | time |
|---|---|---|---|
| "more open sea and evening sky" (the description wrapper) | crisp sea, sharp horizon | sharp horizon, higher than the original's | 13–15 s |
| the same through the **40-step base** (`qwen-edit-2511/expand`, CFG 4) | a second lighthouse in the new area | hazy, consistent | 130 s |
| "Add more open sea…, keeping exactly the same soft haze… as the original photo" | hazy, consistent | a second lighthouse | 14–16 s |
| **empty prompt** ("extend the scene beyond its original edges, continuing it naturally") | hazy, consistent | consistent, a little sharper | 14–18 s |
| "Extend this image…, continuing the scene naturally with more open sea… Match the original's … haze… and do not repeat its objects" | a second lighthouse | consistent | 13 s |
| "Extend this image to show more open sea… continuing the original photo naturally with the same haze, light and colours" | a stub of railing copied | consistent | 13–15 s |

Lessons: the empty prompt is the most faithful continuation, and it is what the dialog and the
docs recommend for "more of the same"; a description gives the model licence to restyle; the
40-step base is no more consistent and ten times slower. The shipped wrappers are unchanged.

**But the table is indicative at best.** Re-running the empty prompt at seed 6 in a steady
server state gave a second lighthouse (`final-expand-empty-s6.png`) where the run above had
none, and the two runs were bit-identical with each other on repeat. Pixel diffs of identical
requests run at different times: 0.00 (seed 8 widen, two repeats of the empty seed 6), 0.58
(the seed-11 fill), 2.19 and 4.53 (seed-6 widens), 21.44 of 765 (the empty seed 6, a second
lighthouse against none). So ComfyUI is deterministic in a steady state and not across
model-load states: the partially offloaded model after a graph switch (next paragraph)
computes slightly differently, and the 8-step Lightning sampler amplifies that into a
different composition at a knife-edge seed. Conclusions about wordings need many seeds in one
server state (a prompt study is listed under §6); what stands is that the model does, at some
seeds, repeat the picture's main object in the new area, whatever the wording.

**Graph switches thrash the server.** Every first Lightning run after the 40-step base, or
after a purge followed by a fill, took 31, 56, 93 and 121 s (2.3–2.5 GB free, torch 63 MB:
ComfyUI loads the newly patched model partially and streams the rest every step), then
13–20 s for the runs after it. (The base's own 130–148 s is not thrash: 40 steps at CFG 4 are
two model passes a step, about 3.6 s each at 0.75 MP against Lightning's single pass at
CFG 1.) **Fixed the same day** by the engine's `run_switching` (§5c): it remembers the model
files of the last run per server and purges before a run that changes them.

## 4. Native-size sampling (shipped)

The official 2511 graph encodes the `FluxKontextImageScale` output, so every request is sampled
at ~1 MP whatever its size. The shipped templates now VAE-encode the uploaded image itself: the
text encoder still sees the 1 MP reference, the sampling latent has the request's own size
(the engine grows the request rectangle to the 16-px grid, caps it at 0.75 MP, see §5, and
floors it at 512 px on the longer side). Same three cases, Lightning 8, same seeds:

| Case | 1 MP latent | native latent | result |
|---|---|---|---|
| edge, seed 7 | 16.2 s | 11.2 s | boat on the water (was on the rocks) |
| edge, seed 11 | 16.5 s | 9.2 s | boat on the water, no seam |
| sky, seed 7 | 16.4 s | 11.0 s | balloon, no seam |
| 120×90 selection (192×160 rect, sent at 512×416) | — | 10.4 s steady, 16.3 s with a model reload | a crisp seagull on the rock |
| 56×70 selection (96×112 rect, sent at 432×512) | — | 10.4 s | a brass porthole on the door |

35–45 % faster, and better: with the latent at the request's size the mask lands exactly
where the selection is, and both boat cases came out on the water.

## 5. Server memory pressure

After several checkpoints had been loaded in one server session (the 2511 base, its LoRA
tiers, Qwen-Image-2.1, the scaled fp8 file), the small-selection fills took 36 and 58 s with
2.5 GB of VRAM free: ComfyUI was staging the 19.6 GB model on every run. `POST /free`
(unload models) brought 30 GB back and the next fills took 16 s (reload included) and 10 s.
That is **Edit › Purge › Generative Models** (`generate.free`) in PhotoCraft. The steady state
with 2511 fp8 is ~1.6–2.5 GB free on a 32 GB card (UNET 19.6 GB + text encoder 7.9 GB + VAE).
In that state a 1 MP latent plus the 1 MP reference did not fit the headroom: Generative
Expand's frame case took 45 s (the log shows the 19.6 GB model re-staged for the run) against
18 s a moment earlier. The request cap is therefore 0.75 MP (768×1024), which keeps the model
resident and costs little: the editing models' own working size is about that.

## 5b. Remove Background and transparent generation (Qwen-Image-2.1, int8)

| Case | time | result |
|---|---|---|
| `generate.removeBackground`, 1024² lighthouse, no prompt | 16.8 s | boat solid, lighthouse a ghost (the model's reading of "the subject") |
| …, prompt "the red boat" | 16.7 s | a perfect 147×46 px boat cutout |
| …, prompt "the lighthouse" | 16.8 s | lighthouse with its rocks; the white tower semi-transparent against the bright sky |
| …, `asSelection` of the same request | 0.14 s | ComfyUI's cache |
| `generate.image` + `transparent`, "a vintage brass compass, product photo", 1024² | 10.8 s | a clean cutout with a real soft alpha |
| the first run after switching from the 2511 models (a purge first) | +~8 s | the 2.1 model and encoder load (6.8 + 8.7 GB) |

The matte is 25 steps of the full model, so it costs a text-to-image. The permissive route
(`template: auto` without the research opt-in) is SAM 3.1 instead, hard-edged but fast:

| Case | time | result |
|---|---|---|
| Remove Background, prompt "the red boat", SAM 3.1 | 1.5 s (checkpoint load included) | the boat, 146×38 px |
| Remove Background, no prompt ("the main subject"), SAM 3.1 | 0.5 s | the lighthouse tower, 111×281 px |
| `select.byPoint` at (350, 665) | 0.3 s | the boat |

A soft matte from a dedicated matting model (BiRefNet) would sit between the two in quality;
it needs a custom node pack the server does not have, so it is not planned for the default.
The detector route softens its mask with the classical edge refinement instead (`refine`).

## 5d. Split into Layers (Qwen-Image-Layered, fp8)

| Case | time | result |
|---|---|---|
| the 1024² lighthouse into 3 layers, 20 steps CFG 2.5 at 640² | 72 s (the 19 GB model loaded in it) | layer 0 a near-white base with the boat's shadow, layer 1 sky and sea with the boat, layer 2 rocks and lighthouse with real soft alpha |

CFG 2.5 means two passes a step, like the 2511 base; the official 640 px working size keeps it
to about a minute. The layers come back resampled to the canvas.

## 5c. Generative Edit and the automatic purge

`generate.edit` sends the whole 1024² lighthouse at 0.75 MP through the official 2511 edit
graph (no noise mask); the result is a layer, masked to the selection when there is one
(`bench/edit-*.png`):

| Case | time | result |
|---|---|---|
| "make the sky dark and stormy with heavy clouds", Lightning 8 | 16.0 s steady (22.6 s with the 2511 reload after a 2.1 run) | a storm with lightning, the lamp lit, the sea rough; the rocks and lighthouse kept |
| "turn the red boat blue" with a 200×120 selection on the boat | 16.1 s | only the boat changed on the canvas (the model re-rendered everything; the mask shows the boat) |
| "make it a sunny day with a blue sky" | 20.2 s after a model switch, 16 s steady | convincing |
| the same storm through the 40-step base (`qwen-edit-2511/edit`) | 145–148 s | two passes a step at CFG 4; the Lightning tier is the default for a reason |

**The purge.** Three edits in one process, Lightning → base → Lightning (`purge-live.ps1`):
16.0 s, 148.5 s (the base's own cost, purged first), **20.2 s** for the Lightning run after
the base. The same return to Lightning without a purge took 121 s earlier in the day. The
engine purges only when the model files change and never on a server's first run; separate
CLI invocations are separate processes, so the CLI only exercises it within one `--cmd` chain.

## 5e. Prompting: what the guides say, and the enhancer

The first user session (dev log, 2026-10-09 evening) turned a line drawing of a face into a plate
of food twice: "make this hyper realistic" gives Qwen-Image-Edit nothing to hold on to when the
picture is a doodle. What the model authors and ComfyUI recommend, checked 2026-10-09:

| Source | What it says | What we do |
|---|---|---|
| Qwen-Image-Edit model card and the `Qwen-Image` repo's `polish_edit_prompt` (Alibaba, 2025–2026) | run the instruction through a VLM that sees the picture and rewrites it: name the objects, one change per instruction, state what stays, quote literal text verbatim | `generate.enhancePrompt`: the same rewrite, on the Krea 2 text encoder (Qwen3-VL 4B) already on the server |
| ComfyUI's shipped Qwen-Image-2.1 "prompt enhancer" subgraph (`comfyui-workflow-templates` 0.11.76) | a 10 KB system prompt for an 8B VLM (`TextGenerate`, temperature 0.3, no thinking) | too long for the 4B model (it echoes or rambles); a 12-line prompt with the same rules works |
| ComfyUI's Krea 2 template | `TextGenerate` on the Krea 2 encoder expands a short idea into a paragraph before `CLIPTextEncode` | `enhance/image`: the same node, our short instruction, 60–120 words |
| Krea 2 prompting guide | detailed natural-language paragraphs, subject first, then materials, lighting, setting, style | the image system prompt orders the paragraph that way and keeps every subject, colour and medium the user gave |
| Qwen-Image-Edit-2511 settings in ComfyUI's templates | 40 steps / CFG 4 base, Lightning LoRA 4 steps / CFG 1 (shift 3), `ModelSamplingAuraFlow 3.1`, `CFGNorm`, `index_timestep_zero` | already what the templates ship (§2, §5c); the 8-step Lightning tier is the default |

**Settings check.** Our 2511 graphs match the official templates node for node (sampler euler /
simple, shift 3.1, CFG norm, the reference-latent method); the Lightning tier is the official
Lightning recipe with 8 steps instead of 4 (visibly cleaner edges at +4 s). Krea 2 Turbo: 8 steps,
CFG 1, `ConditioningZeroOut` negative, CLIPLoader type `krea2`, as shipped. Nothing was left on
the table there; the prompt was the lever.

**The enhancer, live** (`enhance-live.ps1`, the CLI against the server, 5090, ComfyUI 0.39):

| Case | time | the rewriter's sentence |
|---|---|---|
| the doodle, edit, "make this hyper realistic" | 0.9–1.3 s | Turn the doodle of a smiling face into a photograph of a real human face with the same expression and pose, replacing the black outline strokes with real skin, hair, and facial features; keep the white background unchanged. |
| the doodle, edit, "turn this drawing into a hyper realistic portrait that looks similar" | 0.9 s | Make the hand-drawn smiley face look like a hyper realistic portrait of a human face with the same expression and pose, replacing the black outline strokes with real skin, hair, and facial features, keeping the white background unchanged. |
| the doodle, edit, "make it a watercolor painting" | 1.1 s | Turn the hand-drawn smiley face into a watercolor painting of a smiling human face, replacing the black outline strokes with soft watercolor washes and brushstrokes, keeping the simple white background unchanged. |
| the lighthouse, edit, "make the sky stormy" | 1.0 s | Make the sky stormy with dark, swirling clouds and flashes of lightning, keeping the white lighthouse, red boat, and rocky shore unchanged. |
| the lighthouse, edit, "turn it blue" | 0.8 s | Turn the sky and sea blue, keeping the white lighthouse and red boat unchanged. |
| the lighthouse, edit, `put a sign on the lighthouse that says "OPEN"` | 1.1 s | Add a sign reading "OPEN" to the lighthouse's front door, keeping the white lighthouse, red boat, and rocky shoreline unchanged. |
| the lighthouse, fill with the upper part selected, "a hot air balloon" | 1.0 s | Add a hot air balloon floating above the lighthouse, blending with the soft pink sky, while keeping the white lighthouse, red boat, and rocky shoreline unchanged. |
| the same fill, "remove this" | 0.9 s | Remove the white lighthouse from the upper part of the image, keeping the pink sky and rocky shoreline intact. (the limit of the place hint: "this" was read as the lighthouse, not as the selected sky) |
| image idea, "a fox in snow" | 1.9 s | a 110-word paragraph: the fox alert in deep snow, fur, paws, soft diffused daylight, an open field, a photorealistic cool palette |
| image idea, already detailed (a fisherman mending a net, golden hour, 85mm) | 1.8 s | polished, keeping "85mm lens", "golden hour" and the shallow depth of field |

Same seed, same sentence every time (temperature 0.3); the seed changes little. The edit sentences
are what the Qwen guides ask for: the verb first, the subject named from the picture, what stays.

**The chat template.** The first live run answered nothing half the time: of 29 rewrites, 15 came
back empty and 9 began with a stray "assistant". ComfyUI's `TextGenerate` with `thinking: false`
appends an empty `<think>` block after the assistant turn (the Qwen3 convention), which the
non-thinking Qwen3-VL-4B-Instruct answers with an immediate end-of-turn or by restarting the
turn. `probe-enhance5.ps1`, 12 runs per variant: thinking=false 0 clean (7 empty, 5 echoes);
**thinking=true 12 of 12 clean**, with the rules in the user turn or as a real system turn
(`system_prompt`). The templates use thinking=true and the system turn.

**The doodle, through Generative Edit with `enhance: true`** (`bench/enhance-doodle-*.png`,
`doodle-sentence-*.png`): the first wording of the rules ("ends with what must stay unchanged")
made the rewriter write "keeping its simple outline and dot eyes unchanged", and 2511 drew a
loaf of bread with a smiley (as it had from the raw prompt). The sentence for "turn this drawing
into a … portrait" gave a human face with the doodle's lines still drawn over it. Three
hand-written sentences then showed what the model needs: "a photograph of a real human face …
replacing the black outline strokes with real skin, hair and facial features" (A) and "a hyper
realistic portrait photo of a real person … with no drawn lines left" (C) both gave a clean
photographic face in 19–20 s; "look like a hyper-realistic photograph of a real human face"
without the replacement clause (B) gave a face with the lines kept. The rules now say so for a
change of medium, and the rewriter's sentence for "make this hyper realistic" is the A form.

**Cost.** A rewrite is 0.8–1.3 s for a sentence, 1.8–2 s for a paragraph, after a 3 s first load
of the 4B encoder (5 GB staged). An enhanced Krea 2 image took 11.1 s against 6.2 s as typed
(the rewrite, a longer encode and the switch back from 2511). The encoder stays resident: with
2511 (19.6 GB), its own encoder (7.9 GB) and the 4B (5.0 GB) all loaded the card had 1.9 GB free
and 2511 edits ran at 19–25 s instead of 16 s (Krea 2 was resident too); with the enhancer and
2511 alone the steady state is 13.5–14.6 s (§5f).

Through the CLI, end to end (`enhance-doodle-final.ps1`, `bench/enhance-doodle-*-final.png`):
"make this hyper realistic" on the doodle now gives a photographic human face in 18.8 s (the
rewrite included), "turn this drawing into a hyper realistic portrait that looks similar" one in
17.2 s; both layers remember the typed prompt and the sentence the model got.

## 5f. NVFP4 against fp8 on the 5090

Comfy-Org publishes NVFP4 repacks of the 2511 text encoder (`qwen_2.5_vl_7b_nvfp4`, 5.7 GB
against 8.7 GB) and of Krea 2 Turbo (`krea2_turbo_nvfp4`, 7.2 GB against 12.2 GB); Blackwell
GPUs run the 4-bit weights through their own tensor cores (comfy-kitchen 0.2.37's
`scaled_mm_nvfp4`). Measured with the engine's own graphs posted to the server
(`nvfp4-bench.ps1`: a purge, then a cold run and two warm runs with their own seeds; ComfyUI
caches a repeated graph, which made a first attempt report 290 ms "runs"):

| Case (1024², 8 steps) | fp8 | NVFP4 |
|---|---|---|
| 2511 Lightning edit, the encoder swapped | cold 19.6 s, warm 14.1 / 14.0 s | cold 19.8 s, warm 14.1 / 14.2 s |
| Krea 2 Turbo image, the model swapped | cold 11.1 s, warm 7.1 / 7.0 s, 13.4 GB free afterwards | cold 9.2 s, **warm 5.7 / 6.0 s, 18.5 GB free** |
| 2511 edit with the enhancer's 4B encoder resident | 22.9 s with the load, then 13.5 / 13.6 s | 21.1 s, then 14.3 / 14.6 s |

The encoder runs once per prompt, so its precision does not move an edit's time; it saves
3 GB of weights (staged, in RAM and on disk). Krea 2 in NVFP4 is 18 % faster per image and
leaves 5 GB more free. Output: the lighthouse storm is the same picture with either encoder
(`bench/nvfp4-2511-te-*.png`), the fox equivalent (`bench/nvfp4-krea2-*.png`).

**Shipped as a policy, not a switch per file.** Template slots carry an `nvfp4` alternative
(the 2511 templates' encoder, Krea 2's model); Preferences › AI Integrations › **Model
precision** `auto` (the default) loads it when the server's GPU is a Blackwell part
(`is_blackwell` on `/system_stats`'s device name: GeForce RTX 50, RTX PRO Blackwell, B100–B300,
GB200/GB300, GB10) and the server lists the file; `default` never; `nvfp4` always (for
testing). The purge sees the switch (the model set is computed from the prepared request),
`generate.models` reports `device`, `blackwell` and each slot's `nvfp4` / `nvfp4Installed`, and
a caller's own `model` binding is never replaced. On Ampere and Ada there are no NVFP4 kernels
(ComfyUI would dequantize or fail), so `auto` leaves them on the default files. No official
NVFP4 repack of the 2511 diffusion model exists (a community one does); Qwen-Image-2.1 ships
int8 and w4a8 encoders instead, so neither has an alternative yet. Live through the CLI
(`nvfp4-live.ps1`): the server loaded `qwen_2.5_vl_7b_nvfp4` for an edit and
`krea2_turbo_nvfp4` for an image with nothing but the default preference.

## 5g. Reference images and the time estimate (PhotoshopEX, 2026-10-12)

Warm runs through the release CLI, one process, ComfyUI on the 5090 (NVFP4 text encoder by
the `auto` policy): `generate.edit` on the 1024² lighthouse picture (sent at 880×880 = 0.77 MP),
Lightning 8, the same prompt, with 0, 1 and 2 file references (1024² each, scaled to ≤ 1 MP):

| References | Run 1 | Run 2 |
|---|---|---|
| none | 11.9 s | 11.7 s |
| one | 19.9 s | 20.0 s |
| two | 30.1 s | 30.3 s |

Each reference image adds about 9 s (another megapixel of tokens through the encoder and the
reference latents). The templates' `estimatedSeconds` were fitted to these runs and the Generate
panel's fills: the Lightning-8 family is `7.5 s + 5.5 s/MP + 9 s per extra reference` (a fill at
~0.2 MP: 8.7 s measured, 8.6 s estimated; the edits above: 11.7 / 20.7 / 29.7 s estimated). A
reference-driven text to image (`qwen-edit-2511/image-ref`) counts its first reference as the
picture: 12.1 s measured for a 3:2 1K image from one reference, 11.3 s estimated. The 40-step
tier's per-reference cost is scaled by its steps and CFG passes (90 s, not measured).

## 6. Where the time goes, and what is left

- A 2511 fill is sampling-bound: ~1.3 s per Lightning step at a 512-px request, ~2 s at 1 MP
  on the 5090; the encoder 3–5 s when the pixels change; everything on the PhotoCraft side
  under 0.5 s.
- Next levers, in order of expected gain: a working FP8-tensor-core checkpoint (the scaled
  file gave noise here), SageAttention once a wheel exists for this torch build, a 20-step
  middle tier for the base (Comfy's note accepts 20 steps; CFG 4 is still two passes a step),
  TeaCache/EasyCache-style step caching for the 40-step base, detecting the memory-pressure
  state from `timings` to suggest a purge, and a prompt study for Expand over more seeds (§3:
  two seeds showed the wording matters and cannot rank it). The automatic purge on a model-set
  change shipped (§5c).
- On a 24 GB Ampere card (RTX 3090) the same tiers apply with roughly 2–3× the step times and
  no FP8 gain; the text encoder will live in RAM; the Lightning 8-step tier is the one to install.
