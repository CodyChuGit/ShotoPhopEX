# AI integration proposal: everything ComfyUI and a local LLM can add to PhotoCraft

Written 2026-10-08. This is the catalogue of integrations worth building, grouped by the kind of
model that powers them, with a value/effort/licence call on each and a recommended order. It
extends [`roadmap.md`](roadmap.md) (which schedules the first slice) and
[`architecture.md`](architecture.md) (which fixes how any of it plugs in). Facts about the
codebase are cited to [`codebase-orientation.md`](codebase-orientation.md).

## 0. Why the "Firefly inside the editor" ask is the right one

Adobe's generative features are workflows over three model types: a diffusion model that makes
pixels, perception models that turn pixels into masks and maps, and a language model that turns
words into actions. PhotoCraft already has the editor half of each workflow:

| Firefly needs | PhotoCraft already has | Where |
|---|---|---|
| a masked region to fill | selections as coverage surfaces, layer masks, feathering | `Document.selection`, `LayerMask` |
| a place to put the result without harming the original | layers, groups, smart objects with recorded filters, one-undo-step edits | `Session::edit`, `SmartFilter` |
| long work with progress and cancel | the job system (worker thread, `JobCtx`, Esc to cancel, locked document) | `crates/engine/src/jobs.rs` |
| a way for an assistant to act | 500+ commands dispatchable by id from the UI, CLI, a JSON control channel and an MCP server; `document.inspect` and `doc_render_preview` so an agent can see what it did | `commands.rs`, `crates/automation` |
| somewhere to configure it | Edit › Preferences › AI Integrations… (already in the menu) | `prefs.rs` `Integrations` |
| a place for the result's recipe | layer metadata, `.pcraft` manifest, PSD passthrough blocks | `crates/format`, `crates/io` |

The missing piece is the model side, and ComfyUI supplies all three kinds: diffusion (Krea 2,
Qwen-Image), perception (SAM 2, BiRefNet, depth, pose, OCR nodes), and even language (VLM nodes),
behind one local HTTP + WebSocket API. A local LLM runtime (Ollama, LM Studio, llama.cpp) adds the
chat-and-tools half for an assistant. So yes: the ask makes sense, and it is cheaper here than in
most editors because of the command registry and the job system. The constraints that shape it
are licences (some of the best models are research-only), ComfyUI's node names drifting between
releases, VRAM juggling between models (ComfyUI manages it), and the browser build (no server to
talk to; commands disable themselves there).

## 1. Generative pixels (diffusion models through ComfyUI)

The user-facing features. "Default model" means the permissive one the picker pre-selects; the
licence classes are in [`models.md`](models.md).

| # | Feature | What the user does | Pipeline | Default / best model | Value | Effort | Phase |
|---|---|---|---|---|---|---|---|
| G1 | **Generative Fill** | select, type, Generate, pick a variation | composite crop + mask → inpaint-conditioned edit → masked layer | Qwen-Image-Edit-2511 / Qwen-Image-2.1 | very high | M (Phase 1 has the plumbing) | 1–2 |
| G2 | **Generative Expand** | enlarge canvas or crop outward, Generate | pad + mask the new area → G1 | same | high | S after G1 | 3 |
| G3 | **Generate Image** | prompt → new document or layer; optional reference | text-to-image (+ style reference LoRA / multi-ref) | Z-Image Turbo / Krea 2 Turbo | high | S | 3 |
| G4 | **Instruction Edit** ("make it night", "turn the jacket red") | prompt on a layer or the composite, optional mask | image-to-image edit model | Qwen-Image-Edit-2511 / Qwen-Image-2.1 | very high | S after G1 | 3 |
| G5 | **Remove object / Distraction removal** | select, Remove (no prompt) | G4 with a built-in "remove" instruction; falls back to classical `edit.contentAwareFill` when the server is down | edit model | high | S | 2–3 |
| G6 | **Remove / Generate Background** | one click | matte (BiRefNet or Qwen-Image-2.1 RGBA) → layer mask; then G1 on the inverse for a new background | BiRefNet / Qwen-Image-2.1 | high | S–M | 3 |
| G7 | **Harmonize / Relight** | pasted layer + Harmonize | edit model with "match the lighting and colour of the scene"; IC-Light-class relighting as an alternative node graph | edit model | medium-high | M | 5 |
| G8 | **Generate Similar / Variations** | pick a result, "more like this" | same prompt, seed family; image reference | any | medium | S | 2 |
| G9 | **Style / composition reference** | drop a reference image | Krea 2 style-reference LoRA, Qwen-Image-2.1 up to 10 refs, ControlNet (canny/depth/pose) for composition | Krea 2 / Qwen 2.1 | medium-high | M | 4 |
| G10 | **Generative Upscale** | Image › Generative Upscale 2×/4× | upscale-model node (SeedVR2, ESRGAN family) with tiling | licence-checked upscaler | medium-high | S | 5 |
| G11 | **Text effects** | type layer + "chrome", "moss", "neon" | type layer rasterised as the mask; G1 with the mask as a shape constraint (depth/canny ControlNet of the glyphs) | edit model + ControlNet | medium | M | 5 |
| G12 | **Sky replacement (generative)** | Edit › Sky Replacement (upstream wires the menu; the classical version exists) | sky matte → G1 on the sky with a prompt or a reference sky | matte + edit model | medium | M | 5 |
| G13 | **Neural-filter equivalents** | Filter › Neural: colourize, photo restoration, JPEG artefact removal, skin smoothing, super zoom, depth-blur, landscape mixer, style transfer | one ComfyUI graph each; many are small single-purpose models | mixed (verify each) | medium (Photoshop M12 lists them) | S each after infra | 5 |
| G14 | **Generative smart objects** | regenerate a layer from its stored prompt; smart filters on top | `SmartSource` variant holding the request; re-run on demand | any | medium | M | 5 |
| G15 | **Batch generation** | File › Automate with generative steps; CLI `batch`; droplets | commands in actions (free once G1–G6 exist) | any | medium | XS | 3 |
| G16 | **Train a style / character** (LoRA) | pick 10–30 images, name it, Train | ComfyUI training nodes or ai-toolkit on Krea 2 Raw / Qwen-Image-2512 (Apache); result lands in the picker | Krea 2 Raw / Qwen-Image-2512 | medium (power users) | L | 6 |
| G17 | **Live preview painting** | paint a rough shape, see it rendered as you go (Krita's live mode) | fast model (Z-Image Turbo, few steps, low res) re-run on each stroke end | Z-Image Turbo | medium (fun, demo-worthy) | M | 6 |

## 2. Perception (vision models through ComfyUI, no generation)

These feed PhotoCraft's existing tools rather than creating pixels, so they are cheap to integrate
and have no licence drama when the model is Apache/MIT. Each maps to a selection, mask, channel
or map the engine already understands.

| # | Feature | Model class | Lands as | Why it matters | Effort |
|---|---|---|---|---|---|
| P1 | **Select by text** ("select the dog", "the red car", `eye:2`) | **SAM 3.1** (open-vocabulary, text + point/box prompts; native in ComfyUI) | a selection (then Generative Fill, or anything else) | the single biggest UX win after Fill: no lasso. See §2.1 | S–M |
| P2 | **Select Subject / Sky / Hair (ML)** | SAM 3.1 for instances ("person", "sky"); BiRefNet for soft hair/glass mattes | selection or layer mask; swaps the matte source of `cutout_cmds.rs` | upstream's classical versions are weak on hair and glass | S |
| P3 | **Depth map** | Depth Anything / Marigold-class | an alpha channel or a layer; drives Lens Blur (depth-based), fog, relighting, parallax export | upstream lists depth-blur under Neural Filters | S |
| P4 | **Normal / edge maps** | normal estimators, canny/HED | channels; inputs for ControlNet in G9/G11 | composition control | S |
| P5 | **Face and body landmarks** | face landmark / pose models | guides or paths; **Face-Aware Liquify** (upstream's roadmap says it "needs a landmark model") | closes an upstream gap | M |
| P6 | **OCR to type layers** | OCR model (PaddleOCR / Florence-2 OCR) | editable type layers positioned on the canvas | signage/poster workflows; also feeds the assistant | M |
| P7 | **Image → prompt (interrogation)** | VLM captioner (Qwen3-VL / Florence-2) | fills the prompt box; stored on the layer | "make another one like this" | XS once an LLM/VLM path exists |
| P8 | **Auto crop / composition suggestions** | saliency + VLM | crop presets offered in the Crop tool | nice-to-have | S |
| P9 | **Smart tagging and metadata** | VLM | XMP keywords; alt text for export | accessibility, LightCraft hand-off | S |

### 2.1 Select by text with SAM 3.1 (design sketch; `select.byText` and `select.subjectML` shipped 2026-10-09)

SAM 3.1 (Meta, 2026-03-27) segments every instance of a short text concept, optionally refined by
points or boxes, and ComfyUI runs it natively from one 1.75 GB checkpoint
(`sam3.1_multiplex_fp16.safetensors`; templates under Utility). Licence: the SAM License, which
allows commercial use without thresholds but bans military/weapons uses and requires passing the
licence on ([`models.md`](models.md) has the summary). It is the right first perception
integration because it needs no diffusion model, answers in well under a second on the 5090, and
upgrades four existing paths at once.

Commands (engine module `select_ml_cmds.rs`, same crate plumbing as `generate.*`):

| Id | Params | Result |
|---|---|---|
| `select.byText` | `{"prompt":str (≤32 tokens, comma-separated terms, "term:N" caps),"instance":int?=all,"mode":"new|add|subtract|intersect"="new","layer":id?=composite,"feather":px=0}` | the selection; returns `{instances:[{index,bbox,score}]}` so the UI and agents can pick one |
| `select.subjectML` | `{"what":"subject|sky|person|hair|…"="subject","refine":bool=true}` | selection; `refine` runs a BiRefNet-class matte inside SAM's mask for soft edges |
| `select.byPoint` (Object Selection tool, ML mode) | `{"points":[[x,y,label]],"box":[x0,y0,x1,y1]?}` | selection from SAM's point/box prompts |

Pipeline: composite (or the named layer) → PNG upload → SAM 3.1 template with the prompt →
instance masks back as PNG → `photocraft_algo::selection::mask_to_surface` → `Document.selection`
(combined with the current selection per `mode`) in one undo step, as a background job
(`jobs::edit_job`) so the UI stays responsive and Esc cancels. Masks come back at the request's
resolution; the crop-with-margin logic from Generative Fill is reused when a layer is targeted.

UI: a prompt field in the Select menu (**Select › Select by Text…**) and in the Object Selection
tool's options bar (ML mode); the contextual task bar shows the instance strip ("dog 1 of 3")
with next/previous; results feed straight into Generative Fill, layer masks and Quick Mask.
Falls back to the classical `select.subject` when the server is down, with a status message.

What it upgrades: Select Subject (P2), Remove Background's matte (G6), the object masks that
make Remove (G5) and Fill (G1) one click, and the assistant's "select the lamp" step (L1).
Video tracking (Object Multiplex) is irrelevant to PhotoCraft but matters for FilmCraft.

## 3. Language (a local LLM / VLM)

The LLM is not a model inside ComfyUI; it is an **OpenAI-compatible chat endpoint** with tool
calling and optional image input. Local-first choices: Ollama, LM Studio and llama.cpp's server
all expose that API and run Qwen3-VL, Gemma 3 and Llama vision models on the 5090. Cloud
providers can implement the same trait later, off by default, because the project promise is
local and open. ComfyUI's own VLM nodes can serve P7/P9 when no chat runtime is installed.

| # | Feature | How it works | Value | Effort |
|---|---|---|---|---|
| L1 | **Assistant panel** ("remove the lamp, warm it up, add a title in Georgia") | an in-app agent loop: the LLM gets the command registry as tools (`command_list`, `command_run`, `document.inspect`, `doc_render_preview` as an image for VLMs) and the same `authorize` gate MCP clients get; every step is a real command, so it is undoable and recorded as an action | very high: Photoshop's AI Assistant is in beta, and PhotoCraft's registry makes ours more complete on day one | M (the tools exist; the loop, the panel, the safety gate and the UX are new) |
| L2 | **Prompt enhancement** | rewrite a terse prompt into the model's preferred style (Krea 2's official workflow already has a `prompt_enhance` step with an LLM) | medium | XS |
| L3 | **Image → prompt** | VLM captions the layer/composite into the prompt box (P7) | medium | XS |
| L4 | **Explain / critique** ("what would make this poster better?") | VLM on the composite, answers with suggestions that are *commands the assistant can run* | medium | S after L1 |
| L5 | **Auto-name layers, groups and documents** | VLM on each layer's thumbnail | small but constant delight | XS after L1 |
| L6 | **Action authoring** ("make an action that resizes to 2048 and sharpens") | LLM emits an action (`Vec<(id, params)>`) validated against the registry; saved to the Actions panel | medium (power users) | S |
| L7 | **Help that knows the app** ("how do I make a clipping mask?") | LLM with `command_list` and the menu catalogue; answers point at the real menu item and can run it | medium | S |
| L8 | **Agent mode for external agents** | nothing to build: `photocraft-cli mcp` already lets Claude Code, Codex or agy drive the app; the proposal is to document and demo it as a feature | high for the "every feature drivable by agents" goal | docs only |

## 4. Infrastructure that all of the above needs

| # | Item | Notes |
|---|---|---|
| I1 | `photocraft-genai` crate: `GenerativeBackend` + ComfyUI client, workflow templates, request/response pixel conversion | Phase 1; designed in `architecture.md` |
| I2 | `LlmBackend` trait + OpenAI-compatible client (chat, tools, images) | same crate or a sibling `photocraft-assistant`; local endpoints first |
| I3 | Model catalogue as data (`models.toml`): task, licence class, files + sha256, VRAM class, template | Phase 4; the picker and the assistant both read it |
| I4 | Health and resources: server version, VRAM, queue, loaded models, `/free`; "Test connection" in preferences | Phase 2 |
| I5 | Downloads with checksums into the ComfyUI folders; template import (user's API-format workflow with placeholders) | Phase 4 |
| I6 | Generative layer metadata (prompt, seed, model, template hash, server version) in `.pcraft` and as a PSD passthrough block; **C2PA / provenance** when upstream lands it (roadmap M11) | Phase 2 and later |
| I7 | Safety: loopback-only default, research-model opt-in, community-licence notice, content-filter hook, no telemetry | Phase 1–2 |
| I8 | Fake servers for tests (ComfyUI and LLM), one ignored live test each | Phase 1 |

## 5. Recommended order

The roadmap's phases stay as they are; this list says which items go into them.

1. **Phase 1** (plumbing): I1, I7, I8, G1 headless. Prove the round trip on the CLI.
2. **Phase 2** (first user-visible value): G1 in the app, G8, I4, picker and preferences.
3. **Phase 3** (the Firefly core): G2, G3, G4, G5, G6, G15. After this PhotoCraft matches
   Photoshop's generative menu for everyday use.
4. **Phase 3.5, new**: **P1 Select by text** and **P2 ML mattes**. Small effort, large effect,
   and they make G1/G5/G6 feel like Photoshop's.
5. **Phase 4**: I3, I5, G9 (references, ControlNet), P3/P4 (depth and edge maps feed G9).
6. **Phase 4.5, new**: **L1 Assistant panel** with I2, then L2/L3/L5 as cheap follow-ups. This is
   the feature that uses PhotoCraft's "everything is a command" design to its fullest.
7. **Phase 5**: G7, G10, G11, G12, G13 (neural-filter equivalents), G14, P5 (Face-Aware Liquify),
   P6, L4, L6, L7.
8. **Phase 6**: G16 (training), G17 (live painting), native inference backend.

## 6. Decisions to take before Phase 1 code

| Decision | Recommendation |
|---|---|
| Default editor model | Qwen-Image-Edit-2511 (Apache-2.0); Qwen-Image-2.1 behind the research-model opt-in |
| Default generator | Z-Image Turbo (Apache-2.0) as the permissive default; Krea 2 Turbo offered with its community-licence notice |
| "Krea 2 Edit" | experimental lane only (community LoRA + custom nodes), not a default |
| LLM runtime | OpenAI-compatible endpoint, Ollama as the documented default; no provider lock-in; cloud off by default |
| Where the assistant runs | in-process against the live `Session` on the UI thread, through the same gate as MCP; external agents keep using MCP |
| Result placement | always a new masked layer (Photoshop's model); overwrite-in-place never; smart-object recording later (G14) |
| Variations | run as one job; land as hidden siblings in a group; picking toggles visibility (undoable, scriptable) |
| Web build | commands compile and disable themselves; no generative features on the web until a fetch-based client exists |
