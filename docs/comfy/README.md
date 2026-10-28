# Comfy PhotoCraft

**A Firefly-class generative toolset for PhotoCraft, fully local and fully open source.**

Comfy PhotoCraft is a fork of [storytold/photocraft](https://github.com/storytold/photocraft) (the
clean-room, pure-Rust Photoshop reimplementation) whose purpose is to close the one gap upstream
has deliberately deferred: **AI / generative features** (`docs/roadmap.md` grades them "~0%,
deferred by decision (#41)"). The generative engine is a locally running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) server driving open-weight models such as
**Krea 2**, **Qwen-Image-2.1 / Qwen-Image-Edit** and friends, so that Generative Fill, Generative
Expand, text-to-image, instruction editing and background removal run on the user's own GPU with
no account, no cloud and no credits.

This folder is the fork's own documentation. Everything upstream wrote still applies: read
[`AGENTS.md`](../../AGENTS.md) first, then the upstream `docs/`, then this folder.

| Document | What it is for |
|---|---|
| [`codebase-orientation.md`](codebase-orientation.md) | The parts of the ~250k-line workspace a generative feature touches: command anatomy, pixel access, jobs, preferences, UI hooks, automation, tests. Cited to `file:line`. |
| [`architecture.md`](architecture.md) | The design: the `photocraft-genai` crate, the ComfyUI client, workflow templates, the `generate.*` commands, the UI, and how it stays inside upstream's rules. |
| [`models.md`](models.md) | The model catalogue: what each open model does, its licence, VRAM, ComfyUI support and file names. Includes the Firefly feature → model mapping. |
| [`comfyui-setup.md`](comfyui-setup.md) | Running ComfyUI locally on this machine (RTX 5090), the model files to download, and an API primer (HTTP + WebSocket) with exact field names. |
| [`roadmap.md`](roadmap.md) | Phases 0–6 with a definition of done each, ordered so every phase ships something usable. |
| [`ai-integration-proposal.md`](ai-integration-proposal.md) | The full catalogue of what ComfyUI (generation, perception) and a local LLM (assistant, prompts, naming) can add, with value, effort, licence and a recommended order. |
| [`plan-artcraft-experience.md`](plan-artcraft-experience.md) | The executable hand-off plan for the ArtCraft experience in PhotoshopEX: five phases (Generations library and panel, Generate panel with references, model picker, Variations strip, hub/alerts/export/Angles) with data shapes, command specs, code anchors by file and line, tests, docs and an acceptance walkthrough. Start here to build it. |
| [`artcraft-study.md`](artcraft-study.md) | How ArtCraft (storytold's AI studio) delivers generation as an experience: the shell, the Create pages, the prompt box, the feed, the Task Queue, the lightbox and Library, the 2D editor's History Stack, settings; the principles behind it; and the PhotoshopEX translation (Generate panel, Generations panel, Variations strip) with wireframes and a build order. Credits left out: local compute. |
| [`upstream.md`](upstream.md) | Fork hygiene: syncing with upstream, what to contribute back, brand-licence obligations, naming. |
| [`devlog.md`](devlog.md) | Dated log of what landed and what is open, so the next session (human or agent) can pick up. |

## Vision

Adobe's generative features are the reason many people cannot leave Photoshop. Every one of them
is a *workflow* over a diffusion model: select, describe, generate, pick a variation, keep it on
its own masked layer. PhotoCraft already has the hard parts of that workflow: a document model with
layers and masks, a selection system, a command engine where every action is scriptable, a
background-job system with progress and cancellation, and a thin UI that is generated from data.
What is missing is a backend that turns "prompt + pixels + mask" into pixels, and the handful of
commands and panels that wire it in.

ComfyUI is that backend. It is the de-facto standard runtime for open image models, it gets
day-one support for new models (Krea 2 landed in ComfyUI 0.26.0 the day after its release;
Qwen-Image-2.1 the day of), it runs on the user's own GPU, and it exposes a small, stable HTTP +
WebSocket API. Treating it as an external service keeps PhotoCraft pure Rust (no PyTorch, no CUDA
bindings, no C dependencies) and keeps the browser build intact.

## Principles (in addition to upstream's)

1. **Local and open by default.** No cloud, no accounts. Models the UI offers by default must be
   open weights with a licence that permits what the user is doing; the catalogue says which
   licence applies ([`models.md`](models.md)).
2. **Everything is still a command.** `generate.fill`, `generate.expand`, `generate.image`,
   `generate.edit`, `generate.removeBackground` are engine commands with params docs, `enabled`
   predicates, tests and `panic_hunt` coverage. The UI, CLI, control channel and MCP dispatch them
   like any other command. Agents can drive generative edits headlessly.
3. **Non-destructive.** A generation lands on a new raster layer with a layer mask cut from the
   selection, exactly like Photoshop's generative layers. The original pixels never change. Prompt,
   seed, model and workflow are recorded on the layer so a result can be reproduced or varied.
4. **The backend is a trait.** ComfyUI is the first `GenerativeBackend`. A native inference
   backend (candle/burn) or another server can come later without touching the commands or UI.
5. **Never block, never crash.** Generation runs as a background job (`crates/engine/src/jobs.rs`)
   with progress from ComfyUI's WebSocket and cancellation that interrupts the server. A server
   that is down, slow, misconfigured or returns garbage produces an error the user can act on.
6. **Upstream-compatible.** The fork tracks upstream `main`. Generic improvements go back upstream
   as PRs; the generative crate is designed so it could be upstreamed as a whole once #41 is
   decided.

## Firefly feature parity matrix

The target list is Photoshop 2026 (v27.x) as documented by Adobe and the community in 2026
(sources in [`models.md`](models.md) › Sources). "Phase" refers to [`roadmap.md`](roadmap.md).

| Photoshop feature | What it does | Comfy PhotoCraft plan | Default model (licence) | Phase |
|---|---|---|---|---|
| Generative Fill | Inpaint the selection from a prompt; result on a masked layer; several variations | `generate.fill`: composite + selection mask → edit/inpaint workflow → new layer + mask | Qwen-Image-Edit-2511 (Apache-2.0); Qwen-Image-2.1 (research) | 1–2 |
| Generative Expand | Outpaint when the canvas/crop grows | `generate.expand`: pad composite, mask the new area, same workflow | same as Fill | 3 |
| Generate Image | Text-to-image into a new document/layer, reference image optional | `generate.image` | Krea 2 Turbo (Krea 2 Community); Z-Image Turbo (Apache-2.0) | 3 |
| Generate Background | Replace the background behind the subject | `generate.removeBackground` + `generate.fill` of the inverse | Qwen-Image-2.1 RGBA / BiRefNet + editor | 3 |
| Remove tool / Distraction Removal | Remove an object, fill plausibly | `generate.edit` with "remove …" + mask; classical `edit.contentAwareFill` stays as fallback | Qwen edit models | 2–3 |
| Reference Image / Style reference | Steer a generation with an image | `generate.image` / `generate.edit` with `references[]` | Krea 2 style-reference LoRA; Qwen-Image-2.1 (up to 10 refs) | 4 |
| Harmonize | Match colour, light and shadows of a pasted layer to the scene | `generate.edit` preset ("harmonize") on the layer with the composite as context | Qwen edit models | 5 |
| Generate Similar | More variations of a chosen result | re-run with the stored prompt/seed family | any | 2 |
| Generative Upscale | Enlarge up to 4× with new detail | `generate.upscale` through an upscale-model workflow | SeedVR2 / ESRGAN-family (verify licence) | 5 |
| Partner models in the picker | Choose a model per generation | model picker in the task bar, fed by the catalogue + server `object_info` | all installed | 2 |
| Generative Workspace / AI Assistant | Ideation canvas, conversational panel | out of scope for now; MCP already lets any agent drive the app | — | — |

## How to use (PhotoshopEX)

Everything runs on your own ComfyUI server (Edit › Preferences › AI Integrations › ComfyUI
server) with the models you installed there; the app ships no weights.

- **Window › Generate** (the prompt box in the dock): pick Image, Fill, Edit or Expand, type,
  add reference images to the deck (+, or drop files on the panel), choose a model in the picker
  (Auto names the model it will use; `research` models need Preferences › Allow research-only
  models; `not installed` means the server lacks a file), and press **Generate · ~9 s**
  (Ctrl+Enter). On a short window the panel is one row: mode, model glyph, prompt, Generate.
- **The task bar under a selection**: the same prompt in Fill or Edit mode when the Generate panel
  is not on screen (Preferences › Show generative bar: auto / always / never).
- **Window › Generations**: every run of this and earlier sessions, kept on disk. Running runs
  show their percent and Cancel; failed ones the reason with Recreate; results Place, Generate
  Similar, Save… and Copy prompt on hover. ✓ (Select) picks several for **Save selected** into
  Preferences › Save results dir, named by Save results naming (`{model}_{date}_{seed}`).
- **The Variations strip** at the canvas edge: the document's last five runs as tiles. Click one
  to show it, × to delete it, ✓ to keep it and delete the others, ✦ for more like it.
  Preferences › Variations strip: right, bottom or off.
- **Edit › Generative Angles…**: drag the camera around the subject in the orbit widget (8
  directions, 4 heights, 3 distances) and Generate; needs fal's Multiple Angles LoRA in
  ComfyUI/models/loras.
- **The Home screen and Help › Generative Features…**: one card per feature; an Edit card asks
  for a picture first when none is open.
- **Quality** (the menu beside the model in the Generate panel, or Preferences › AI
  Integrations › Quality): Best runs the full-step models and keeps the 4-bit weights off; Balanced
  takes the 8-step Lightning tier; Fast the 4-step tier. It only changes what Auto picks.
- **The Object Selection tool (W) with SAM 3.1**: rest the pointer on something and the object
  finder tints what a click would select; click to select it, drag a box around an object to find
  it inside, ⇧ to add another object, ⌥ to subtract one. Switch the options bar to **Refine** to
  grow or trim the last object: click a spot it must include, ⌥-click one it must not. **Refine
  edge** softens the detector's edge from the pixels (hair, fur). Select › Subject uses the
  detector too.
- **The Quick Selection tool (W) with SAM 3.1**: paint over an object and the detector selects
  the object under the stroke (a handful of spots along it are the point prompt), not the
  similar-coloured region; another stroke adds, ⌥ subtracts the object under the stroke; **Enhance
  Edge** is the detector's edge refinement. Without a server it is the classical quick selection.
- **Preferences › Alerts**: a short sound when a run finishes (`done`) or also when it fails
  (`doneAndFailed`); the Windows and macOS packages are built with the `alerts` feature.

## Status

- 2026-10-08: Phase 0 (this documentation, the Windows environment, the `comfy-photocraft`
  branch). The release build and test suite were run on the development PC; see
  [`devlog.md`](devlog.md). No generative code exists yet.
