# Model licences and the model downloader

Reviewed 2026-11-07. This is an engineering summary of each model's published terms, not legal
advice; the linked licence text governs. "Checked" means the licence field and card of the
Hugging Face repository were read on 2026-11-07 through the Hugging Face API and model pages.

## How the app handles weights

| Question | Answer | Evidence |
|---|---|---|
| Does the repository or any release package contain model weights? | **No.** The v0.1.0 MSI, portable ZIP and Linux tarball were unpacked and listed; they hold the two executables, documentation and NVIDIA's CUDA libraries only. | `RELEASE_COMPLIANCE.md` |
| Does the project re-host weights? | **No.** Every download URL is the publisher's own Hugging Face repository or Comfy-Org's repack of it (two tokenizer files come from the ComfyUI GitHub repository). | `crates/engine/model-packs.json` |
| What does the app do with weights? | It **loads** files the user put in their Models folder, or **downloads** them on request through **Help › Get AI Models…** (`models.get`) with the system `curl`, over HTTPS, checking each file's size and SHA-256. | `crates/engine/src/model_packs.rs` |
| Is anything downloaded automatically? | **No.** A download starts only when the user presses Download (or an automation client calls `models.get`). | `crates/ui-egui/src/get_models_ui.rs` |
| Are models locked behind a licence switch? | No (owner's standing decision: every model is bring-your-own and selectable). Research-licensed models are labelled in the model picker (`license: "research"` in each template). | `crates/genai/workflows/*.json` |

### Change made in this pass (FIXED)

Before this pass, Get AI Models showed file names and sizes only. The default pack for 24 and
32 GB cards (`standard`) downloads Qwen-Image-2.1 (non-commercial) and Krea 2 (revenue-capped)
from Comfy-Org's repacks, which do not ask the user to accept anything, while Krea's own
`krea/Krea-2-Turbo` and Meta's `facebook/sam3` repositories are gated behind accepting the
licence. The downloader therefore bypassed the acceptance step the licensors set up.

Now:

- every file and runtime archive in `model-packs.json` names a licence (`license`), and the
  manifest has a `licenses` list (name, link, commercial `yes`/`conditional`/`no`/`unknown`, a
  one-line summary, and whether the licensor's own page asks for acceptance);
- `models.packs` returns the licences, and **Get AI Models lists the licences of exactly the files
  it is about to fetch**, each linked, with commercial-use status, plus a note that the weights are
  not covered by ShotoPhopEX's licence;
- **Download stays disabled until the user ticks "I have read these licences and accept them for
  the files I download"**, and `models.get` refuses to download anything unless the caller passes
  `"acceptLicenses": true`. (Remote automation, the control server and MCP, cannot call
  `models.get` at all: `crates/automation/src/workspace.rs` denies it at the automation boundary.
  The parameter guards every other caller, such as scripts using the engine directly.)
- tests: `get_downloads_nothing_until_the_licences_are_accepted`,
  `packs_list_the_licences_of_what_they_offer`, and `the_manifest_is_complete` now fails if a
  file names no licence or an unknown one.

Model loading is unchanged: a file already in the Models folder loads as before.

This is a click-through acknowledgement inside ShotoPhopEX, not the licensor's own acceptance
record. Whether it satisfies a licensor that gates its own repository is an open question
(`UNRESOLVED_RISKS.md`, U-3).

## Per-model table (files Get AI Models can download)

"Pack" is where the file is offered: **S** = standard (24/32 GB), **L** = low-VRAM (12/16 GB,
mac16), **Sel** = selection (8 GB), **E** = optional extra. Every file is **downloaded on
request, never bundled**.

| Model / file | Publisher, source URL | Weight licence | Commercial use | Redistribution | Attribution | Acceptable-use terms | Source asks to accept terms? | Pack | Status |
|---|---|---|---|---|---|---|---|---|---|
| Krea 2 Turbo `krea2_turbo_fp8_scaled` | Krea; Comfy-Org/Krea-2 repack of krea/Krea-2-Turbo | Krea 2 Community License | Only below US$1M annual revenue | Under the licence's terms; keep the licence | Per licence | Krea 2 Use Policy (krea.ai/krea-2-use-policy); card says deployers must implement content filtering or review | **Yes** at krea/Krea-2-Turbo (gated "auto"); the repack is not gated | S | VERIFIED (licence field); thresholds from the licence PDF |
| Qwen-Image-2.1 DiT `qwen_image_2.1_bf16` | Alibaba Qwen; Comfy-Org/Qwen-Image-2.1 | Qwen Research License | **No** ("FOR NON-COMMERCIAL PURPOSES ONLY") | Non-commercial, with the licence | Derived models must say "Built with Qwen" / "Improved using Qwen" | Research/evaluation only | No | S | VERIFIED |
| Qwen-Image-2.1 VAE | Comfy-Org/Qwen-Image-2.1 | Qwen Research License (hosting repo) | No | As above | As above | As above | No | S | VERIFIED (repo-level licence) |
| Qwen3-VL 8B text encoder (`qwen3vl_8b_bf16`) | Comfy-Org/Qwen-Image-2.1 (Qwen3-VL-8B-Instruct is Apache-2.0 upstream) | Apache-2.0 upstream; hosted in a Qwen Research repo | Yes upstream | Apache-2.0 | Keep notices | None | No | S | UNRESOLVED (U-4): the repack's repo-level licence is Qwen Research; manifest labels it Apache-2.0 per the upstream model |
| Qwen3-VL 4B text encoder (`qwen3vl_4b_fp8_scaled`) | Comfy-Org/Krea-2 (Qwen3-VL-4B-Instruct is Apache-2.0 upstream) | Apache-2.0 upstream; hosted in the Krea 2 repack | Yes upstream | Apache-2.0 | Keep notices | None | No | S | UNRESOLVED (U-4), as above |
| Wan 2.1 VAE | Comfy-Org/Wan-2.1 (Wan-AI) | Apache-2.0 | Yes | Yes | Keep notices | None | No | S | VERIFIED |
| Qwen-Image VAE | Comfy-Org/Qwen-Image | Apache-2.0 | Yes | Yes | Keep notices | None | No | S, E | VERIFIED |
| Krea 2 AnyPaint LoRA | yijunwang2/krea2-anypaint | Krea 2 Community License | Below US$1M | Per licence | Per licence | Krea use policy | No (repo not gated) | S | VERIFIED; repo's own NOTICE and PIPELINE_LICENSE not read |
| Qwen 2.1 Outpaint LoRAs (v2, wide) | ausboss/Qwen-Image-2.1-Outpaint-LoRA | "qwen-research-license" (no text linked) | No (assumed, as derivative of Qwen-Image-2.1) | Non-commercial | — | Research | No | S | UNRESOLVED (U-5): licence named but no text |
| Qwen 2.1 Viggle turbo LoRA | Viggle/Qwen-Image-2.1-viggle-turbo | Qwen Research License (LICENSE in repo) | No | Non-commercial | Repo NOTICE (not read) | Research | No | S | VERIFIED (licence field) |
| SAM 3.1 `sam3.1_multiplex_fp16` | Meta; Comfy-Org/sam3.1 repack | SAM License | Yes | Allowed; pass the licence on | Acknowledge SAM in publications | No military/warfare, weapons, nuclear, espionage, ITAR or other trade-control-restricted end uses | **Yes** at facebook/sam3 and facebook/sam3.1 (gated "manual"); the repack is not gated | S, L, Sel | VERIFIED (facebook/sam3 licence read; sam3.1 licence file not read) |
| SAM 3.1 text tokenizer (CLIP `tokenizer.json`) | openai/clip-vit-large-patch14 (pinned commit) | **None stated** on the Hugging Face page; OpenAI's CLIP code repository is MIT | Unclear | Unclear | — | Card: "Any deployed use case … is currently out of scope" | No | S, L, Sel | UNRESOLVED (U-6) |
| ViTMatte base | hustvl/vitmatte-base-composition-1k | Apache-2.0 (README previously said MIT; corrected) | Yes | Yes | Keep notices | None | No | S, L, Sel | VERIFIED |
| Kromoss (MOSS Turbo) DiT | lodestones/Kromoss | Apache-2.0 | Yes | Yes | Keep notices | None | No | L | VERIFIED |
| Kromoss 8-step DMD2 LoRA | lodestones/Kromoss | Apache-2.0 | Yes | Yes | Keep notices | None | No | L | VERIFIED |
| MOSS-Image SFT DiT (int8) | Comfy-Org/MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT) | Apache-2.0 | Yes | Yes | Keep notices | None | No | L | VERIFIED |
| Qwen3.5 9B text encoder + tokenizer | Comfy-Org/MOSS-Image; tokenizer from OpenMOSS-Team/MOSS-Image-SFT | Apache-2.0 | Yes | Yes | Keep notices | None | No | L | VERIFIED |
| FLUX.2 small VAE `flux2-vae_small` | Comfy-Org/MOSS-Image | Apache-2.0 (hosting repo) | Yes per repo | Yes per repo | Keep notices | None | No | L | UNRESOLVED (U-7): provenance of the "small" VAE not traced back to Black Forest Labs' own terms |
| **MOSS Edit adapter** + config | chinoll/Moss-Image-Edit | **No licence stated** for the weights (the project's code is GPLv3) | **Unknown** | **Unknown** | — | — | No | L (default pack) | **UNRESOLVED (U-2)**: no permission established; offered with "No licence stated" and commercial "unknown" |
| Qwen-Image-Edit-2511 DiT | Comfy-Org/Qwen-Image-Edit (Qwen/Qwen-Image-Edit-2511) | Apache-2.0 | Yes | Yes | Keep notices | None | No | E | VERIFIED |
| Qwen2.5-VL 7B text encoder | Comfy-Org/Qwen-Image | Apache-2.0 | Yes | Yes | Keep notices | None | No | E | VERIFIED |
| 2511 Lightning 4/8-step LoRAs | lightx2v/Qwen-Image-Edit-2511-Lightning | Apache-2.0 | Yes | Yes | Keep notices | None | No | E | VERIFIED |
| 2511 Multiple Angles LoRA | fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA | Apache-2.0 | Yes | Yes | Keep notices | None | No | E | VERIFIED |
| 2511 Object Remover LoRA | prithivMLmods/QIE-2511-Object-Remover-v2 | Apache-2.0 | Yes | Yes | Keep notices | None | No | E | VERIFIED |
| Krea 2 General Edit LoRA v2 | conradlocke/krea2-identity-edit | Krea 2 Identity Edit v2 licence (LICENSE_V2.md) **and** Krea 2 Community License | **No** without the author's commercial licence | Per licence | Per licence | Krea use policy | No | E | VERIFIED (V2 terms read) |
| Krea 2 style reference LoRA | ostris/krea2_turbo_style_reference | Krea 2 Community License | Below US$1M | Per licence | Per licence | Krea use policy | No | E | VERIFIED |
| Qwen tokenizer: vocab | Qwen/Qwen2.5-VL-7B-Instruct (pinned commit) | Apache-2.0 | Yes | Yes | Keep notices | None | No | S, E | VERIFIED |
| Qwen tokenizer: merges, config | ComfyUI GitHub repository (pinned commit) | Qwen's tokenizer data (Apache-2.0 at Qwen), hosted in a GPL-3.0 repository | Yes (data originates at Qwen) | — | — | — | No | S, E | UNRESOLVED (U-8, low): hosting repo licence differs from the data's origin |
| NVIDIA CUDA runtime, cuBLAS, cuRAND (Windows, when no runtime folder is set) | developer.download.nvidia.com redist archives | NVIDIA CUDA Toolkit EULA | Yes | Downloaded by the user from NVIDIA, not redistributed by us | — | EULA restrictions | Not through the downloader | runtime | VERIFIED |

### Models the app runs but does not download

| Model | Licence | Notes |
|---|---|---|
| TAESD / taew2_1 / taeqi2_1 preview decoders (`vae_approx/`) | MIT per madebyollin; taew2_1 repo not found on 2026-11-07 | Bring your own; UNRESOLVED (U-9, low) for the exact repo |
| Qwen-Image-2.1 Restore LoRA (`qwen-image-2.1-restore`) | "qwen-research-license" (no text linked) | Same as U-5 |
| Any user-supplied checkpoint or LoRA | User's responsibility | The app does not inspect licences of files it did not download |

### Planned Mac models (thread "Build the Mac MLX .dmg version"; not yet in the manifest)

| Model | Licence | Commercial | Source gate | What the downloader must do |
|---|---|---|---|---|
| FLUX.3 Klein (BFL `black-forest-labs/flux-3-image-klein`) | FLUX Kommunity License v1.0 | Limited; production use above BFL's threshold needs a paid licence; distilling/training other models needs a licence | **Yes** (gated at BFL) | Use licence id `flux-kommunity` (already in the manifest). If the Mac pack downloads the community MLX conversion `mflux-community/flux-3-image-klein-mflux-q4` (or q8/q6), which is not gated, the in-app acceptance is the only acceptance step: that is U-3. Redistribution of BFL weights or conversions is allowed only non-commercially, with the licence text, BFL's attribution notice and a statement that the files were modified: whether the community repo complies is that repo's question, but ShotoPhopEX should prefer BFL's own repo with the user's token where practical. |
| MOSS-Image SFT 4-step LoRA (`akhaliq/MOSS-Image-SFT-4-steps`) | Apache-2.0 (per the Mac thread's survey) | Yes | No | `apache-2.0` |
| MOSS-Image 4-step adapter (`darask0/MOSS-Image_4-step_adapter_lora`) | Apache-2.0 (per survey) | Yes | No | `apache-2.0` |
| MOSS Edit (`chinoll/Moss-Image-Edit`) on Mac | None stated | Unknown | No | `unstated`; U-2 |
| mflux (MIT, Python/MLX) as a porting reference | MIT | Yes | — | If code is translated from mflux into Rust, keep mflux's MIT copyright notice in `genai-native/NOTICE` and the ported files |

## Obligations for the project

- **Never ship or re-host weights** in the repository, release packages or a mirror. (VERIFIED:
  none shipped.)
- **Keep the acceptance step** in Get AI Models and `models.get` whenever new files are added;
  give every new manifest entry a `license` id (enforced by `the_manifest_is_complete`).
- **Keep licence labels in the picker**: research-only templates stay labelled
  (`license: "research"`).
- **Do not describe open weights as open source** or as free for commercial use in the README or
  release notes unless the licence says so (README "Licences matter" was corrected).
- For a model with an **acceptable-use policy** (Krea 2, SAM 3.1, FLUX), the app does not filter
  prompts or outputs. Krea's card says deployers must implement content filtering or equivalent
  review: whether a local, bring-your-own editor is a "deployer" is U-10.
