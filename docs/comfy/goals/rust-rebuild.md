# The Rust rebuild: one executable, no Python, no ONNX Runtime

> **Mission.** Every generative feature of PhotoshopEX runs in PhotoshopEX's own Rust code:
> the editor and its native worker (`photocraft.exe --engine`, genai-native's `qir`,
> `gn-diffusion`, `gn-perception`, `gn-engine`), on candle and our own CUDA kernels. No
> ComfyUI, no `python.exe` process, no ONNX Runtime, no Python step to prepare a model. The
> only libraries beside the executable are NVIDIA's (cuBLAS/cuBLASLt/cuRAND, loaded on first
> use). ComfyUI stays an optional engine for people who have it, never a requirement.

Owner's ask (2026-10-27): "the full rust rebuild of onnx and stuff so everything can be one
exe and rust native without having weird python instances that OOM", then "we need to get
started on the rust rebuild of this project, make this its own task; use Codex as much as you
need for parallelizing tasks".

This file is the task: its definition of done, the work queue, who does what, and the
progress log (newest last). The plan behind it is `docs/comfy/plan-native-engine.md`
(phases 5 and 6); the measurements are in `docs/comfy/benchmarks.md`.

---

## 1. Done when

1. On this PC with ComfyUI **not running and not configured** (`integrations.comfyPath` and
   `comfyServer` empty) and `integrations.engine = native`, every generative command in the
   feature table (§2) produces its result, and no `python.exe` starts (`walkthrough-live.ps1`
   extended to check the process list).
2. The worker loads no `onnxruntime*.dll` and no `cudnn*.dll` (checked with the module list of
   the running worker).
3. The models come from one folder the user points at, in ComfyUI's own layout
   (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`, `sam3/`), with no converted or
   exported copy: no `convert-fp8`, no `export_sam31.py`.
4. `packaging/windows/package.ps1 -Native` builds a portable zip whose runtime folder holds
   only NVIDIA's libraries (target under 700 MB; cuDNN's 542 MB and ONNX Runtime's 205 MB
   gone), and a fresh Windows user profile on this PC runs the walkthrough from it.
5. Each model moved to native is at parity with its ComfyUI route by the harness's rule
   (`plan-native-engine.md` §12.1) or beats it on a measured benchmark, and is at least as
   fast. Tests, clippy, wasm, layers, parity and scorecard gates stay green at every commit.

## 2. Where it stands (2026-10-28)

| Feature | Default model | Native today | Left to do |
|---|---|---|---|
| Generate Image | Krea 2 Turbo | yes (fp8, flash, fused; 4.0 s; from ComfyUI's folder as is since R4) | — |
| Generative Fill (prompted) | Qwen-Image-2.1 fill / edit by instruction | yes (with the research opt-in) | the permissive default is still 2511 on ComfyUI (R6) |
| Remove, Content-Aware Fill, Spot Healing (generative) | 2.1 edit | yes | — |
| Generative Expand | 2.1 expand | yes | — |
| Generative Edit | 2.1 edit; Krea 2 edit (General Edit V2) | yes | — |
| Remove Background (matte) | 2.1 matte | yes | — |
| Select by text / subject / point, Object Selection | SAM 3.1 | **on candle from ComfyUI's checkpoint**, f16 on the GPU (since 2026-10-28: by text on a new picture 273–278 ms through the app against ONNX Runtime's 249–346 ms); the -Native build has no ONNX Runtime at all | — (the ONNX graphs stay as the fallback of builds without `native-diffusion`) |
| Prompt rewriter, judge, naming | 2.1's Qwen3-VL 8B as a chat model | yes | — |
| Fill from a reference | 2.1 edit by instruction (since 2026-10-27; 2511 fill-ref without the opt-in) | yes | — |
| Image from references | 2.1 image-ref (since 2026-10-27; 2511 image-ref without the opt-in) | yes | — |
| Generative Angles | 2.1 edit by a camera instruction (since 2026-10-27; 2511 + Multiple Angles LoRA without the opt-in) | yes | compare with 2511 + the LoRA (not installed here) |
| Krea 2 style reference | Krea 2 + style LoRA | no (ComfyUI) | the Krea 2 reference path exists now; the style LoRA (R2) |
| Split into layers | Qwen-Image-Layered | yes, under `auto` (at parity; 57 s against ComfyUI's 58 s at 640 px, 2026-10-27) | speed at the owner's 1 MP 3:4 size (179 s against 140 s; the documentation-pass session's Codex job) |
| Generative Upscale | 2.1 edit tiles | yes | — |

## 3. The queue

Owners: **Claude** writes kernels, model code, integration and decides; **Codex** (`codex exec`,
read-only or in its own worktree) runs the parallel grind: audits, porting specs from the
reference code, parity dump scripts, test scaffolds, reviews of Claude's commits. Every Codex
result is checked by Claude before anything lands. Measurements run on the one GPU, one at a
time.

| # | Work | Owner | Gate |
|---|---|---|---|
| R0 | Inventory of every Python / ComfyUI / ONNX Runtime dependency at run, build and setup time | Codex (audit), Claude (check) | the table in §2 confirmed with file:line |
| R1 | ✅ 2026-10-28. **SAM 3.1 on candle** (`gn-perception` without `ort`): the vision backbone, text encoder, geometry encoder, transformer, heads and mask decoder, read from ComfyUI's checkpoint | Codex (porting spec, dump scripts), Claude (port) | masks at IoU ≥ 0.99 against the ONNX route on the harness's `sam` suite; select.byText as fast or faster; `ort` out of the worker |
| R2 | **The 2511 holdouts on native routes**: fill from a reference and image from references (2.1 edit takes nine references; Krea 2 edit LoRAs take two), Generative Angles (Krea 2 + `krea2_anygles`, or 2.1), the Krea 2 style reference | Claude; Codex reviews | each beats or matches its 2511 route on a small benchmark like the swap set; then `auto` sends them native |
| R3 | ✅ at 640 px, 2026-10-27 (1 MP speed open). **Qwen-Image-Layered native** (`generate.split`) | Codex (porting spec), Claude (port) | the split's layers at the harness's rule against the ComfyUI route |
| R4 | **One models folder in ComfyUI's layout**: the worker resolves each model from `diffusion_models/`, `text_encoders/`, `vae/`, `loras/` by file name (Krea 2 already runs from ComfyUI's fp8 files; 2.1 still wants a diffusers snapshot), the first run defaults to ComfyUI's folder when one exists, a "Get models" page of links | Claude; Codex writes the resolver tests | the walkthrough with `integrations.modelsPath` = ComfyUI's `models` folder and nothing else |
| R5 | ✅ 2026-10-28 (the profile is a fresh one in this account, not a second Windows account). **Packaging**: the runtime folder without ONNX Runtime and cuDNN, the clean-profile install, the walkthrough from the zip | Claude; Codex scripts the clean-profile check | §1.2, §1.4 |
| R6 | ⏳ waits on the owner's licence decision (`auto` already prefers the worker). **Native by default**: `integrations.engine` defaults to native when the worker has its folders; the ComfyUI sidecar only when asked; the permissive build's defaults decided with the owner (2.1 is research-only) | Claude, the owner decides the licence question | §1.1 |

Order: R0 → R1 and R3 specs in parallel (Codex) while Claude does R2 (no new models needed) →
R1 port → R4 → R3 port → R5 → R6.

## 4. Rules for this task

Everything in `CLAUDE.md` and `AGENTS.md` holds: never crash, no `unsafe` outside the kernel
FFI, tests are the gate, BYOM (no weights shipped), research-only models never a permissive
default, commits local, genai-native not pushed without the owner, one commit per logical
step, the dev log and roadmap updated.

## 4b. Specs (Codex, checked)

- R0 inventory: [`research/notes/rust-rebuild-r0-inventory.md`](../research/notes/rust-rebuild-r0-inventory.md).
  Adds one gap to §2: the standalone `generate.enhancePrompt` command routes through
  `backend(s)`, so under `auto` it goes to ComfyUI although the worker answers `enhance/*`.
- R1 SAM 3.1: [`research/notes/rust-rebuild-r1-sam31.md`](../research/notes/rust-rebuild-r1-sam31.md).
  Read the raw `sam3.1_multiplex_fp16.safetensors` (1 590 F16 tensors) directly: ViTDet trunk
  (32 blocks, 1024 wide, 24×24 windows with global blocks 7/15/23/31, tiled absolute positions
  plus axial RoPE), two FPNs, CLIP text (24 blocks, QuickGELU, causal), box geometry with
  ROIAlign, a 6+6-layer detector with a learned box bias, the interactive two-way mask decoder.
  4 000–6 000 lines in ten parity-gated steps; no cuDNN; F16 attention at head dims 64/32/16 is
  the speed work (our flash kernel is D=128 bf16 only).
- R4 one models folder: [`research/notes/rust-rebuild-r4-comfy-layout.md`](../research/notes/rust-rebuild-r4-comfy-layout.md).
  ComfyUI's bf16 2.1 files load with three adaptations: the DiT's fused `img_mlp.gate_up` (gate
  rows first), the text encoder rooted at `model`, the VAE's Wan-style names (a bijection, with
  76 singleton time axes). The tokenizer is Qwen's standard one: assembled at run time from
  the vocabulary, merges and added tokens in ComfyUI's `comfy/text_encoders/qwen25_tokenizer`
  (the 11 MB `tokenizer.json` we have is Qwen's pretty-printed; its provenance is a research
  snapshot or Krea's repo, so it is not embedded). ComfyUI's `int8_convrot` files need signed
  bytes, per-row scales and a reversed Hadamard basis (later).
- R3 Qwen-Image-Layered: [`research/notes/rust-rebuild-r3-qwen-image-layered.md`](../research/notes/rust-rebuild-r3-qwen-image-layered.md).
  The older 60-block dual-stream Qwen-Image DiT (3072 wide), a Qwen2.5-VL 7B text encoder
  (needs a variant: biases, no q/k norm, final norm, legacy fp8 scales), the 4-channel Wan 2.1
  VAE (our new Wan encoder plus RGBA), L+1 latent slices; the fp8mixed DiT reads with `qir::fp8`
  as it is (all layers `full_precision_matrix_mult`, biased). 1 800–3 200 lines in seven steps;
  28 GB of weights, so staged residency.

## 5. Progress log

- 2026-10-27, evening: task opened. Native today: Krea 2 text to image and edit, every
  Qwen-Image-2.1 route (image, fill, edit, expand, matte, removals, upscale, the rewriter), the
  engine page and GPU detection. ComfyUI is still needed for the four 2511 holdouts, the
  Layered split and the Krea 2 style reference; ONNX Runtime and cuDNN for SAM 3.1. Codex runs
  R0, the R1 spec and the R3 spec in parallel (read-only).
- 2026-10-27, evening: the three Codex reports landed and are filed (§4b). R2 started: a fill
  from references under `auto` with the native worker now runs on the 2.1 edit template (nine
  reference slots) by instruction ("Replace the <thing> with <prompt> exactly as image 2 shows
  it"), not 2511's fill-ref on ComfyUI; routing test added. Next: measure it against the 2511
  fill-ref pictures of 2026-10-21, then image from references, then R1 step 1 (a `gn-segment`
  crate: weights and ops) with Codex building the ops and their torchvision fixtures in its
  own worktree.
- 2026-10-27, night: **R1 step 1 landed** (genai-native `3f1c9f1`, written by Codex in a
  worktree, reviewed and landed by Claude: Codex's sandbox cannot commit to the shared git
  metadata): `gn-segment` validates the raw SAM 3.1 checkpoint from its header (1 264 tensors,
  none missing) and has torchvision-exact ROIAlign, the antialiased resize and the channel
  LayerNorm, fixture-tested (< 1e-5). **R2: images from references native** (PhotoshopEX
  `30ec008b`, genai-native `e1c0bb7`: `qwen-2.1/image-ref`, 2.1's text to image with one to
  nine references at the asked size). Fill by instruction measured with the rewriter on and
  turned back off by default (the 8B named the selection's neighbour on 3 of 15 swaps).
  Running: the native fill-from-reference and image-from-references checks; Codex on R1 step 4
  (the CLIP text encoder against ComfyUI's own features) and R3 step 2 (the RGBA Wan VAE),
  each in its own worktree.
- 2026-10-27, late: **R1 step 4 landed** (genai-native `a810339`, Codex: SAM 3.1's CLIP text
  encoder within 2.7e-5 of ComfyUI's on the real checkpoint) and **R3 step 2** (`4d6013d`,
  Codex: the Wan VAE in RGBA for the Layered file, within 2.5e-6 of ComfyUI's loader; Krea 2's
  RGB path unchanged). The R0 gap fixed (PhotoshopEX `e1fc5fda`: Improve Prompt asks the worker
  under auto). **R2 measured**: fill from a reference natively level with 2511's (a faint
  neckline ghost), image from references right in 11 s (`benchmarks.md`). In the §2 table that
  leaves on ComfyUI: Generative Angles, the Krea 2 style reference, the Layered split; on ONNX
  Runtime: SAM 3.1. Codex now on R1 step 5 (the ViT trunk and both FPNs) and step 7 (the
  interactive point/box heads), in parallel worktrees.
- 2026-10-27, later: **every SAM 3.1 part is on candle**, each against ComfyUI's own on
  the real checkpoint (Codex in worktrees, landed by Claude; genai-native): the vision trunk and
  both pyramids (`155aa14`, 1.4e-4), the interactive heads (`e0538e3`, 1.1e-4), the box
  geometry and CLS (`45d76f3`, 2e-6), the detector (`181f623`: all 200 boxes and logits within
  3e-5, masks 6e-4, the same top queries). Codex N now composes them (one checkpoint read, the
  select-by-text and interactive flows, the post-processing) against the ONNX route end to end.
  **R3**: the Qwen2.5-VL 7B encoder (Codex L; ids exact, conditioning 9e-4) and the 60-block
  Layered DiT streamed from the fp8mixed file (Codex M; velocity 9.2e-4, 1.1 GB RAM) pass on CPU.
  **R2**: Generative Angles natively (PhotoshopEX `da4ce52b`, 2.1 edit by a camera
  instruction, 6 s; Krea 2's edit LoRA is no route for it). **R4**: 2.1 and Krea 2 load from
  ComfyUI's `models` folder as it is (`comfy_files`; the prequantized-DiT cache goes to
  `%LOCALAPPDATA%\genai-native\qir-cache`, never into ComfyUI's folder); Krea 2 from ComfyUI's
  files is pixel-identical to the snapshot route.
- 2026-10-27: R4 landed (genai-native `58f7a15`: 2.1 image and edit pixel-identical too; a Q8_0
  text-encoder GGUF cached once, 2.1 cold load 15 s), R3 encoder (`9994e39`) and DiT (`9941269`).
  **R1 step 9**: the worker's perception engine runs SAM 3.1 on candle on the GPU whenever the
  checkpoint is there (`7619fd6`; PhotoshopEX `41b6d0ef` builds it with `native-diffusion`), the
  CLIP tokenizer assembled from ComfyUI's `sd1_tokenizer`. `sam-live.ps1` through the app:
  same bounds as the ONNX route, pixel counts within 1%, points 30 ms on both, by text on a
  new picture 1.7–1.9 s against 0.3–0.4 s. Codex on step 10 (GPU speed: weights cast once,
  bf16, no host round trips) and on R3 steps 5–6 (the Layered noise, schedule and pipeline).
- 2026-10-27, night: **R1 step 10, first pass** (`7eefc85`, Codex): select by text on a new
  picture 515 ms in f32 on the 5090 (1.7 s before), 260 ms with an opt-in f16 path, ONNX
  Runtime 190–210 ms; points 9–13 ms (ONNX 15–24). **The parity reference is ComfyUI's node**
  (`8404ffd`, CUP.md): on the one picture that disagreed, candle equals ComfyUI's f32 SAM 3.1
  exactly; the ONNX graphs' mixed precision drifts. Codex now measures f16 against ComfyUI on
  12 pictures. **R3 end to end**: the Layered pipeline (`ec668ba`, Codex) and the worker's
  `qwen-layered/split` (`3056cda`; `7f5abb8` routes its fp8 linears through qir's kernels on
  CUDA): through the app on 640×640, 3 layers, against ComfyUI's split, mean RGB difference
  1–3/255 and alpha IoU 0.994–1.000 per layer; **86 s against ComfyUI's 58 s**, so it serves
  `engine = native` (PhotoshopEX `29308319`) and joins `auto` when it is at least as fast.
  **R4 app side** (`8872dbcf`): Models folder empty means the ComfyUI install's `models`.
  **R5**: `runtime.ps1` without ONNX Runtime and cuDNN, 581 MB (`130e4b1b`).
- 2026-10-27, late night: the split as fast as ComfyUI at 640×640 (`4fcbfc3`, Codex: 57 s vs
  58 s) and under `auto` (PhotoshopEX `182ded54`). The owner's 1 MP 3:4 test (`maxPixels`,
  `980e410e`: 864×1152) on a 3:4 portrait: native and ComfyUI agree within 1.4/255 over visible
  pixels and alpha IoU ≥ 0.996 at 1 MP and at the 640 default (480×640), but native is slower
  there (179 s vs 140 s; 44 s vs 36 s). Codex on that. (An earlier metric averaged RGB under
  zero alpha too and reported 14/255; `layered-live.ps1` now counts visible pixels only.)
- 2026-10-28, overnight (after a PC crash at about 23:47: Kernel-Power 41, bugcheck 0, no GPU
  driver or WHEA event; my -Native build and the documentation-pass session's 1 MP Layered
  benchmark were both running, so the cause is not pinned): **R1, R5 done; R6 waits on the
  owner.**
  - **The Layered RoPE kernel compiled at build time** (genai-native `b4fe9e7`,
    `kernels/fusednorm/layered_rope.cu`): Codex's version compiled through NVRTC at run time,
    and the packaged runtime folder has no nvrtc library. Bit-identical to the unfused path.
    `c3b1659`: gn-segment's `sam31_f16` example requires `cuda`, so `--all-targets` checks again.
  - **ONNX Runtime out of the -Native build** (genai-native `d10fafa`, `2849310`; PhotoshopEX
    `94cbfe7a`): `gn-perception`'s `onnx` feature (default on), PhotoshopEX's `sam-onnx` (the
    apps' default; the workspace takes `photocraft-genai` without defaults), `package.ps1
    -Native` builds with `--no-default-features`: `cargo tree` has no `ort` in the -Native graph
    and still has it in the plain one. A build with neither SAM backend compiles and answers
    ModelMissing.
  - **SAM 3.1 in f16 by default** (genai-native `766ecb0` `--sam-precision`; PhotoshopEX
    `cd33f8bc`, `PHOTOCRAFT_SAM_PRECISION=f32` to go back): `sam-live.ps1` through the app, by
    text on a new picture **273–278 ms** against ONNX Runtime's 249–346 ms and candle f32's
    508–515 ms; points 28 ms (ONNX 27–36); masks within 0.3 % of f32's pixels. F16.md: every
    available mask at IoU ≥ 0.993 against ComfyUI's own f32 node (3 of its 12 pictures have no
    object to find, so 9 positives back the gate).
  - **The -Native zip** (`package.ps1 -Arch x64 -Native -NoMsi`, PhotoshopEX `8a05359c`,
    `14793b47`: `-NoMsi` for a machine without WiX; the CUDA kernels compiled `/MT` because the
    package links the C runtime statically): **557 MB** with the 581 MB runtime folder
    (cuBLAS, cuBLASLt, cuRAND, cudart).
  - **Clean profile** (`clean-profile-native.ps1`, `ed14c8f8` … `ddf59025`): the zip unpacked
    fresh, APPDATA / LOCALAPPDATA / USERPROFILE / TEMP in a new folder (empty qir cache), a
    Windows-only PATH, ComfyUI neither running nor configured, engine = native, models from
    ComfyUI's `models` folder, research models allowed. **11 of 11** cases, each in its own CLI
    process from cold: image (Krea 2) 21–25 s, by text 2.1–2.3 s, by point 1.4–1.5 s, fill
    32–34 s, Remove 15 s, edit 13 s, expand 12–13 s, matte 13 s (41 s with an empty cache),
    angles 13 s, upscale 2× 82–83 s, split 62–63 s. **No python.exe started; the worker loaded
    63 modules and none of onnxruntime, cudnn, python or torch**; from the zip it loaded only
    cuBLAS, cuBLASLt and cuRAND.
  - Found on the way: **Remove Background with an empty prompt** (2.1 matte) kept nothing on 3
    of 5 random seeds; with "the man" 3 of 3 at 514–515 k pixels. Not a rebuild regression (the
    route's own behaviour), open for the owner.
  - **R6**: `integrations.engine` is `auto` by default and `auto` already sends every template
    the worker serves to it (`native_serving`); the ComfyUI sidecar starts only for a template the
    worker does not serve. What stays on ComfyUI in the permissive build (research models off,
    the default): prompted fill, edit, expand, the removals, angles and upscale default to
    Qwen-Image-Edit-2511 there. Left as it is: the licence decision is the owner's.
