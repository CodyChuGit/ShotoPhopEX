# Plan: the native engine (`genai-native`)

A hand-off plan for an agent (or a person) to take PhotoshopEX from "a Rust editor that drives a
ComfyUI process" to **production software with its own inference engine**: one executable, no
Python in the box, the models the owner cares about (Krea 2 Turbo, Qwen-Image-2.1 for generation
and editing, SAM 3.1 for selection) running in-process on the RTX 5090 through our own Rust
engine, everything else still available through engines we do not have to write. Written
2026-10-18 against commit `9e75bf4` of branch `photoshopex`
(`C:\Users\5090\Documents\PhotoshopEX`, the only checkout from now on).

The owner's brief, in their words, across the 2026-10-16 to 2026-10-18 sessions: "this needs to
leave the lab and become real production software", "I'm not distributing models, it's always
BYOM", "my biggest concern is that there will be new frontier models by then", "I want to do this
once and do it right". Every section below answers one of those four sentences.

---

## 0. How to work this plan

**Read first, in order:** `CLAUDE.md` (the owner's local file), [`../../AGENTS.md`](../../AGENTS.md)
(never crash, everything is a command, tests are the gate, never break wasm), the research report
[`research/Native inference for PhotoCraft on RTX 5090.md`](research/Native%20inference%20for%20PhotoCraft%20on%20RTX%205090.md)
and its five notes in `research/notes/`, [`architecture.md`](architecture.md),
[`codebase-orientation.md`](codebase-orientation.md), [`models.md`](models.md) (licences),
[`benchmarks.md`](benchmarks.md) (every number we have), the top of [`devlog.md`](devlog.md), and
[`plan-artcraft-experience.md`](plan-artcraft-experience.md) §0 for the working rules this plan
inherits (translations ×13, preferences read in the same change, surgical edits to shared
upstream files, the per-phase gate list, how the CLI and the live scripts are driven).

**Rules that bind every phase of this plan, on top of those:**

- **The editor never dies because a model did.** No `unwrap`/`expect`/`panic`/`unsafe` in our
  Rust; every FFI boundary (CUDA, ONNX Runtime, sd.cpp) sits behind a `catch_unwind` and a
  typed error; and the engine runs in a **worker process** (see §3.4), so a CUDA fault or an
  out-of-memory kills a child, not a document.
- **Everything is measured before it is believed.** No kernel, model port or engine switch lands
  without a number from the parity harness (§12.1) on this machine, recorded in
  `benchmarks.md` with the commit, driver and date. Self-reported numbers from other projects
  are hypotheses until reproduced here.
- **Parity before speed.** A port is not done when it produces a nice picture; it is done when
  it matches the reference implementation within the thresholds in §12.1 on the fixture set.
- **BYOM.** The app ships no weights, no model code copied from a restricted repo, and reads
  whatever the user has: official safetensors, ComfyUI repacks, GGUF. Model and licence facts go
  in `models.md` with a source and a date. Research-licensed models (Qwen-Image-2.1) run when
  the user installs them and are never a default (`integrations.allowResearchModels`).
- **Clean room where a licence asks for it.** Qwen's Research License defines "Qwen" to
  include its "machine-learning model code": the Qwen-Image-2.1 architecture is ported from the
  Apache-2.0 diffusers implementation and ComfyUI's Apache-marked
  `comfy/ldm/qwen_image/model.py`, never from `QwenLM/Qwen-Image`. SAM 3.1 comes from the
  Apache transformers port and community ONNX exports, never from Meta's repo. Record the source
  file and commit of every port in `NOTICE`.
- **Perishable code stays thin.** A model definition (§3.2 layer 3) is configuration and wiring
  over shared blocks. If a port needs a new block, the block goes in the block layer with its own
  parity test, not inline in the model.
- **Routing is a preference, not a rewrite.** Every `generate.*` and `select.*` command keeps
  working through whichever engine is available; the native engine takes a model over only when
  it is faster and at parity. The user can pin any model to any engine.
- **Gates per phase:** the inherited list (tests, clippy, `cargo xtask layers`, `parity`,
  `scorecard`, the wasm engine build, a release build, dev-log entry, roadmap markers, one commit
  per phase) plus: the harness run for that phase, and the `benchmarks.md` row.

**Decisions already taken (owner, 2026-10-16/18):**

1. PhotoshopEX (`photoshopex`) is the only line; `Projects\comfy-photocraft` is retired.
2. BYOM, always. No weights, no downloads on the user's behalf without a click.
3. Scope of the native engine: **Krea 2 Turbo, Qwen-Image-2.1 (generate + edit), SAM 3.1.**
   Everything else (Qwen-Image-Edit-2511, Qwen-Image-Layered, Z-Image, the Angles LoRA, the
   prompt enhancer) rides a rented engine until it earns a port.
4. Quality first, then speed (Quality mode `best` is the default; the native engine must match
   it before it may replace it).
5. **Cross-platform: Windows and Linux first, macOS after** (§3.5). Nothing platform-specific
   outside the few places §3.5 names; every phase closes on both platforms.
6. Build it out (2026-10-18): Phase 0 started on this PC.

**Decisions this plan needs from the owner** (§15 lists them with recommendations): the
engine-process model, a second GPU box, the repository split, the test-weight downloads.

---

## 1. What "once and right" means here

The four sentences from the brief, as design commitments:

**"Production software."** No interpreter, no console window, no `pip`. One signed executable
plus a runtime folder of native libraries (CUDA runtime, ONNX Runtime). First run asks for a
models folder and shows what is installed. Failures are sentences in the app, with a log the
user can send. The engine is isolated so a model crash is a retry, not a lost document.

**"Always BYOM."** The weight loader is a first-class subsystem (§3.2 layer 4), because the
user's files will arrive as official bf16 safetensors, ComfyUI's int8/fp8 repacks with their own
tensor names, community GGUF, or a folder someone zipped. A model is "installed" when the
loader can map it, whatever its origin; `generate.models` already reports installed state per
slot and keeps doing so.

**"New frontier models by then."** The engine is a **component library with thin model
definitions**, which is how `stable-diffusion.cpp` ships day-0 support (it added FLUX.3 Klein on
2026-10-16, Qwen-Image-2.1 on its release day 2026-09-20, Krea 2 on 2026-06-25): primitives plus
a short definition per architecture. Every image model in the current frontier is the same kit
(an MMDiT-style transformer under flow matching, a VLM/LLM text encoder, a VAE); they differ in
configuration. So a new model in a known family is days of work in layer 3, and the **day-0
lane** (§3.3) means users run it through a rented engine the day it appears, before anyone ports
anything. What is hand-written is the durable part: kernels, blocks, encoders, VAEs, the loader,
the harness, SAM.

**"Once and right."** The things that get rebuilt when projects are done twice are the
boundaries. This plan fixes them now: the engine's public API is the existing
`GenerativeBackend` trait (unchanged); the engine's internal layers have one-way dependencies
(§3.2); the GPU lives behind a kernel trait so a second backend (Metal, Vulkan) is a new crate,
not a refactor; the process boundary (§3.4) is the same whether the engine is our Rust, sd.cpp,
or ComfyUI; and the parity harness is built before the first port, so every later change is
judged by the same yardstick.

---

## 2. What exists today (the ground)

**In PhotoshopEX (cited to code, commit `9e75bf4`):**

- `crates/genai/src/lib.rs`: the engine-agnostic contract. `Task` (`Fill, Expand, Image, Edit,
  Matte, Upscale, Segment, Layers, Text`), `Request` (template id, prompt, negative, seed, steps,
  guidance, `image: Option<Rgba8>`, `mask: Option<Gray8>`, `models: Vec<(slot, file)>`, `size`,
  `params`, `references: Vec<Rgba8>`), `Response` (images, texts, seed, run_id, elapsed_ms,
  `Timings`), `Progress` (report + cancelled), `Health`, `Precision` (`Auto|Default|Nvfp4`),
  `is_blackwell`, and **`trait GenerativeBackend { health, run, model_files, prepare, free }`**,
  whose doc already says "an in-process runtime could be another" implementation.
- `crates/genai/src/comfy.rs` (673 lines): the ComfyUI implementation; `fake.rs`: the in-process
  fake server every test uses (`Options{segments, segments_for, missing_files, fail, delay_ms,
  device}`); `template.rs`: workflow templates with `Meta` (capability flags, references,
  placeholders, model slots with `nvfp4` alternatives, `estimate_seconds`), 22 JSON templates in
  `crates/genai/workflows/`.
- `crates/engine/src/generate_cmds.rs` (2,783 lines): `generate.fill/expand/image/edit/
  removeBackground/splitLayers/enhancePrompt/similar/angles/models/health/purge`, the quality
  modes (`auto_order`), `backend()` (constructs the ComfyUI backend from prefs), `run_switching`
  (purge on model-set change), `plan_common`; `generate_jobs.rs`: the Generations library;
  `select_ml_cmds.rs` (652 lines): `select.byText/byPoint/subjectML/probe` on `Task::Segment`,
  the probe cache, the fallback chain, `refine_coverage`.
- Jobs (`crates/engine/src/jobs.rs`: `edit_job`, `start_job`, progress, cancel), preferences
  (`prefs.rs` `integrations.*`: `comfy_server`, `quality`, `model_precision`,
  `allow_research_models`, `enhance_prompts`, …; `tests/prefs_usage.rs` enforces that each is
  read), the UI (`generate_panel.rs`, `generative_bar.rs`, `object_select.rs`, `retouch_ui.rs`,
  `prefs_ui.rs`), the control channel (`docs/control-protocol.md`), the CLI, the live scripts in
  `docs/comfy/bench/scripts/` and the bench images in `docs/comfy/bench/images/` (ignored).
- Numbers: `benchmarks.md` (Lightning tiers, request caps, NVFP4 +18 % on Krea 2, SAM probe
  0.5 s warm / ~19 s cold, enhancer ~1 s); the dev log's test baseline (9 GPU/env failures on
  this PC, one of them in the UI crate).

**The research (2026-10-16, five researchers, report + notes):** no Rust runtime runs this model
set; candle measured ~0.65× ComfyUI bf16 stock; burn's attention kernel is an open blocker and
Burn 0.22 (2026-10-06) dropped its candle backend; TensorRT has a FLUX path only; NVFP4 is a
GEMM/VRAM lever (1.2–1.8× per image), not a general speed-up; embedding Python in-process is
unacceptable because a GPU fault then kills the editor; the recommended path was "keep the
trait, ship a sidecar, small models native first, then one DiT on trial". The decision matrix
and the licence table in the report stand; this plan is the execution of its recommendation
with the scope the owner chose.

**The landscape (checked 2026-10-16, see the dev log for the session):**

| Project | What it proves for us | Caveats |
|---|---|---|
| [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) (C++/ggml, MIT, ~7.7k stars) | A no-Python engine already runs **Krea 2, Qwen-Image-2.1, the Qwen Image Edit series, Z-Image, FLUX.2/3** with LoRA, flash attention, CUDA/Vulkan/Metal, GGUF and safetensors. Day-0 cadence. A Rust wrapper exists (`diffusion-rs`). | No FP8/NVFP4 tensor-core path (INT8 "convrot"); **no published speed number against ComfyUI on a 5090**; rolling API. |
| [`umag/qwen-image-rs`](https://github.com/umag/qwen-image-rs) (Rust + CUDA on candle, Apache-2.0) | Qwen-Image-2.1 text-to-image ported from diffusers with parity (DiT cosine 0.9999, VAE PSNR 56 dB); self-reported 0.62 → 0.148 s/step on a 4090 (INT8 CUTLASS GEMMs with Hadamard rotation, vendored SageAttention, fused norms/RoPE); beats TensorRT bf16 and ComfyUI INT8 in the author's runs. `HANDOVER.md` is a development log we can follow. | One developer, 1 star; sm89-specific kernels; text-to-image only (no edit, no LoRA, no masks); 27 s cold start; candle version blocks CUDA graphs. |
| [`SceneWorks/inference` PR #1014](https://github.com/SceneWorks/inference/pull/1014) (merged 2026-09-23) | Qwen-Image-2.1 on candle **and MLX** with reference-image editing (1–10 refs, no mask), q8/q4 tiers, parity ~1e-7 against upstream. Confirms the 2.1 edit interface. | No speed numbers; licence of the repo unverified. |
| SAM 3 TensorRT/ONNX ([yimin-pan](https://yimin-pan.github.io/sam3-trt/), [dataplayer12](https://github.com/dataplayer12/SAM3-TensorRT), [pagarcia](https://github.com/pagarcia/sam3-onnx-cpp)) | Community ONNX exports of the text, point and box paths; **24 ms** text-prompt and **85 ms** point/box on an RTX 5090 FP16. | No official export from Meta ([issue #224](https://github.com/facebookresearch/sam3/issues/224)); the tracker (point/box) export is the finicky part; weights gated on Hugging Face. |
| [TensorRT for RTX](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/) | NVIDIA's redistributable C++ runtime with FP8/FP4 GEMMs for diffusion; FLUX FP4 demo on the 5090. | NVFP4 quantization needs ModelOpt (Python) once, upstream of us; engines are per runtime version. |
| `mistral.rs`, OxiBonsai, `comfyui-rust` | FLUX in Rust exists (mistral.rs; OxiBonsai FLUX.2 Klein pure Rust); a Rust ComfyUI clone exists (2 stars, sd.cpp FFI + Python fallback). | None is a basis for us. |

**Reading of the landscape:** the speed is in CUDA kernels whatever the host language;
qwen-image-rs shows a Rust host with its own kernels can beat ComfyUI on one model, at the price
of per-model, per-GPU work; sd.cpp shows a component engine can keep day-0 pace. We take the
structure from sd.cpp, the kernel approach from qwen-image-rs, and keep both as rented engines
behind the trait while ours grows.

---

## 3. The target, in one picture

```
PhotoshopEX (editor process, Rust)
│  commands: generate.*  select.*            prefs: integrations.*  (engine routing, models dir)
│  crates/genai: Request/Response/Task, GenerativeBackend trait, templates, routing (new)
│
├── EngineClient ──(local channel: pipe/UDS + shared memory for pixels)──► photocraft.exe --engine
│                                                                          (engine worker process, Rust)
│                                                                          ┌──────────────────────────┐
│                                                                          │ genai-native             │
│                                                                          │  5 perception  (ort/SAM) │
│                                                                          │  4 weights     (loader)  │
│                                                                          │  3 models      (defs)    │
│                                                                          │  2 blocks      (library) │
│                                                                          │  1 kernels     (CUDA…)   │
│                                                                          └──────────────────────────┘
├── SdCppBackend ──(same channel)──► engine worker with sd.cpp linked (FFI)   [rented, day-0 lane]
└── ComfyBackend ──(HTTP/WS)──────► ComfyUI sidecar, hidden, app-managed      [rented, day-0 lane, optional]
```

### 3.1 The contract stays

`GenerativeBackend { health, run, model_files, prepare, free }` is the engine API. Nothing above
`crates/genai` learns which engine ran a request; `Response.timings` and a new
`Response.engine: String` say so for the Generations library, the benchmarks and the status
line. New `Task`s are added only when a feature needs one (none are needed for this plan).

### 3.2 `genai-native`: five layers, one-way dependencies

| # | Crate | Owns | Depends on | Changes when |
|---|---|---|---|---|
| 1 | `gn-kernels` | The GPU. A `Kernels` trait (GEMM in bf16/FP8/NVFP4, attention, norms, RoPE, SwiGLU, conv, upsample, casts) with a CUDA implementation (`cudarc` + our `.cu`/CUTLASS kernels compiled to fatbins for sm120 first, sm89 second). Device memory arena, streams, CUDA graphs. | nothing of ours | a new GPU family, a faster kernel |
| 2 | `gn-blocks` | The library: attention (joint/MMDiT double-stream, single-stream), AdaLN-Zero variants, RoPE variants (axial 2D/3D), flow-match schedulers (shifted, Turbo few-step), patch embed/unembed, VAE families (Wan 2.1, Qwen 64-channel RGBA, tiled decode), encoder families (Qwen3-VL, Qwen2.5-VL: tokenizer, vision tower, text tower, template formatting), LoRA application (fused at load). Each block has a parity test against a tiny reference tensor set. | 1 | a new architectural idea (rare) |
| 3 | `gn-models` | **Thin definitions**: `krea2.rs`, `qwen_image_21.rs`, later `z_image.rs`, `qwen_edit_2511.rs`. Each is config (dims, layers, heads, RoPE, schedule), the forward wiring over blocks, the conditioning recipe (how prompt + references + image + mask become the DiT's inputs), and the key map into layer 4. Target: a few hundred lines each. | 2 | a new model in a known family (days) |
| 4 | `gn-weights` | The BYOM loader: safetensors (mmap, hardened parser, no pickle ever), GGUF (dequant on load or quantized paths where kernels exist), ComfyUI repack naming, sharded files, per-model key maps and dtype policies, LoRA files, integrity checks, a model index for a folder (what is installed, which slot it fits). | 1 | a new file layout or quant type |
| 5 | `gn-perception` | SAM 3.1 (and any future detector/matting model) through ONNX Runtime (`ort` crate, CUDA/TensorRT providers), engine cache, pre/post-processing, the prompt kinds (text, points, negatives, box). | — (own runtime) | a new perception model |

Plus `gn-harness` (§12.1), `gn-engine` (the worker process: channel server, request lifecycle,
VRAM budget, model residency, `GenerativeBackend` impl on the editor side as `NativeBackend`),
and `gn-sdcpp` (the sd.cpp FFI, same worker protocol).

Tensor plumbing: **candle** (what qwen-image-rs, SceneWorks and mistral.rs chose; MIT/Apache).
candle is used for what it is good at (tensor types, safetensors, CPU reference paths, glue); the
hot path runs through `gn-kernels`, so candle is replaceable later without touching layers 2–5.
Burn is not used (training-oriented; dropped its candle backend; attention kernel open).

### 3.3 Routing and the day-0 lane

**Status (2026-10-20):** `generate_cmds::backend_for(s, template)` exists with `auto | native |
comfy`; under `auto` the families in `generate_cmds::NATIVE_AUTO_FAMILIES` (SAM 3.1 since
2026-10-19, Qwen-Image-2.1 since 2026-10-20) go to the native worker when `runtimePath` and
`modelsPath` are set and the worker's index lists the template, everything else to ComfyUI;
`native` sends everything to the worker, which serves SAM 3.1 and (built with
`native-diffusion`) Qwen-Image-2.1 text to image, edit and fill (every generative command
resolves its backend per template since 2026-10-20). `sdcpp` and the per-model `engineFor` map
come with the engines that need them.

A model can be served by up to three engines; a request picks one:

- `integrations.engine` = `auto | native | sdcpp | comfy` (default `auto`), and a per-model
  override map `integrations.engineFor` (`{"krea2-turbo": "native", …}`), both read by
  `generate_cmds::backend_for(template)` and `select_ml_cmds`.
- `auto` picks the first available in the order **native → sdcpp → comfy** for a given template,
  where "available" means: the engine implements the template's task for that model family, the
  weights the user has are loadable by that engine, the engine's health is OK, and (for native)
  the parity flag for that model is green in the shipped build.
- The Generations library records `engine` on every run; the model picker shows a small badge
  (⚡ native, ◇ sd.cpp, ☁ ComfyUI) so the user learns which path a model takes without being
  asked to care.
- New model from the community tomorrow: a workflow template (today's mechanism) makes it run on
  the sidecar; an sd.cpp definition (upstream's) makes it run Python-free; a `gn-models`
  definition makes it fast. Three lanes, one UI.

### 3.4 The engine is a worker process

The research marked embedded Python "unacceptable: GPU faults kill the editor". That is true of
any in-process CUDA, including ours. So the native engine (and sd.cpp) runs in **a child process
of the same executable** (`photocraft.exe --engine --channel <name>`), never in the editor's
address space:

- Crash isolation: a kernel fault, an OOM or a driver reset ends the worker; the editor reports
  "the engine stopped (CUDA error …), retrying" and respawns it. Documents are never at risk.
- VRAM hygiene: quitting the worker frees everything; "Purge generative models" is a restart.
- One code path for all three engines: the editor talks to a local channel whether the far end
  is `gn-engine`, `gn-sdcpp`, or the ComfyUI client proxy.
- Transport: a named pipe (Windows) / Unix socket, length-prefixed messages (serde + bincode
  or postcard), **pixels in shared memory** (a 1 MP RGBA request is 4 MB; copying it is fine,
  mapping it is better; start with copying and keep the message format able to carry a shm
  handle).
- Lifecycle: spawned on first generative command or at app start (preference), kept warm with
  models resident up to a VRAM budget, idle-unloaded after N minutes (preference), stopped on
  quit. Progress and cancel are messages; a cancel also tears down the worker if it does not
  answer within 2 s.
- On wasm: the native engine does not exist (`cfg(not(target_arch = "wasm32"))`, the crate is
  an optional dependency); the engine build stays green.

This is the one decision that is expensive to change later; §15 asks the owner to confirm it.

**Status (2026-10-19):** built as `genai-native/crates/gn-engine` (protocol, server, client,
fake engine, binary; child-process tests pass; a worker exits when its greeted client leaves).
Transport: loopback TCP with a token (one code path on all platforms, as the control channel),
behind a transport-agnostic protocol. The editor side is in: `photocraft_genai::native::
NativeBackend`, `photocraft --engine` / `photocraft-cli --engine` as the worker, the
`integrations.engine` preference and routing in `generate_cmds::backend`; the repository is
published (private) and pinned by rev. The worker serves the fake engine until §7.

### 3.5 Platforms: Windows and Linux first, macOS as a backend later

The owner's requirement (2026-10-18): cross-platform, **Windows and Linux first**, macOS after.
What that means for each layer, decided now so nothing is written Windows-only by accident:

- **Kernels (layer 1):** CUDA is the first `Kernels` implementation and it serves Windows and
  Linux identically (same `.cu`, same fatbins; nvcc + MSVC on Windows, nvcc + gcc/clang on
  Linux). macOS is a second implementation of the same trait (Metal via candle-metal or MLX;
  Apple's unified memory changes the residency policy, not the API). No kernel code may
  assume a platform; CI builds both.
- **Worker process (§3.4):** transport is a named pipe on Windows and a Unix domain socket on
  Linux/macOS behind one `Channel` type; the worker is spawned with the platform's "no console"
  flag on Windows and a process group on Unix so a cancel kills the whole tree; child discovery
  by the parent's own executable path on both.
- **Sidecar (§5.4):** `integrations.comfyPath` points at a ComfyUI checkout; the launcher finds
  the interpreter per platform (the portable `python_embeded` on Windows, a `venv/bin/python`
  or `python3` on Linux), builds the same flags, and reads readiness the same way.
- **Perception (layer 5):** ONNX Runtime ships CUDA and TensorRT providers for both; macOS gets
  the CoreML provider later. Engine caches are per platform + driver.
- **Loader (layer 4):** paths are `PathBuf`s, case-sensitive on Linux (the ComfyUI repack names
  are matched case-insensitively on purpose), mmap through `memmap2` on all three.
- **Packaging (§10):** a Windows installer and a Linux tarball + AppImage (`.deb` when the
  runtime folder is settled), each with its platform's CUDA redistributables and ORT; one
  `packaging/` script per platform, same manifest. Linux is also where the GPU build box runs
  CI and the nightly harness (§15), because the CUDA toolchain is simpler there.
- **Gates:** every phase's "done" includes a Linux build of `genai-native` and of PhotoshopEX
  with the engine, and the harness run on both when a Linux GPU exists; a phase does not close
  on one platform.
- **Dev machines:** this PC (Windows 11, RTX 5090) and the recommended second box, which
  should be **Linux** (Ubuntu 24.04, CUDA 13) so both platforms are exercised daily.

---

## 4. Repository and crate layout

**Separate repository `genai-native`** (recommended, §15): `C:\Users\5090\Documents\genai-native`,
MIT OR Apache-2.0, its own CI, consumed by PhotoshopEX as a git dependency pinned by revision
(path dependency during development). Reasons: it is useful to others without the editor; its
licence story (CUDA DLLs, ORT, NVIDIA RTX licence texts) is contained; its CI needs a GPU runner
and PhotoshopEX's does not; agents can work it without touching the editor's tree.

```
genai-native/
  Cargo.toml                 workspace
  crates/gn-kernels/         src/, cuda/ (*.cu, CUTLASS as a submodule), build.rs (nvcc → fatbin; prebuilt fatbins committed for CI without nvcc)
  crates/gn-blocks/
  crates/gn-models/
  crates/gn-weights/
  crates/gn-perception/
  crates/gn-harness/         fixtures/, references/ (ignored: generated locally), src/bin/gn-bench
  crates/gn-engine/          the worker: protocol, residency, budget; src/bin/gn-engine (standalone for tests)
  crates/gn-sdcpp/           FFI to a pinned stable-diffusion.cpp commit (git submodule), feature-gated
  docs/                      ARCHITECTURE.md (the five layers), PARITY.md (thresholds), PORTING.md (how to add a model), KERNELS.md
  NOTICE                     every ported file's origin and licence
```

**In PhotoshopEX:**

- `crates/genai/src/native.rs`: `NativeBackend` (the editor side of the channel) implementing
  `GenerativeBackend`; `crates/genai/src/routing.rs`: `backend_for`, availability, the parity
  flags; `crates/genai/src/engine_proc.rs`: spawn/respawn/stop the worker (also the hidden
  ComfyUI sidecar's launcher, §5.4).
- `apps/photocraft/src/main.rs`: the `--engine` entry point (one `if` at the top; the worker
  main lives in `genai-native`).
- `crates/engine/src/generate_cmds.rs` and `select_ml_cmds.rs`: `backend()` → `backend_for(...)`;
  `generate.engine.*` commands (§12.3). Surgical edits only.
- `crates/ui-egui/src/prefs_ui.rs`: the Engine section; `model_picker.rs`: the engine badge;
  `jobs_ui.rs`: the engine status dot.
- Layers (`xtask/src/layers.rs`): `genai` stays L4; `genai-native` is external, so nothing
  changes in the layer map; the wasm build excludes it by `cfg`.

---

## 5. Phase 0: ground truth (week 1)

**Goal.** Numbers on this machine for every claim the plan rests on, the harness that will judge
all later work, and the sidecar so the product stops looking like a lab while the engine is
built. Nothing in this phase is a port.

### 5.1 The parity harness, first

`gn-harness` (§12.1 has the full spec). Deliverables this week: the fixture set (12 prompts × 2
seeds × 3 sizes for text-to-image; 8 edit cases with references; 8 fill cases with masks; the 8
bench pictures for SAM with text/point/box prompts), the reference generator (a script that runs
them through ComfyUI and saves PNGs + per-step latents where the graph allows), the metrics
(PSNR, LPIPS via a small ONNX VGG, cosine on latents, mask IoU/boundary-F for SAM), and the
report format (one Markdown table per model per engine, committed under `benchmarks/`).

### 5.2 Bring up qwen-image-rs on the 5090

Clone (do not fork yet) to `C:\Users\5090\Documents\_eval\qwen-image-rs`. Build with the arch
flag for sm120 (its kernels are sm80+/sm89; expect `-gencode arch=compute_120a` changes and
possibly SageAttention2's sm120 variant). Run its Qwen-Image-2.1 text-to-image on the harness
prompts; record s/step, s/image, cold start, VRAM, and parity against the ComfyUI reference.
Read `HANDOVER.md` end to end and write `docs/notes/qwen-image-rs-review.md`: what to keep
(kernels, parity method, VAE), what to restructure (the monolith into layers), what is missing
(edit, references, LoRA, masks, repack loading, CUDA graphs).

Needs: **Qwen-Image-2.1 bf16 safetensors + its Qwen3-VL 8B encoder (~30 GB)**, a test download
the owner approves; nothing of it ships.

### 5.3 Measure sd.cpp on the 5090

Build `stable-diffusion.cpp` with CUDA (MSVC + CUDA 13; use the prebuilt sm120 `sd-cli` from the
research table if the build fights back). Run Krea 2 Turbo (GGUF Q8_0 and the official
safetensors) and Qwen-Image-2.1 (GGUF Q8_0, the `leejet` conversion) on the harness prompts;
record the same columns; compare with ComfyUI fp8 and NVFP4. Needs **Krea 2 Turbo GGUF Q8
(~12 GB) and Qwen3-VL-4B GGUF (~4 GB)**, the second test download to approve.

Decision rule: sd.cpp within **1.3×** of ComfyUI `best` on a model → it is that model's
Python-free lane in Phase 5 without further work; **1.3–2×** → still the lane, flagged for the
kernel track; **>2×** → the sidecar stays the fallback for that model and sd.cpp is not packaged.

### 5.4 The hidden ComfyUI sidecar

Not the destination, but a week of work that removes the console window today and builds the
process-management code the native worker reuses (§3.4): `engine_proc.rs` spawns ComfyUI from
`integrations.comfyPath` with the flags from `benchmarks.md`, `CREATE_NO_WINDOW`, stdout to a
rotating log in the config dir, readiness by polling `/system_stats`, stop on quit or keep warm,
port conflict handling, "engine stopped" notices; `generate.engine.start/stop/status`; the
status dot; Preferences › AI Integrations › Engine. Live script `sidecar-live.ps1` (spawn → ready
→ fill → stop).

**Status (2026-10-18/19):** §5.4 done and live (dev log); §5.1 built in the `genai-native`
repository (`gn-harness`, `gn-bench`; the ComfyUI baseline is recorded in `benchmarks.md` §8);
§5.3's first row is in (sd.cpp Krea 2 Turbo Q8: 1.45× ComfyUI's step, the "lane, flagged for
the kernel track" band; the `--engine sdcpp` harness runner and the Qwen-Image-2.1 row follow);
§5.2 is attempted on this PC first (CUDA 13.4 + VS 2022 Build Tools installed 2026-10-19, nvcc
`sm_120a` smoke test green), WSL2 or the Linux box as the fallback (qwen-image-rs's build script
is POSIX-leaning and wants a Linux cuDNN wheel, see the review note); §5.3 runs from sd.cpp's
prebuilt Windows CUDA 12.8 zip (arch 120).

### 5.5 Done when

A `benchmarks.md` §8 table exists with, per model: ComfyUI fp8 / NVFP4, qwen-image-rs (2.1 only),
sd.cpp (both), with s/step, s/image warm, cold start, VRAM, PSNR/LPIPS vs reference. The
sidecar works live. The review note and the harness are committed. The owner has read the
numbers and confirmed §15's decisions.

---

## 6. Phase 1: SAM 3.1 native (weeks 1–3, parallel with Phase 0)

**Goal.** Select Subject, Select by Text, the object finder, box prompts and Refine mode run
in-process at tens of milliseconds, with ComfyUI off, through `gn-perception`; the probe tint
appears as the pointer moves rather than after a rest.

**Status (2026-10-19): parity reached in the harness.** `gn-perception` runs SAM 3.1 through
`ort` (runtime loaded from a user-named directory, CUDA + cuDNN preloaded from named
directories), from **our own export of ComfyUI's modules** (`genai-native/tools/export_sam31.py`,
five fp32 graphs, no community export and no Hugging Face token needed: the user's existing
`sam3.1_multiplex_fp16.safetensors` is the source), and `sam31.rs` reproduces the `SAM3_Detect`
node's text, point and box routes including its crop refinement. 16/16 SAM cases at bounds IoU
0.99–1.00 against the ComfyUI references (`bench/sam-ort-comfy-compare-2026-10-19.md`). **Wired
into the app the same day**: `PerceptionEngine` serves the worker, `integrations.runtimePath` /
`modelsPath` + `engine = native` make `select.byPoint/byText` native (the fox's masks match the
references exactly; with the fp16 graphs, ORT's fused attention and the feature maps kept on the
device: 10 ms per prompt on an embedded picture, 66 ms for a new picture, against the ComfyUI
node's 94 ms per point). Not yet: a CUDA graph for the trunk, the TensorRT provider; see 6.2 and
the dev log.

### 6.1 Artifacts

- ONNX graphs for the three paths SAM 3.1 exposes: text (PCS: image encoder + text encoder +
  detector), points (with negatives) and box (tracker). Start from the community exports (the
  yimin-pan / dataplayer12 scripts; the pagarcia repo documents the tracker's RoPE pitfalls);
  pin the transformers version they need. Publish our export script and the SHA-256 of the
  resulting files in `gn-perception/docs/EXPORT.md`. **The user obtains the weights** (gated HF
  repo, their token); the app offers a one-click "export to ONNX" that runs a bundled script
  **only if Python is present**, and otherwise points at the published ONNX files from a
  community mirror the user chooses. (This is the one place a Python step survives, upstream of
  the app, once per machine; the plan accepts it because Meta ships no export.)
- Engine cache: `ort`'s TensorRT provider builds an engine per GPU + runtime version on first
  use (~1–3 min for SAM); cache under the config dir, keyed by file hash + driver + ORT version;
  the CUDA provider is the fallback when TensorRT is unavailable (slower, no build step).

### 6.2 Implementation

- `gn-perception::sam3`: `load(dir) -> Sam3`, `embed(image) -> ImageEmbedding` (cached per
  document revision, the expensive part), `detect_text(emb, phrase, threshold)`,
  `segment_points(emb, positives, negatives)`, `segment_box(emb, rect)`, all returning instance
  masks at image resolution + scores; pre/post-processing (resize/pad to the export's size,
  normalisation, mask upsampling, NMS) in Rust.
- `NativeBackend::run` for `Task::Segment`: maps the existing template ids
  (`sam3.1/segment`, `segment-point`, `segment-box`) and the existing `params`
  (`threshold`, `points`, `negatives`, `bx,by,bw,bh`) onto the calls above, so
  `select_ml_cmds.rs` does not change except for `backend_for`. The image embedding cache makes
  the probe 10–30 ms after the first call on a picture.
- `object_select.rs`: when the engine reports `native`, lower `PROBE_DELAY` to 0.08 s and keep
  the cell logic (the finder follows the pointer).
- Fallback: no ONNX files → the ComfyUI route as today → the classical finder.

### 6.3 Tests

`gn-perception` unit tests on a 64×64 synthetic ONNX model (a 2-op graph committed as a
fixture) that exercises the pre/post pipeline; `select_ml_cmds` tests against a `FakePerception`
(the engine-side fake, like `FakeComfy`) so the command layer is covered without weights; one
`#[ignore]` live test with the real ONNX files behind `PHOTOCRAFT_SAM3_DIR`. Harness: mask IoU
≥ 0.97 and boundary-F ≥ 0.95 against the ComfyUI SAM 3.1 route on the 8 bench pictures, all
prompt kinds; latency table.

### 6.4 Done when

`object-select-live.ps1` passes with ComfyUI stopped; probe ≤ 30 ms warm, first call on a
picture ≤ 150 ms (embedding), model load ≤ 2 s; `benchmarks.md` row; dev log; commit.

---

## 7. Phase 2: the engine skeleton and the first definition (weeks 2–5)

**Goal.** `genai-native` exists with its five layers, the worker process, the loader, and
Qwen-Image-2.1 **text-to-image at parity** with the ComfyUI reference, built on the pieces of
qwen-image-rs that survive review.

### 7.1 Fork and restructure

**Status (2026-10-20):** imported as `genai-native/crates/qir` by `git subtree` (history and
attribution kept, Apache-2.0, root `NOTICE`), building on Windows with CUDA 13.4 for sm_120 and
running on the 5090 at 0.116–0.14 s/step (the authors' 4090 speed; ComfyUI's INT8 path is ≈
0.15); `gn-bench --engine qir` drives it. Added there: ComfyUI's sampling conventions
(`qir::comfy`: PyTorch's CPU generator reproduced bit for bit, the `simple` schedule over the
model's shift-0.69 sigmas, the bf16 timestep rounding; `generate --comfy`), non-square pictures,
and a resident library pipeline (`qir::pipeline`: encoder, DiT and VAE loaded once, progress,
cancellation). The restructuring into `gn-kernels` / `gn-blocks` / `gn-models` is still to do;
the pipeline is what the worker hosts meanwhile.

Fork `umag/qwen-image-rs` into `genai-native` (keep history and attribution in `NOTICE`). Move
its CUDA kernels into `gn-kernels` behind the `Kernels` trait (bf16 GEMM, INT8 ConvRot GEMM,
SageAttention v1/v2, fused LayerNorm/RMSNorm/SwiGLU/RoPE, the cuDNN convs for the VAE); its
DiT/encoder/VAE code into `gn-blocks` (generic blocks) and `gn-models/qwen_image_21.rs` (the
definition); its safetensors loading into `gn-weights`. Compile fatbins for `sm_120a` first,
`sm_89` second; commit prebuilt fatbins so CI and other agents build without nvcc.

### 7.2 The loader

`gn-weights`: safetensors with a hardened header parser (bounded sizes, no allocation from
untrusted lengths, fuzz target); GGUF reader (header, tensor info, the quant types we accept:
Q8_0, Q4_K_M, Q6_K first; dequantize to bf16 on load until quantized kernels exist); key maps
per model for **official**, **ComfyUI repack** (`qwen_image_2.1_int8_convrot.safetensors`,
`qwen3vl_8b_int8_convrot.safetensors`, the dedicated VAE) and **GGUF** layouts; a `ModelIndex`
that scans a folder tree (the ComfyUI layout `diffusion_models/ text_encoders/ vae/ loras/` and
flat folders) and reports what fits which slot, feeding `generate.models`.

**Status (2026-10-19):** the readers, the index and family identification from the tensors
exist (`genai-native/crates/gn-weights`, `gn-inspect`); all 16 files of the owner's install are
identified and their repack conventions recorded in `genai-native/docs/MODELS-ON-DISK.md`. Key
maps and dequantisation come with the first definition (§7.4); ranges are read on demand
through a seekable file rather than an `unsafe` mmap.

### 7.3 The worker and the editor side

`gn-engine`: the protocol (hello/health, load, run with progress stream, cancel, free, index),
model residency with a VRAM budget read from the device minus a reserve, the request lifecycle
mirrored on `Request`/`Response`; `src/bin/gn-engine` for standalone tests. PhotoshopEX:
`engine_proc.rs` (shared with the sidecar), `NativeBackend`, `routing.rs` with `auto` order and
the per-model parity flags (a `const` table shipped in the build, plus the preference override).

**Status (2026-10-20):** `gn-diffusion` is the worker-side engine for `qwen-2.1/image` (task
`image`), and since the evening `qwen-2.1/edit` and `qwen-2.1/fill` (§8): it loads the snapshot `<models>/qwen-image-2.1/` (Hugging Face's layout) on the first
request through `qir::pipeline` and keeps the three models on the device; the INT8 DiT cache
lives in `<models>/.qir-cache/`. `gn_engine::Multi` puts it and the perception engine behind one
worker (each request goes to the engine whose index lists the template). PhotoshopEX's worker
serves both when built with `--features native-diffusion` (nvcc needed; the SAM route builds
without it), and puts the runtime folders on the worker's `PATH` so the CUDA libraries the
kernels link are found. The parity table is `generate_cmds::NATIVE_AUTO_FAMILIES`; the
preference override map is still to come.

### 7.4 Parity and speed for Qwen-Image-2.1 text-to-image

Harness thresholds (§12.1) met against the ComfyUI reference at bf16; then INT8 ConvRot +
SageAttention on sm120 measured against ComfyUI `best`. The definition's size is a gate too: if
`qwen_image_21.rs` exceeds ~600 lines, something belongs in a block.

**Status (2026-10-20):** at parity. ComfyUI's noise (PyTorch's CPU generator, bit for bit),
its `simple` schedule over the model's own sigma table (`ModelSamplingFlux`, the exponential
shift with `0.69` over 10 000 timesteps: the 2.1 model is a `FLUX`-type model in ComfyUI, and
the day's detour was a table built from the other class), its timestep rounding and its prompt
tokens: the fox agrees with the ComfyUI route at **39.9 dB** (21 dB with the wrong table); the
28-case numbers are in the dev log and `benchmarks.md` §8. The text encoder runs its attention
in f32 (2.7 % from the bf16 oracle; ComfyUI's INT8 encoder is 8.8 % from it). `auto` routes
`qwen-2.1/image` to the native worker since the same day (`NATIVE_AUTO_FAMILIES`).

### 7.5 Tests and done

Block parity tests with tiny committed reference tensors (generated once from diffusers, a few
MB), loader tests on hand-made safetensors/GGUF files, protocol tests (the worker as a child
process in the test, including kill-and-respawn), `FakeEngine` on the editor side for the UI
and command tests. Done when `generate.image` with the 2.1 template routes `native` under
`auto` on this machine, at parity, and `benchmarks.md` shows the number next to ComfyUI's.

---

## 8. Phase 3: Qwen-Image-2.1 editing, Fill and Expand (weeks 5–7)

**Goal.** Everything `generate.*` does with 2.1 today (image, edit, fill, expand, matte via the
RGBA output, references) runs natively.

- **Reference conditioning** (the 2.1 edit interface, 1–10 ordered references, no mask input):
  the conditioning recipe in the definition (reference latents concatenated along the sequence
  with their own positional ids, the prompt template the encoder expects). Validate against the
  SceneWorks and diffusers implementations and the ComfyUI `qwen-2.1-edit` graph.
- **Fill and Expand** the way the engine already does them for the ComfyUI route: the composite
  around the selection as the reference/edited image, the model's whole-picture output
  composited to the selection through the existing soft dithered edge (`edge: soft`) and the
  layer mask; Expand's edge-replicating prefill and the "empty prompt continues the scene"
  finding apply unchanged. No new UI.
- **Transparent output / matte**: the 64-channel RGBA VAE path (already in qwen-image-rs's
  VAE? verify; SceneWorks has it), feeding `generate.image {transparent}` and the research-opt-in
  matte route.
- **VRAM budget**: 2.1's DiT (7B) + 8B encoder + VAE at bf16 is ~30 GB; INT8 brings the DiT
  under 8 GB; the encoder is loaded for the prompt and may be evicted before sampling when the
  budget says so (qwen-image-rs keeps it resident; we make it a policy).
- **Progress and cancel** per step; previews every N steps (a cheap latent → RGB projection) for
  the task bar.
- Harness: the 8 edit and 8 fill cases, PSNR/LPIPS vs the ComfyUI route *and* a human look at
  the contact sheet (parity on edits is softer: same seed can diverge after sampler differences;
  the threshold is LPIPS, not PSNR). Done when `auto` routes 2.1 edit/fill/expand to `native`
  and the Variations strip shows no difference in behaviour.

**Status (2026-10-20): done but for the previews.** Edit, fill, expand, matte and transparent
output native and under `auto`. The VAE encoder, the Qwen3-VL vision tower, the multimodal
encoder and the DiT's reference sequence are in `qir`, each checked against ComfyUI's running
objects (dev log); the prefix cache (ComfyUI's `QwenImage21Cache`) makes an edit step cost a
text-to-image step (3.8 s of sampling for a 1024² edit). `gn-diffusion` serves `qwen-2.1/edit`,
`fill`, `expand` and `matte`; through the app's own CLI the edit suite is 8 of 8 and the fill
suite 8 of 8 at the gate against the ComfyUI route, the matte at soft IoU 0.992. Expand is the
in-context way: the picture placed at its own position inside the canvas as the reference
(`RefImage::offset`), the whole canvas as the target, the editor's mask confining the result;
the new `qwen-2.1/expand` template carries it (and `expandMargin: 1`). The VRAM budget policy
exists (`evict_text_encoder`: the encoder leaves after the prompt when the cache would not
fit). Open: previews for the task bar.

---

## 9. Phase 4: Krea 2 Turbo (weeks 6–10)

**Goal.** The second definition proves the block library: Krea 2 Turbo text-to-image and style
reference natively, at parity, faster than ComfyUI NVFP4 once the kernel track delivers FP8.

- **Architecture reference**: ComfyUI's model code for Krea 2 (check the file's licence header;
  ComfyUI is GPL except files marked otherwise) and `stable-diffusion.cpp`'s C++ (MIT) are the
  readable references; diffusers' `Krea2Pipeline` is the parity oracle. Port clean-room into
  `gn-models/krea2.rs`: the DiT config and any block Krea 2 has that 2.1 lacks (expect
  differences in the modulation and the positional scheme; each new block gets its parity test).
- **Components**: Wan 2.1 VAE (new VAE family in `gn-blocks`), Qwen3-VL **4B** encoder (the
  same family as 2.1's 8B: a config, plus the image-grounded prompt formatting Krea 2's style
  reference uses), the Turbo few-step schedule.
- **LoRA loader** in `gn-weights` + fused application in `gn-blocks` (needed later for the
  Lightning tiers of 2511 and the Angles LoRA; built here because Krea 2 Raw users will bring
  LoRAs).
- **Precision**: the user's files may be bf16, the official NVFP4 checkpoint, or GGUF; until the
  NVFP4 GEMM exists (§11) the NVFP4 file is dequantized on load (correct, not fast) and the
  routing keeps Krea 2 on ComfyUI NVFP4 if that is faster. The routing rule is "faster at
  parity", per model, measured.
- Harness: the 12 × 2 × 3 text-to-image set + 6 style-reference cases. Done when the definition
  is ≤ 600 lines, blocks added are tested, parity holds, and `benchmarks.md` has the row.

**Status (2026-10-20 night): text to image ported, at the model's own level of agreement,
faster than ComfyUI.** The weights come from ComfyUI's fp8 repacks through
`qwen-image-rs convert-fp8` (bf16 safetensors; BYOM). In `qir`: the Qwen3-VL-4B config and
tapped hidden states (`forward_taps`, Krea 2's twelve), the single-stream DiT
(`model/krea2.rs`, ~560 lines: GQA with the sigmoid gate, AdaLN-single, `txtfusion`, Flux's
3-axis RoPE), the Wan 2.1 VAE decoder for one frame (`model/wan_vae.rs`), the shift-1.15
schedule, `Krea2Pipeline`; `gn-diffusion` serves `krea2-turbo/image` with one family resident
at a time. Each piece was checked against ComfyUI's running objects (conditioning 1 %, one
DiT forward 1.6 %, VAE 0.24 %). On the 28 cases the native route is at 16–17 of 28 against
ComfyUI's fp8 and bf16 routes, which agree with each other on 20 of 28: an 8-step Turbo at CFG
1 is sensitive to its arithmetic (`benchmarks.md` §8). Speed: 3.5–4.9 s a picture against
ComfyUI's 4.0–4.7 (through the app, resident: 4.3–4.85 s). With the bf16 DiT the native route
reaches 21 of 28 against the bf16 route (ComfyUI's fp8 route: 20) at 8.6–15 s. Krea 2 stays on
ComfyUI under `auto` (not faster at the precision ComfyUI reaches); `native` serves it. Open:
the fp8 GEMM (§11) to keep both speed and precision, the style reference (6 cases), LoRA loading, the
NVFP4 file (dequantised on load would work through the same converter path; the NVFP4 GEMM is
§11), the definition's ≤ 600-line gate is met but the blocks are not yet in `gn-blocks` (§7.1).

**Status (2026-10-27): text to image under `auto`, faster than ComfyUI at its own agreement.**
The worker reads ComfyUI's `fp8_scaled` files as they are (`qir::fp8`, `genai-native`
`6354b13`): the five fp8 linears of a block through cuBLASLt's FP8 GEMM with ComfyUI's recipe,
the three the file marks `full_precision_matrix_mult` dequantized to bf16, the encoder's layers
in full precision as ComfyUI encodes; exact attention through our own flash kernel (§11) and
Krea 2's small ops fused. 19 of 28 at the gate against the ComfyUI route (20 against its bf16
route; ComfyUI's two routes agree on 20), 3.2 s of sampling and 4.0 s a picture through the
app against 4.0–4.7 (`benchmarks.md` §8). `krea2` is in PhotoshopEX's `NATIVE_AUTO_FAMILIES`.
The model folder is `<models>/krea2-turbo/`: `transformer/` = ComfyUI's
`krea2_turbo_fp8_scaled.safetensors`, `text_encoder/` = `qwen3vl_4b_fp8_scaled.safetensors`,
`vae/` = `wan_2.1_vae.safetensors`, `tokenizer/tokenizer.json` (Qwen3-VL's); the bf16 files of
`convert-fp8` still load (ConvRot, as before). Open: the style reference (6 cases), the NVFP4
file, AnyPaint's reference path.

**LoRA loading (2026-10-26): done as a runtime side path**, in `qir::lora` (`genai-native`).
For every linear of both DiTs, `y += s · B (A x)` with `A (r, in)`, `B (out, r)` in the model's
dtype and `s = strength · alpha / r`; the base weights stay as loaded, INT8 ConvRot included
(the merged q|k and gate|proj GEMMs add their deltas after the fact, the fused SwiGLU's out
delta forms `silu(g)·p` only when an adapter asks), so adapters are set per request on the
resident model (`QwenImageDit::set_loras`, `Krea2Dit::set_loras`, a `LoraSlot` on each linear)
and a strength is a number. Files: `lora_A/lora_B` and `lora_down/lora_up`, `alpha`, the
`diffusion_model.` / `transformer.` prefixes, ComfyUI's fused `img_mlp.gate_up` of 2.1 split
as its own loader does (gate first). The 2.1 tails (modulation, timestep embedder, img_in,
norm_out, proj_out) take adapters too (Viggle's Turbo LoRA touches them); `txt_in` does not
yet. The worker (`gn-diffusion`) reads `params.loras` (`[{file, strength}]`, a file name under
`<models>/loras/` or an absolute path, six files cached on the device) and the editor's fill,
edit, image and removal commands take `loras` (`generate_cmds::Tuning`); a ComfyUI template
ignores the list and carries its own LoRA nodes. Unmatched module names are logged, so a file
for another model is visible. Parity against ComfyUI with the same file is the next
measurement (the Consistency LoRA on a 2.1 edit, AnyPaint on Krea 2 once its reference path
exists); the owner's installed set is in `models.md`.

**Krea 2 inpainting and outpainting natively (added 2026-10-21, from AusBoss's clothing
workflow, `docs/comfy/research/notes/ausboss-krea2-clothing-workflow.md`).** Krea 2 has no mask
input and no reference slot; the community's **AnyPaint** LoRA (`krea2_anypaint_rank32`,
yijunwang2, Krea 2 Community License: BYOM, never a default) gives it both with a recipe the
engine can serve once the LoRA loader exists: the masked canvas as **one reference of at most
384 px** on its long edge (multiple of 16), the masked pixels of that reference replaced by the
**median colour** of the known ones, the reference's tokens **registered over the whole canvas**
in the DiT's attention with their keys and values computed **once per run** (the 2.1 pipeline's
prefix cache is the same idea), the reference also shown to the vision tower so the text
encoding is grounded, full-denoise sampling with the unmasked latents **restored after every
step** and a **32-px band** at the mask's edge left for the model, 8 steps at CFG 0–1. The
in-context pair (reference stitched beside the hole, the noise mask on the hole's half) needs
no model change at all and is the cheap first step. Parity oracle: the AusBoss nodes on ComfyUI.

---

## 10. Phase 5: packaging without Python (weeks 9–12)

**Goal.** A build a stranger installs on a clean Windows 11 PC with an RTX card, points at a
models folder, and uses, with no Python, no ComfyUI, no console.

- **Runtime folder**: `cudart`, `cublas`/`cublasLt` (if the kernels use them; CUTLASS paths do
  not), `cudnn` (VAE convs; consider dropping cuDNN for our own conv kernel to save 700 MB),
  ONNX Runtime + CUDA/TensorRT providers, the fatbins. Licence texts for each (CUDA EULA
  redistributables, NVIDIA RTX licence, MIT/Apache) in `THIRD_PARTY/`; NOTICE for ported code.
  Size target under 1 GB; measure.
- **Installer**: the existing packaging scripts (`packaging/`) gain the runtime folder, the
  `--engine` entry, and an sd.cpp DLL when Phase 0's gate admitted it. No model downloads.
- **First run**: Preferences › AI Integrations › Models folder (defaults to ComfyUI's if one
  exists, else `%LOCALAPPDATA%\PhotoshopEX\models`); the model index shows what is installed per
  slot with its engine badge; a "Get models" page lists the files each feature wants with their
  Hugging Face pages and licence summaries from `models.md` — links, never downloads.
- **GPU detection**: `gn-kernels` reports the device, compute capability and VRAM; a non-NVIDIA
  or pre-Ampere card disables the native engine with a sentence and keeps the rented lanes.
- **The engine page**: status (stopped / starting / ready / failed with reason), resident models
  and VRAM, "Purge", "Restart engine", the log, the per-model engine override table.
- **Retire the console**: with the sidecar from Phase 0 and the native engine, no visible
  process remains; ComfyUI becomes an optional engine for users who have it.
- Done when the clean-machine install is exercised live (a second user account on this PC, or a
  VM with GPU passthrough is not available, so: a fresh Windows user profile with the models
  folder copied in), and `walkthrough-live.ps1` passes with `integrations.engine = native`.

**Status (2026-10-27): started.** ✅ The editor starts on a PC without CUDA: a
`native-diffusion` build delay-loads `cublas64_13`, `cublasLt64_13` and `curand64_10` (the
apps' `build.rs`; `dumpbin` shows no load-time CUDA import, and `photocraft-cli --version` runs
with no CUDA on `PATH`). ✅ The worker checks its runtime before serving the diffusion models
(`native::missing_cuda_libraries`: a sentence in its log, and the editor falls back to ComfyUI
under `auto`, instead of a fault on the first delay-loaded call). ✅ The runtime folder:
`packaging/windows/runtime.ps1` assembles it from a CUDA toolkit, an ONNX Runtime GPU build and
cuDNN 9 (no downloads) with `THIRD_PARTY.txt`; **1.33 GB** (CUDA 581 MB, of which cuBLASLt
493; ONNX Runtime 205; cuDNN 542), over the 1 GB target (cuDNN is ONNX Runtime's, for SAM; our
own conv path would drop it). Exercised: Krea 2's 28 cases through the CLI and worker with only
that folder and no toolkit on `PATH`, 19 of 28 at 4.0 s a picture. ✅ The app uses a `runtime`
folder next to the exe when Integrations › Runtime folders is empty
(`Integrations::runtime_folders`); `package.ps1 -Native` builds with the engine and ships the
folder in the portable zip. ⬜ The models-folder first run and "Get models" page, the
clean-profile install, the MSI.

**Status (2026-10-27 evening): GPU detection and the engine page.** ✅ The worker asks the
driver about the card before it touches CUDA (`qir::device::probe`: `nvcuda.dll` looked for
first, since cudarc aborts the process when the driver library is missing; then the name, the
compute capability and the memory, with no context made) and the diffusion engine refuses a
card older than the kernels the build carries (`device::check` against `CUDA_COMPUTE_CAP`, 12.0
for this build) with a sentence: it lists no templates, so `auto` keeps those models on ComfyUI,
and the sentence is in its health (`Health.detail`, new in `gn-engine`'s protocol, defaulted so
older workers still parse). Missing runtime libraries give the same kind of sentence. ✅
`generate.engine.status` carries the worker's state under `native`
(`off|unconfigured|stopped|starting|running|ready|failed`, the card's name, total and free
memory, the loaded model folders, the reason), asked without waiting for a running request and
without starting a worker (`NativeBackend::status`); `generate.engine.start / stop / restart`
take `{engine: "native"}` and `generate.engine.purge` unloads the models of both engines. The
worker's stderr is its log (`photocraft-native-engine.log` in the temp folder, `RUST_LOG=info`).
✅ The status-bar dot follows the engine generative commands go to and its menu has a section
per engine; Preferences › AI Integrations shows the native engine's lines with Start / Stop /
Purge models / Show log. ⬜ The per-model engine override table.

---

## 11. Phase 6: the kernel track (from week 4, ongoing)

**Goal.** The speed the owner is after. Each kernel lands only with a harness number and a
parity check; each is written once in `gn-kernels` and offered upstream where it fits.

| Order | Kernel | Why | Gate |
|---|---|---|---|
| 1 | SageAttention2 for sm120 (verify the community sm120a build's approach, write ours) | attention is ~40 % of DiT time; the only kernel lever that applies to every model | ≥ 1.25× step time vs SDPA on 2.1 and Krea 2, cosine ≥ 0.999 |
| 2 | FP8 (e4m3) GEMM with per-block scales (CUTLASS sm120) | the format the user's `fp8` files already are; Blackwell tensor cores | ≥ 1.4× vs bf16 GEMM, parity thresholds held |
| 3 | NVFP4 GEMM (CUTLASS block-scaled FP4, the format Nunchaku/ModelOpt files use) | the +18 % we already see on Krea 2 through ComfyUI, without dequantizing | Krea 2 NVFP4 native ≥ ComfyUI NVFP4 |
| 4 | CUDA graphs for the sampling loop (fixed shapes per size bucket) | removes launch overhead at few-step Turbo schedules | ≥ 10 % at 4–8 steps |
| 5 | Fused QKV + RoPE, fused AdaLN modulation | long tail | measured |

**Status (2026-10-27).** FP8 (row 2): done through cuBLASLt rather than CUTLASS (`qir::fp8`):
313–568 TFLOPS at Krea 2's shapes against 186–220 for bf16, per-tensor scales as ComfyUI's
files carry them. Attention: an exact bf16 flash kernel of our own (`qir::flash`, FA2's
algorithm, PV in fp16, ≈ 120 TFLOPS) became Krea 2's default because SageAttention2 cost three
to four cases of agreement on an 8-step Turbo; row 1's SageAttention stays for Qwen-Image-2.1.
Row 5 for Krea 2 (`qir::krea2_fused`: modulated RMSNorm, head norm + RoPE, SwiGLU, gates): 4.54
→ 3.20 s of sampling. Rows 3 and 4 open.

Upstream policy: kernels that are general (FP8/NVFP4 GEMM, SageAttention sm120) are offered to
ggml's CUDA backend (so sd.cpp and our rented lane speed up too) and to candle; we keep our own
copies either way. A fork of ggml/sd.cpp is justified only to carry a patch upstream will not
take, and then only of the kernel files.

---

## 12. Cross-cutting specifications

### 12.1 The parity harness (`gn-harness`)

- **Fixtures** (committed, small): prompts (12 text-to-image across subjects, styles and text
  rendering; 8 edit instructions with 1–2 references each; 8 fill prompts with masks on the
  bench pictures; 6 Krea 2 style references), seeds (2 per case), sizes (1024², 1344×768,
  768×1344), SAM cases (8 pictures × text/point/negative/box prompts).
- **References** (generated locally, ignored): ComfyUI outputs via the existing templates at
  Quality `best` with the same seeds; where a graph exposes it, per-step latents; diffusers
  outputs for the block-level tests (tiny tensors, committed).
- **Metrics**: text-to-image: PSNR and LPIPS between engines at equal seed (thresholds: PSNR ≥
  30 dB *or* LPIPS ≤ 0.08 — samplers may legitimately diverge, so the softer of the two decides,
  and a contact sheet is always produced for a human); block level: cosine ≥ 0.9995 and max-abs
  error ≤ 1e-2 bf16; VAE: PSNR ≥ 50 dB; SAM: IoU ≥ 0.97, boundary-F ≥ 0.95; latency: s/step,
  s/image warm, cold start, peak VRAM; determinism: same seed twice → identical bytes on the
  native engine.
- **Runner**: `gn-bench run --engine native|sdcpp|comfy --model … --suite t2i|edit|fill|sam
  --out benchmarks/<date>-<commit>/`; writes the Markdown table, the contact sheet PNG, and a
  JSON the dev log links. `gn-bench compare a/ b/` diffs two runs.
- **CI**: the GPU box (§15) runs the suite nightly on `main`; PhotoshopEX CI runs only the
  fakes.

### 12.2 Failure taxonomy (engine → user sentence)

| Kind | Example | Shown as | Recovery |
|---|---|---|---|
| `EngineUnavailable` | no NVIDIA GPU, driver too old, runtime DLL missing | "The native engine needs an NVIDIA RTX card with driver ≥ X; using ComfyUI" | routing falls through |
| `ModelMissing` | slot has no loadable file | "Krea 2 Turbo is not installed (looked in …)" + Get models link | user action |
| `ModelUnreadable` | corrupt safetensors, unknown GGUF quant | "… could not be read: <reason>" | user action |
| `OutOfMemory` | budget exceeded mid-run | "Not enough VRAM for this size; try 1 MP or Balanced" | automatic retry at the next size cap once |
| `EngineCrashed` | worker exited | "The engine stopped (<signal/code>); restarting" | respawn, one automatic retry of the run |
| `Cancelled` | user | nothing | — |
| `Parity`/`Unsupported` | a template the engine cannot serve | silent | routing picks the next lane |

All map onto the existing `GenFailureKind` labels where one exists (plan-artcraft §8.2).

### 12.3 Commands and preferences

Commands (all with `CommandSpec.params` docs, control-protocol rows, fake-backed tests):
`generate.engine.status → {native, sdcpp, comfy: {state, device, vram, resident: [...]}}`,
`generate.engine.start {engine}`, `generate.engine.stop {engine}`, `generate.engine.restart`,
`generate.engine.index {dir?} → model index`, `generate.engine.bench {suite, engine}` (runs the
harness from the app; CLI-only in practice). `generate.models` gains `engine` per entry.

Preferences (`integrations.*`, each read in the same change): `engine` (`auto|native|sdcpp|comfy`,
default `auto`), `engineFor` (map), `modelsDir`, `engineStartOnLaunch` (default false),
`engineIdleUnloadMinutes` (default 10), `engineVramReserveMb` (default 1536), `comfyPath`
(the sidecar), `sam3Dir`. The Preferences page's Engine section shows all of them with the
status dot.

### 12.4 Security of BYOM files

Model files are untrusted input: the safetensors parser is bounds-checked and fuzzed; GGUF
strings and counts are capped; no pickle, ever (`.pt`/`.ckpt` are refused with a sentence);
LoRA keys are matched, never evaluated; the ONNX files are the user's and run in the worker
process; the worker runs with no network access needed and asks for none.

### 12.5 Performance budgets

Engine spawn to ready (no models) ≤ 1.5 s; model load from a warm page cache: 2.1 INT8 ≤ 6 s,
Krea 2 ≤ 5 s, SAM ≤ 2 s; per-request overhead outside the model (channel, pixel copy, encode)
≤ 50 ms at 1 MP; UI thread never blocks on the engine (all calls through jobs); idle worker
RSS ≤ 300 MB; VRAM released within 2 s of "Purge".

### 12.6 Tests to add, by crate

- `gn-kernels`: per-kernel numeric tests against a CPU reference at several shapes and dtypes;
  a GPU-marked test group skipped without a device.
- `gn-blocks`: parity per block against committed reference tensors; schedulers against known
  sigma tables.
- `gn-models`: definition smoke tests on random weights (shapes flow, no NaN); the harness for
  real weights (`#[ignore]`).
- `gn-weights`: hand-built safetensors/GGUF fixtures, key maps for the three layouts, the
  fuzz target, `ModelIndex` on a synthetic folder.
- `gn-perception`: the 2-op ONNX fixture; pre/post-processing; prompt encodings.
- `gn-engine`: protocol round trips, cancel mid-run, kill-and-respawn, budget eviction order.
- PhotoshopEX: `routing` unit tests (availability, overrides, fall-through), `FakeEngine`-backed
  command tests for every `generate.*`/`select.*`, UI tests for the badge, the status dot and the
  Engine page, control-channel tests for the new commands, `prefs_usage` for the new
  preferences, translations ×13.

### 12.7 Live scripts (`docs/comfy/bench/scripts/`)

`sidecar-live.ps1`, `sam-native-live.ps1` (object-select-live with ComfyUI stopped),
`engine-live.ps1` (start, index, generate on each lane, purge, crash injection via
`generate.engine.stop` mid-run, respawn), `parity-live.ps1` (drives `gn-bench` and pastes the
table into the dev log entry), the existing `walkthrough-live.ps1` under `engine = native`.

### 12.8 Licence and attribution checklist (before any published build)

`models.md` rows for every model the index recognises (Krea 2 Community: the revenue cap and the
content-filter obligation surfaced in the Get-models page; Qwen Research: non-commercial, never
default; SAM: licence text shipped, gated weights are the user's); `NOTICE` lists ported files
(diffusers, ComfyUI Apache-marked files, qwen-image-rs, SceneWorks if used, sd.cpp headers);
`THIRD_PARTY/` carries the CUDA, ONNX Runtime and (if used) TensorRT-RTX licence texts; the
brand checklist in `upstream.md`.

---

## 13. Acceptance walkthrough (the owner's test)

With ComfyUI **not running** and no console window anywhere:

1. Launch from the taskbar pin. The status bar shows a grey engine dot; nothing starts yet.
2. Open a photo. Press W, Quick Selection, paint across the subject: the dot turns green within
   two seconds (engine + SAM load), the selection appears, and moving the pointer with the
   Object Selection tool tints objects as the pointer moves.
3. Select › Subject: instant. Switch to Refine, Alt-click a limb: instant.
4. Generative Fill with a prompt: the task bar shows "native" in its tooltip, previews tick
   every few steps, the result lands at the same quality as Quality `best` did through ComfyUI
   (compare in the Generations library: an older run with the ☁ badge, this one with ⚡).
5. Generate Image with Krea 2 Turbo at 1024²: the time estimate is at or below what the
   benchmarks say ComfyUI NVFP4 took.
6. Edit › Purge › Generative Models: VRAM returns to idle within two seconds (Task Manager).
7. Kill `photocraft.exe --engine` from Task Manager mid-generation: the app says the engine
   stopped and restarts it; the document is untouched; Retry works.
8. Preferences › AI Integrations › Engine: set Krea 2 to "ComfyUI", start ComfyUI from the same
   page (sidecar), generate: the ☁ badge; set it back.
9. Drop a brand-new community model's GGUF into the models folder with an sd.cpp definition
   upstream: it shows up in the index with the ◇ badge and generates. No update of PhotoshopEX
   was needed.
10. Install the build on a fresh Windows user profile, point it at the models folder, repeat 2–5.

---

## 14. Timeline, staffing, risks

**Timeline at the owner's demonstrated pace** (the Blood Gulch week: a lead model plus a swarm
of bounded workers, 8 days, ~90 k lines), with one correction: this work serializes on the GPU
and on a dependency chain, so the swarm is 3–4 parallel tracks, not 20.

| Week | Track A (engine) | Track B (perception) | Track C (product) | Track D (kernels) |
|---|---|---|---|---|
| 1 | harness; qwen-image-rs on sm120; sd.cpp bench | SAM export + ORT bring-up | sidecar; engine prefs page | — |
| 2–3 | fork, restructure into layers; loader (official + repack) | SAM native behind `select.*`; probe follow | routing + badges; `FakeEngine` | — |
| 4–5 | 2.1 t2i parity; worker process; residency | engine cache; TensorRT provider | Generations `engine`; status dot | SageAttention sm120 |
| 5–7 | 2.1 references/edit/fill/expand/RGBA | — | task-bar previews | FP8 GEMM |
| 6–10 | Krea 2 definition; Wan VAE; Qwen3-VL 4B; LoRA loader | — | Get-models page; model index UI | NVFP4 GEMM |
| 9–12 | GGUF paths; VRAM policies; determinism | — | installer; runtime folder; clean-profile test; licences | CUDA graphs |

**~6–8 weeks to "works end to end"** (SAM, 2.1 generate+edit, Krea 2, Python-free, routed);
**~12 weeks to the acceptance walkthrough** on a clean profile. The kernel track continues past
that and is where speed beyond ComfyUI comes from.

**Risks and what we do about them**

| Risk | Likelihood | Mitigation |
|---|---|---|
| qwen-image-rs's kernels do not come up on sm120 in week 1 | medium | the restructure proceeds on bf16 candle paths; kernels are Track D's first item instead of a prerequisite |
| sd.cpp > 2× slower than ComfyUI | medium | the sidecar stays the fallback lane; sd.cpp is not packaged; nothing else changes |
| SAM 3 point/box ONNX export fights back | medium | ship text-prompt native first (Select Subject, by text); point/box through TensorRT engines built by `trtexec` (the C++ route with published numbers); ComfyUI route as fallback meanwhile |
| Krea 2 has a block the library lacks and the reference is GPL-only | low | sd.cpp's MIT C++ is a second reference; diffusers is the oracle; write the block from the paper |
| VRAM: 2.1 (7B + 8B encoder) and Krea 2 resident together | high | residency policy evicts encoders after prompt encoding; one DiT resident at a time by default; budgets in prefs |
| A new frontier model mid-plan (expected) | high | it runs through the day-0 lanes immediately; port only if it clears the quarter-stable rule |
| Single GPU serializes agents | high | second box (§15); otherwise strict track ownership and nightly harness runs |
| Windows CUDA toolchain days | high | prebuilt fatbins committed; nvcc needed only by whoever edits a kernel; document the exact VS/CUDA versions in `dev-environment-windows.md` |
| Scope creep into 2511/Layered/Z-Image | medium | they are definitions for later; the rented lanes serve them now; the plan's gate is the walkthrough, not model count |

**What cuts scope if it must:** drop Krea 2 native to a later plan (sd.cpp or ComfyUI NVFP4
serve it today at good speed), keeping SAM + 2.1 native; or drop GGUF reading from the loader
(repack + official only) and let sd.cpp own GGUF.

---

## 15. Decisions for the owner

1. **Engine as a worker process** (§3.4) rather than in the editor's address space.
   *Recommendation: yes.* Costs ~50 ms per request and one more process; buys crash isolation,
   VRAM hygiene and one code path for all engines. This is the hardest thing to change later.
2. **Separate repository `genai-native`** (§4) consumed by PhotoshopEX, vs a crate in-tree.
   *Recommendation: separate.*
3. **A second GPU box for agents**, running **Linux** (Ubuntu 24.04, CUDA 13; a used 4090 is
   enough and also gives sm89 coverage). *Recommendation: yes before Phase 2;* it is the Linux
   platform gate (§3.5), the CI/nightly-harness runner, and without it the timeline's parallel
   tracks collapse to about two.
4. **Test-weight downloads on this machine** (never shipped): Qwen-Image-2.1 bf16 + Qwen3-VL 8B
   (~30 GB, research licence, for the parity oracle and qwen-image-rs), Krea 2 Turbo GGUF Q8
   (~12 GB), Qwen3-VL-4B GGUF (~4 GB), the SAM 3.1 ONNX exports (owner's HF token).
   *Recommendation: approve all four for Phase 0.*
5. **Python survives in exactly one place**: the one-time SAM ONNX export upstream of the app
   (§6.1), because Meta ships no export. *Recommendation: accept, and publish our export script
   and file hashes so most users never run it.*
6. **Qwen-Image-Edit-2511 native**: in or out of this plan? *Recommendation: out;* the rented
   lanes serve it, and 2.1 covers editing for the native scope. Revisit after the walkthrough.
   *2026-10-26:* the owner first chose the port (the ComfyUI route's removal floor is 22–25 s a
   dab, all of it the server's re-staging), then, on AusBoss's Qwen-Image-2.1 workflows, set
   **Krea 2 and Qwen-Image-2.1 as the main models of the build**: removals go to the native
   2.1 route (measured on the bench sets, `native21-run.ps1`), LoRA loading came first (above),
   and 2511 stays the permissive fallback on ComfyUI. The port stays out.

---

## 16. File map

**New, in `genai-native`:** `crates/gn-{kernels,blocks,models,weights,perception,harness,engine,sdcpp}`,
`docs/{ARCHITECTURE,PARITY,PORTING,KERNELS}.md`, `NOTICE`, `THIRD_PARTY/`, `benchmarks/`.

**New, in PhotoshopEX:** `crates/genai/src/{native.rs,routing.rs,engine_proc.rs}`,
`crates/engine/src/generate_engine_cmds.rs` (the `generate.engine.*` commands),
`crates/ui-egui/src/engine_ui.rs` (status dot, Engine page section),
`docs/comfy/bench/scripts/{sidecar,sam-native,engine,parity}-live.ps1`, `docs/comfy/benchmarks.md` §8+.

**Edited surgically:** `apps/photocraft/src/main.rs` (`--engine`), `crates/genai/src/lib.rs`
(`Response.engine`), `crates/engine/src/{generate_cmds.rs,select_ml_cmds.rs,prefs.rs,commands.rs}`,
`crates/ui-egui/src/{prefs_ui.rs,model_picker.rs,jobs_ui.rs,object_select.rs}`,
`docs/control-protocol.md`, `docs/comfy/{architecture,roadmap,models,README}.md`, `packaging/*`,
`CLAUDE.md` (the engine repo's location and build steps), `dev-environment-windows.md`
(CUDA/VS versions, nvcc, the GPU box).

---

## 17. Glossary

**Lane** — one of the three engines a model may run on (native, sd.cpp, ComfyUI). **Day-0 lane** —
a rented engine that gets a new model before anyone ports it. **Definition** — a `gn-models`
file: configuration and wiring of a model over shared blocks. **Parity** — matching the reference
implementation within §12.1's thresholds on the fixture set. **Rented engine** — one we run but
do not write (sd.cpp, ComfyUI). **Residency** — which models stay loaded in VRAM. **Worker** —
the `photocraft.exe --engine` child process. **BYOM** — bring your own models: the app ships
none and reads whatever the user has.
