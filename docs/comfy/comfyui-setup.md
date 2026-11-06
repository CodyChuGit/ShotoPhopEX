# ComfyUI: local server setup and API primer

Written 2026-10-08 for the development PC (Windows 10, RTX 5090 32 GB). ComfyUI is the generative
backend Comfy PhotoCraft talks to; PhotoCraft never imports PyTorch or any Python. The server runs
as a separate process on loopback and PhotoCraft is one of its API clients.

## 1. Which ComfyUI to install

| Option | When | Notes |
|---|---|---|
| **Portable build** (`ComfyUI_windows_portable`, from the GitHub releases) | recommended for development | Embedded Python + CUDA PyTorch, updated with the bundled scripts; you control the version. Blackwell (RTX 50-series) needs a CUDA 12.8+ PyTorch, which current portable builds ship. |
| **ComfyUI Desktop** | for end users | Installer, bundled runtime, auto-updates on the *stable* channel, which lags the Git release by days to weeks (Qwen-Image-2.1 and Krea 2 needed a fresh release when they landed). |
| **Git clone + venv** | if you already keep Python | Needs Python 3.12/3.13. This PC has **no Python** (only the Microsoft Store stub), so this is the least convenient option here. |
| **Pinokio** | one-click | Pinokio is installed on this PC (`C:\Users\5090\AppData\Local\Programs\Pinokio`) with no apps yet; its ComfyUI script installs a conda env and the server. Fine for trying things, less controllable for API work. |

Minimum versions: **ComfyUI ≥ 0.37.0** for Qwen-Image-2.1 (released 2026-09-20),
**≥ 0.26.0** for Krea 2 (2026-06-23). Always run a recent release; the official docs say to use
nightly if a workflow's nodes are missing.

## 2. Launch flags

```bash
python main.py --listen 127.0.0.1 --port 8188 --preview-method auto
```

- Keep the server on **loopback** (`127.0.0.1`, the default). PhotoCraft's own control channel is
  loopback-only with a bearer token for the same reason; ComfyUI has no authentication, so a
  `--listen 0.0.0.0` exposes every model on the machine to the LAN.
- `--preview-method auto` makes the server stream low-resolution previews over the WebSocket while
  sampling (Phase 2 shows them on the canvas).
- Base URL assumed throughout: `http://127.0.0.1:8188`. It becomes the preference
  `integrations.comfyServer` (Edit › Preferences › AI Integrations…) in Phase 1.

### Speed flags (researched 2026-10-09, measured in [`benchmarks.md`](benchmarks.md))

ComfyUI's `--fast` is a bundle of experimental optimisations; name the ones you want rather
than passing a bare `--fast`, which also turns on whatever a later version adds.

| Flag | What it does | For us |
|---|---|---|
| `--fast fp8_matrix_mult` | Hardware FP8 matrix multiplication for fp8 (scaled) models on Ada and Blackwell (SM ≥ 8.9); Ampere falls back to the slow emulated path | **Yes on RTX 40/50**, measured ~5 % on the 2511 `fp8mixed` file, nothing lost. lightx2v's `fp8_e4m3fn_scaled` 2511 file was 35 % faster but produced noise with our graph: stay on Comfy-Org's `fp8mixed` |
| `--fast fp16_accumulation` | fp16 accumulation in matmuls; Comfy ships a `run_nvidia_gpu_fast_fp16_accumulation.bat`; the maintainer says it only speeds up **fp16** models, not bf16/fp8; outputs change at the same seed and the docs say quality may drop | No: our models run bf16, fp8 or int8 |
| `--fast autotune` | per-resolution kernel autotuning; the first run at a new size pays seconds (an old issue reported minutes) | Optional; measure |
| `--fast cublas_ops` | needs an extra package (comfy-kitchen's cublas extra) | No |
| `--highvram` | keep models in VRAM instead of unloading them after use | **No, measured twice:** with 2511 fp8 (20 GB) + its 8.7 GB encoder resident, 2 GB stayed free on the 32 GB card, the 40-step run got slower (192 s vs 166 s) and the Lightning LoRA run thrashed on reload; on 0.39's dynamic loader (2026-10-26, 5 GB of the card held by other processes) the first Lightning fill never finished in ten minutes |
| `--disable-fast-disk` | stage weights from unpinned RAM instead of the disk's page cache | **No, measured** (2026-10-26, 61 GB RAM, clean card): the cold 2511 fill 40 s instead of 25, a Krea 2 image 12 s instead of 5, the warm fill unchanged at 16 s |
| `--disable-dynamic-vram` | the old estimate-based loader | **No, measured** (2026-10-26, 5 GB of the card held by other processes): the VAE decode ran "loaded partially, 0 MB usable" and one Lightning fill took 323 s |
| `--reserve-vram N` | VRAM kept free for the OS and other apps | 24 GB card that drives the monitors: `--reserve-vram 2` |
| `--use-sage-attention` | SageAttention kernels (~30 % faster sampling in community reports) | Not yet: needs a wheel built for this exact torch 2.14 / cu130 / Python 3.13, and the global flag is reported to produce black images with Qwen; the per-node patch needs KJNodes |
| `--lowvram` | a no-op while dynamic VRAM is on | No |

The development PC's recommended launch is `C:\Users\5090\ComfyUI\start-comfyui-fast.ps1`
(`--fast fp8_matrix_mult`); the plain `start-comfyui.ps1` stays as the baseline.

If fills become several times slower than usual after switching between model families (the
server then streams weights on every run with ~2 GB of VRAM free), **Edit › Purge › Generative
Models** (`generate.free`, `POST /free`) unloads everything; the next run reloads one family.

### Other cards (RTX 3090 / 24 GB class)

- Ampere has no FP8 tensor cores: fp8 weights still load (they are upcast for compute), so
  `qwen_image_edit_2511_fp8mixed` (20 GB) fits a 24 GB card with the 8.7 GB text encoder
  offloaded to RAM by ComfyUI's dynamic VRAM; expect roughly 2–3× the 5090's step times.
- The step count is the lever that matters there: install the Apache-2.0 Lightning LoRA
  (850 MB) and PhotoCraft's `auto` fill template takes the 8-step tier by itself; the 4-step
  tier is the fallback for the impatient.
- Leave the VRAM mode flags alone; add `--reserve-vram 2` if the card also drives the display.
  GGUF quantisations (`unsloth/Qwen-Image-Edit-2511-GGUF`) need the ComfyUI-GGUF custom node
  and are not covered by the shipped templates.

## 3. Model files

Paths are relative to the ComfyUI install (`ComfyUI/models/`). Every file is a plain
`.safetensors`; download them from the Hugging Face pages in [`models.md`](models.md). Gated repos
(`krea/Krea-2-*`) need a Hugging Face account that accepted the licence; the `Comfy-Org/Krea-2`
repack is not gated.

| Purpose | Folder | File |
|---|---|---|
| Krea 2 Turbo (FP8, recommended) | `diffusion_models/` | `krea2_turbo_fp8_scaled.safetensors` |
| Krea 2 Turbo for style reference | `diffusion_models/` | `krea2_turbo_int8_convrot.safetensors` |
| Krea 2 style reference LoRA | `loras/` | `krea2_style_reference.safetensors` |
| Krea 2 style LoRAs (optional) | `loras/` | e.g. `krea2_softwatercolor.safetensors` |
| Krea 2 text encoder | `text_encoders/` | `qwen3vl_4b_fp8_scaled.safetensors` |
| VAE (shared by Krea 2 and Qwen-Image) | `vae/` | `qwen_image_vae.safetensors` |
| Qwen-Image-Edit-2511 (Apache-2.0 editor) | `diffusion_models/` | `qwen_image_edit_2511_fp8mixed.safetensors` (or `_bf16`) |
| Qwen-Image text encoder (2511 line) | `text_encoders/` | the Qwen2.5-VL-7B encoder file from the 2511 template |
| Qwen-Image-2.1 (research licence) | `diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` (or `qwen_image_2.1_bf16.safetensors`) from `Comfy-Org/Qwen-Image-2.1` |
| Qwen-Image-2.1 text encoder | `text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors` (or `qwen3vl_8b_bf16.safetensors`); optional prompt enhancers `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` / `…_pe_i2i.int8_convrot.safetensors` |
| Qwen-Image-2.1 VAE (its own, not the Qwen-Image one) | `vae/` | `qwen_image_2.1_vae_bf16.safetensors` |
| Z-Image Turbo (Apache-2.0) | `diffusion_models/` | Comfy-Org BF16 release |
| SAM 3.1 (select by text, mattes; SAM License) | `checkpoints/` | `sam3.1_multiplex_fp16.safetensors` from `Comfy-Org/sam3.1` (≈ 1.75 GB; native nodes, templates "SAM3: Image Segmentation" under Utility) |
| Background removal | custom nodes' own folders | BiRefNet weights (MIT) |

The exact encoder file names for the Qwen templates are in the template's "Download models"
panel inside ComfyUI (Workflow › Browse Templates); copy them from there rather than from this
page, which will go stale. Record what you installed in [`devlog.md`](devlog.md).

## 4. Smoke test

PowerShell, with the server running:

```powershell
Invoke-RestMethod http://127.0.0.1:8188/system_stats | ConvertTo-Json -Depth 5
```

Expect `system.comfyui_version`, `system.python_version` and a `devices[]` entry naming the RTX
5090 with `vram_total` / `vram_free`. Then:

```powershell
(Invoke-RestMethod http://127.0.0.1:8188/object_info).PSObject.Properties.Name | Select-String -Pattern "Krea|QwenImage"
```

lists the Krea 2 and Qwen-Image node classes, which proves the server version is new enough.

### From PhotoCraft

Once the server runs, PhotoCraft's own commands check it (headless, no window):

```powershell
.\target\release\photocraft-cli.exe run --new '{"width":64,"height":64}' --cmd generate.health
.\target\release\photocraft-cli.exe run --new '{"width":64,"height":64}' --cmd generate.models
.\target\release\photocraft-cli.exe commands --filter generate
```

`generate.health` reports `ok`, the server version and free VRAM; `generate.models` lists the
templates and whether each model file is installed. The smallest live test is Krea 2 Turbo
text-to-image (about 17 GB of downloads: `krea2_turbo_fp8_scaled.safetensors` ≈ 12 GB,
`qwen3vl_4b_fp8_scaled.safetensors` ≈ 4.5 GB, `qwen_image_vae.safetensors` < 1 GB, all from the
ungated `Comfy-Org/Krea-2` repack; Krea 2 Community License):

```powershell
.\target\release\photocraft-cli.exe --% run --new "{\"width\":1024,\"height\":1024}" --cmd generate.image --params "{\"prompt\":\"a lighthouse at dusk, film photograph\"}" --out lighthouse.png
```

A Generative Fill from the command line (Qwen-Image-Edit-2511, about 30 GB of downloads):

```powershell
.\target\release\photocraft-cli.exe run photo.png --cmd select.rect --params '{"x":200,"y":150,"width":400,"height":300}' --cmd generate.fill --params '{"prompt":"a red bicycle leaning on the wall"}' --out filled.psd
```

## 5. API primer

Everything PhotoCraft needs is in the self-hosted server's routes. Field names below come from the
official route and message docs where they are documented and from `server.py` conventions (widely
used by every client library) where they are not; both are stable across 2024–2026.

### HTTP

| Route | Use | Request | Response |
|---|---|---|---|
| `GET /system_stats` | health check, version, VRAM | — | `{"system": {"os", "python_version", "comfyui_version", …}, "devices": [{"name", "type", "vram_total", "vram_free", "torch_vram_total", "torch_vram_free"}]}` |
| `GET /object_info` · `GET /object_info/{class}` | which nodes exist, their inputs and value lists (model files appear as combo options of the loader nodes) | — | `{ "<ClassName>": {"input": {"required": {...}, "optional": {...}}, "output": [...], "name", "display_name", "category"} }` |
| `GET /models` · `GET /models/{folder}` | model files per folder | — | arrays of file names |
| `POST /upload/image` | send the source image | multipart form: `image` (file), `subfolder` (optional), `type` (`input`), `overwrite` (`true`/`false`) | `{"name", "subfolder", "type"}` |
| `POST /upload/mask` | send a mask for an uploaded image | multipart form: `image` (file), `original_ref` (JSON string `{"filename","subfolder","type"}`), `type` | as above |
| `POST /prompt` | queue a workflow | JSON `{"prompt": <API-format graph>, "client_id": "<uuid>", "extra_data": {...}}` | `{"prompt_id", "number", "node_errors"}`; on validation failure `{"error": {...}, "node_errors": {...}}` |
| `GET /prompt` · `GET /queue` | queue state | — | `{"queue_running": [...], "queue_pending": [...]}` |
| `POST /queue` | clear or delete queued prompts | `{"clear": true}` or `{"delete": ["<prompt_id>"]}` | — |
| `POST /interrupt` | stop the running prompt (our **cancel**) | — | — |
| `GET /history/{prompt_id}` | outputs of a finished prompt | — | `{"<prompt_id>": {"prompt": [...], "outputs": {"<node_id>": {"images": [{"filename", "subfolder", "type"}]}}, "status": {"status_str", "completed", "messages"}}}` |
| `GET /view?filename=…&subfolder=…&type=output` | fetch an output file (PNG bytes) | query params | image bytes |
| `POST /free` | unload models to free VRAM | `{"unload_models": true, "free_memory": true}` | — |

### WebSocket

Connect to `ws://127.0.0.1:8188/ws?clientId=<same uuid as client_id>`. Text frames are JSON
`{"type": "...", "data": {...}}`:

| `type` | When | `data` fields |
|---|---|---|
| `status` | queue changes | `status.exec_info.queue_remaining` |
| `execution_start` | a prompt starts | `prompt_id` |
| `execution_cached` | nodes skipped because their outputs are cached | `prompt_id`, `nodes` |
| `executing` | a node starts; `node: null` means the prompt finished | `node`, `prompt_id` |
| `progress` | per sampling step | `node`, `prompt_id`, `value`, `max` |
| `executed` | a node produced UI output (e.g. SaveImage) | `node`, `prompt_id`, `output` |
| `execution_error` | failure | `prompt_id` plus error details (`node_id`, `node_type`, `exception_message`, `traceback`, …) |
| `execution_interrupted` | after `/interrupt` | `prompt_id`, `node_id`, `node_type`, `executed` |
| `execution_success` | all nodes done | `prompt_id`, `timestamp` |

Binary frames carry previews: a 4-byte big-endian event type (`1` = preview image), a 4-byte
image type (`1` = JPEG, `2` = PNG), then the image bytes. Previews arrive only when a preview
method is enabled.

### The flow PhotoCraft implements

1. Create a `client_id` (UUID v4), open the WebSocket with it.
2. `POST /upload/image` for the source composite crop and, for inpainting, `POST /upload/mask` for
   the selection mask (or upload the mask as a second image and reference it with `LoadImage`).
3. Fill a workflow template: set the `LoadImage` file names, the prompt text, seed, steps, model
   file names, output prefix. `POST /prompt` with the API-format graph and the `client_id`.
4. Read `progress` and `executing` messages to drive the job's progress bar; `jobs.cancel` calls
   `POST /interrupt`; a dropped socket falls back to polling `/history/{prompt_id}`.
5. On `execution_success` (or `executing` with `node: null`), `GET /history/{prompt_id}`, then
   `GET /view` for each `images[]` entry of the SaveImage node.
6. Decode the PNG with `photocraft-codecs`, build the new layer and mask, apply as one undo step.

### API-format workflows

ComfyUI stores workflows in two shapes. The **UI format** (what Save writes) carries node
positions and widgets; the **API format** is the flat `{"<node_id>": {"class_type", "inputs": {...}}}`
map that `/prompt` accepts. Export it from the app with **Workflow › Export (API)** (older
frontends: enable *Dev mode* in Settings first). PhotoCraft ships API-format templates with
placeholder values and never sends the UI format.

### Client libraries, for reference

Rust: `comfyui-client` (HTTP + `/ws` event stream, MulanPSL-2.0) and `comfyui-rs` (async, typed
REST, WebSocket progress with polling fallback). Neither is wasm-capable or in our dependency
policy; they are useful as reference implementations. The official docs' "API examples" page and
the Krita plugin (`Acly/krita-ai-diffusion`, Python) show the same upload → prompt → ws → history
→ view loop. See [`architecture.md`](architecture.md) for why we write our own thin client.

## Sources

- [ComfyUI server routes (official)](https://docs.comfy.org/development/comfyui-server/comms_routes) · [WebSocket messages (official)](https://docs.comfy.org/development/comfyui-server/comms_messages) · [API examples (official)](https://docs.comfy.org/development/comfyui-server/api-examples)
- [Krea-2 workflow (official)](https://docs.comfy.org/tutorials/image/krea/krea-2) · [Qwen-Image-Edit-2511 workflow (official)](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511) · [Qwen-Image workflow (official)](https://docs.comfy.org/tutorials/image/qwen/qwen-image)
- [Hosting a ComfyUI workflow via API (9elements)](https://9elements.com/blog/hosting-a-comfyui-workflow-via-api/) · [ComfyUI API endpoints (runflow)](https://www.runflow.io/blog/comfyui-api-endpoints)
- [comfyui-client (docs.rs)](https://docs.rs/comfyui-client) · [comfyui-rs (docs.rs)](https://docs.rs/comfyui-rs)
- [Krita AI Diffusion (Acly)](https://github.com/Acly/krita-ai-diffusion)
