# R0 inventory: what still needs Python, ComfyUI or ONNX Runtime

> Written by Codex (`codex exec`, read-only, 2026-10-27) for the Rust rebuild (`docs/comfy/goals/rust-rebuild.md` R0); checked by Claude against the code. Line numbers are of that evening's tree (before the fill-from-reference change).

**The “one Rust executable, no Python or ONNX Runtime at runtime” goal is not yet met across all features.** Native diffusion runs through candle, but native SAM runs through ONNX Runtime. Reference-fill, reference-image, angles, layered splitting, and Krea style-reference defaults still route to ComfyUI.

This was a static, read-only audit. No files were modified, builds executed, or models run. Actual DLL loads and the contents of existing executables are **unverified**.

Citation abbreviations below identify these exact files:

- **G** = `PhotoshopEX/crates/engine/src/generate_cmds.rs`
- **S** = `PhotoshopEX/crates/engine/src/select_ml_cmds.rs`
- **D** = `genai-native/crates/gn-diffusion/src/lib.rs`

**1. Feature routing and native coverage**

The local worker advertises nine diffusion/text templates: `qwen-2.1/{image,edit,fill,expand,matte}`, `krea2-turbo/{image,edit}`, and `enhance/{edit,image}`. SAM advertises `sam3.1/{segment,segment-point,segment-box}`. These are implementation capabilities; installed-model checks can reduce the advertised set. [D:44–87][d44], [D:303–309][d303], [gn-perception/engine.rs:26][p26]

| User-facing feature / command | Default template selection | Native worker support today | Fallback / alternative |
|---|---|---|---|
| Generative Fill — `generate.fill` | Preference defaults to `auto`. With research models allowed and native routing configured, chooses `qwen-2.1/fill`; with prompt enhancement enabled, `fillAsEdit` defaults true and chooses `qwen-2.1/edit`. Otherwise uses the 2511 fill order. [G:1697–1728][g1697] | **Yes**, both 2.1 templates. **No** 2511 templates. [D:78–87][d78] | 2511 order: quality `best` → `qwen-edit-2511/fill`; `balanced` → `fill-lightning-8`, then `fill`; `fast` → `fill-lightning-4`, then 8, then base. Resolution selects the first installed candidate, otherwise the base template. [G:50–59][g50], [G:565–575][g565], [G:1761–1776][g1761] |
| Generative Edit — `generate.edit` | Under `auto`, `qwen-2.1/edit` with research/native configuration; otherwise 2511 edit order. [G:519–527][g519], [G:3772][g3772] | **Yes** 2.1; **no** 2511. [D:78–87][d78] | `best` → `qwen-edit-2511/edit`; other qualities → `edit-lightning-8`, then base. [G:534–537][g534], [G:565–575][g565] |
| Generative Expand — `generate.expand` | Under `auto`, `qwen-2.1/expand` with research/native configuration; otherwise 2511 expand order. [G:519–527][g519], [G:3515][g3515] | **Yes** 2.1; **no** 2511. [D:78–87][d78] | `best` → `qwen-edit-2511/expand`; other qualities → `expand-lightning-8`, then base. [G:55–59][g55], [G:565–575][g565] |
| Generate Image — `generate.image` | Preference default is `krea2-turbo/image`. [prefs.rs:990–991][prefs990], [G:2142][g2142] | **Yes**; explicit `qwen-2.1/image` is also served. [D:78–87][d78] | Under engine `auto`, ComfyUI serves the **same selected template** if native routing cannot provide it. No automatic switch to another image model is established here. [G:894–906][g894] |
| Multiple Angles — `generate.angles` | `qwen-edit-2511/angles`. [G:484–486][g484] | **No**, absent from native list. [D:78–87][d78] | ComfyUI under `auto`; forced-native mode does not select a ComfyUI fallback. [G:894–906][g894] |
| Split into layers — **`generate.splitLayers`** | `qwen-layered/split`. The implemented command ID is not `generate.split`. [G:421–422][g421] | **No**, absent from native list. [D:78–87][d78] | ComfyUI under `auto`; forced-native mode stays native. [G:894–906][g894] |
| Generative Remove Background — `generate.removeBackground` | `auto` selects `qwen-2.1/matte` when research models are allowed; otherwise `sam3.1/segment`. [G:3914–3933][g3914] | **Yes**: candle diffusion matte; **yes through ORT**: SAM segmentation. [D:78–87][d78], [gn-perception/engine.rs:26][p26] | If automatic matte selection finds no installed matte template, substitutes `sam3.1/segment` **on the already selected backend**. This is availability selection, not retry after failed inference. [G:3977–3984][g3977], [G:4035–4040][g4035] |
| Text selection — `select.byText` | `sam3.1/segment`. [S:37–39][s37], [S:239–244][s239] | **Yes, ORT**, not candle. [gn-perception/engine.rs:26][p26], [engine.rs:70–75][p70] | ComfyUI under `auto` if native routing is unavailable. [G:894–906][g894] |
| ML subject selection — `select.subjectML` | Same text template, with subject-specific text. Default subject search tries “the main subject”, “the foreground object”, “person”, “animal”, “object”. [S:570–571][s570], [S:613][s613] | **Yes, ORT**. [gn-perception/engine.rs:26][p26] | Backend fallback follows `backend_for`; subject fallback is repeated **text prompts**, not another model or a classical selector. [S:414–433][s414], [G:894–906][g894] |
| Point/box selection — `select.byPoint`; hover probe — `select.probe` | `sam3.1/segment-point`, or `sam3.1/segment-box` when a box is supplied. [S:239–244][s239] | **Yes, ORT**, both templates. [gn-perception/engine.rs:26][p26] | ComfyUI under `auto`. `select.byPoint` can also apply a retained probe instead of requesting inference. [G:894–906][g894], [S:870][s870] |
| Remove tool — `paint.remove`, generative path | Calls shared `generative_remove`: automatic native/research choice is **`qwen-2.1/edit`**; otherwise 2511 fill. [remove.rs:214–221][remove214], [G:1491–1514][g1491] | **Yes** 2.1 edit; **no** 2511 variants. [D:78–87][d78] | Without the generative/configuration/pixel-target conditions, continues through the classical Remove implementation. Once a generative job is selected, an inference failure does not establish a classical retry. [remove.rs:214–233][remove214] |
| Content-Aware Fill — `edit.contentAwareFill`, generative path | Calls the same `generative_remove` selection. [edit_menu_cmds.rs:359][caf359], [edit_menu_cmds.rs:710–726][caf710], [G:1491–1514][g1491] | Same as Remove. [D:78–87][d78] | When generative mode is not selected/configured, continues through the classical PatchMatch job. [edit_menu_cmds.rs:728–735][caf728] |
| Spot Healing — `paint.spotHealing`, type `generative` | Calls the same `generative_remove` selection. [retouch_cmds.rs:845–860][spot845], [G:1491–1514][g1491] | Same as Remove. [D:78–87][d78] | If generative configuration/pixel-target conditions fail, explicitly substitutes `SpotType::ContentAware`. [retouch_cmds.rs:854–866][spot854] |
| Removal adapter candidate | On the 2511 automatic path with multiple variations, may use installed `qwen-edit-2511/remove-lightning-8` for the first candidate; remaining candidates use ordinary fill. [G:2593–2619][g2593], [G:3136][g3136] | **No**. [D:78–87][d78] | Ordinary fill remains the alternative when the adapter is absent. Removal `best` quality is reduced to `balanced`; `fast` stays fast. [G:555–558][g555] |
| Whole-object detection for Remove / Spot Healing | `sam3.1/segment-point`, with optional subsequent `sam3.1/segment` text detection. [S:778–786][s778] | **Yes, ORT**. [gn-perception/engine.rs:26][p26] | If detection is unavailable, fails, or finds nothing, removal proceeds from the stroke alone. [S:778–791][s778] |
| Generative Upscale — `generate.upscale` | `qwen-2.1/edit` when research/native routing is configured; otherwise **base** `qwen-edit-2511/edit`. Adds installed Restore LoRA automatically when no LoRAs were explicitly supplied. [generate_upscale.rs:112–133][up112] | **Yes** 2.1; **no** 2511. [D:78–87][d78] | 2511 base edit is the planning alternative; this code does not run the Lightning auto-order. [generate_upscale.rs:112–119][up112] |
| Standalone prompt rewrite — `generate.enhancePrompt` | `enhance/edit` when an image is supplied; otherwise `enhance/image`. [G:380–386][g380] | **Yes in local source**, using the 2.1 family. [D:78–87][d78] | **Routing exception:** standalone command calls `backend(s)`, so engine `auto` uses ComfyUI; forced-native mode uses native. Empty usable output returns the typed prompt; backend errors propagate. [G:4581][g4581], [G:4608–4615][g4608], [G:852–874][g852] |
| Embedded prompt rewrite / removal judge | Same `enhance/*`; the judge supplies an image and therefore uses `enhance/edit`. [G:386][g386], [G:3205–3209][g3205] | **Yes in local source**. Native jobs use their own worker when its index lists `enhance/edit`. [G:915–922][g915] | Otherwise enhancer routing is tried, then the job backend. `qwen3-vl` is outside `NATIVE_AUTO_FAMILIES`, so ordinary auto enhancer routing is ComfyUI. Image rewrite failure retains the planned prompt; judge failure uses edge mismatch. [enhance-edit.json:5][enhance5], [G:954–967][g954], [G:2224][g2224], [G:2843][g2843] |
| Fill from reference — `generate.fill` with references | Without an explicit compatible template: `qwen-edit-2511/fill-ref` for `best`, otherwise `fill-ref-lightning-8`. Native 2.1 automatic selection is skipped when references exist. [G:1715–1722][g1715], [G:1136][g1136] | **No** for these defaults. [D:78–87][d78] | ComfyUI under `auto`; no automatic native replacement template. [G:894–906][g894] |
| Image from references — `generate.image` with references | If the default template accepts none, switches to `qwen-edit-2511/image-ref`. Thus default Krea text-to-image + references takes this route. [G:2064][g2064], [G:2144–2153][g2144] | **No** for the default reference template. [D:78–87][d78] | ComfyUI under `auto`. A configured default that already supports references is retained. [G:2148–2150][g2148] |
| Krea 2 style reference — `generate.image` with `krea2-turbo/image-style` | Explicit/reference-capable selection; metadata declares task `image`, family `krea2`, one or two required references. [workflow:3–6][style3], [workflow:29][style29] | **No**; not listed natively despite its allowed family. [D:78–87][d78], [G:963–967][g963] | ComfyUI under `auto`. Workflow notes say live execution was not performed because the LoRA was absent; current successful operation is **unverified**. [workflow:25][style25] |
| Krea 2 instruction edit — `generate.edit` with `krea2-turbo/edit` | Explicit template alternative, not the automatic edit default. [workflow:3–6][kedit3], [G:519–527][g519] | **Yes in local source**, conditional on `krea2_general_edit_v2.safetensors` being installed. [D:59–61][d59], [D:303–309][d303] | ComfyUI under `auto` if not advertised. Its graph requires the custom `comfyui-krea2edit` nodes. [G:963–967][g963], [workflow:51][kedit51] |
| Generate Similar — `generate.similar` | Reuses stored image/edit/fill template; an expanded area is regenerated as a fill. Split layers explicitly reject Similar. [G:4467–4507][g4467] | Depends on selected/reused template. | Uses the same template routing rules; expand-to-fill uses native 2.1 when eligible, otherwise 2511 auto-fill. [G:4496–4507][g4496] |

The remaining generation controls—variation selection/deletion/keeping, information, models, health, LoRA listing, and freeing models—are registered alongside these features but do not introduce another generation template. [G:4709–4800][g4709]

**Routing qualifications that affect the table:**

- Engine `native` unconditionally chooses native; engine `auto` requires configured folders, an allowlisted family, and the template in the worker index. Otherwise it chooses ComfyUI. This is **backend selection**, not a general failover after inference errors. [G:894–906][g894], [G:960–967][g960]
- `native21_auto` uses `loras::runs_native`, which checks configuration/family eligibility, **not actual worker template availability**. Therefore initial 2.1 selection can occur before availability is confirmed; a later ComfyUI route can receive that same 2.1 template. [G:519–527][g519], [generate_loras.rs:59–65][loras59]
- The checked-in manifest pins `f2bf496…`, but this machine’s user Cargo configuration patches all three worker crates to the local checkout. The pinned diffusion source advertises six templates; the local source adds `enhance/edit`, `enhance/image`, and `krea2-turbo/edit`. Existing executable provenance is **unverified**. [genai/Cargo.toml:38–40][manifest38], [user config.toml:5–8][userpatch], [pinned lib.rs:57][pinned57], [D:78–87][d78]

**2. Runtime ONNX Runtime and cuDNN use**

| Location | Verified runtime use |
|---|---|
| `gn-perception/Cargo.toml` | Sole direct `ort` dependency found in the audited Rust source: version `=2.0.0-rc.15`, features `load-dynamic`, `std`, `cuda`. [Cargo.toml:24][ort24] |
| `gn-perception/src/runtime.rs` | Loads `onnxruntime.dll`/platform equivalent through `ort::init_from`; creates ORT sessions; selects CUDA plus CPU execution providers, or CPU alone. CUDA provider initialization errors are surfaced. [runtime.rs:75–79][rt75], [runtime.rs:131–178][rt131] |
| `gn-perception/src/sam31.rs` | SAM 3.1 vision, optional tracker, text, geometry, decoder, and heads are ORT sessions. CUDA capture uses ORT device allocation and I/O binding. [sam31.rs:31–33][sam31imports], [sam31.rs:214–238][sam31sessions] |
| `gn-perception/src/sam3.rs` | Alternate Hugging Face SAM 3 implementation opens four ORT graphs. It is retained in the library/tool, although the editor worker loads `Sam31`. [sam3.rs:103–117][sam3open], [engine.rs:75][p75] |
| `gn-perception/src/graph.rs` | Shared extraction/error helpers operate on ORT session outputs and errors. [graph.rs:40–59][graph40] |
| Editor native worker | Installs the perception engine, which lazily loads ORT and SAM on demand. [native.rs:465–466][native465], [engine.rs:58–75][p58] |
| `gn-sam` diagnostic executable | Opens ORT directly and can run either SAM export layout. [gn-sam.rs:108–109][gnsam108], [gn-sam.rs:153][gnsam153], [gn-sam.rs:202][gnsam202] |
| `gn-harness` | Its SAM/ORT adapter launches `gn-sam` with runtime/model arguments; this is indirect ORT use for benchmarking. [engines.rs:206][harness206], [engines.rs:268][harness268] |

There are **two distinct cuDNN paths**:

1. **SAM through ORT’s CUDA execution provider.** The loader preloads ORT’s CUDA/cuDNN lists and every additional matching cuDNN 9 library in configured directories. CPU SAM avoids this CUDA preload path. [runtime.rs:83–122][rt83], [runtime.rs:136–139][rt136], [runtime.rs:168–178][rt168]

2. **Optional qir VAE acceleration.** qir’s `cudnn` feature enables a custom cuDNN convolution bridge, with library-search setup in `build.rs`. VAE code calls that bridge conditionally and has fallback handling. **The editor’s `gn-diffusion/cuda` feature does not enable `qir/cudnn`.** [qir/Cargo.toml:36–50][qircudnn], [qir/build.rs:122–139][qirbuild122], [cudnn_conv.rs:21–22][conv21], [vae.rs:195–212][vae195], [gn-diffusion/Cargo.toml:17][diffcuda]

ComfyUI itself conditionally configures PyTorch cuDNN benchmarking when CUDA/cuDNN are available. This does **not** establish that every ComfyUI feature strictly requires cuDNN; per-operation requirements are **unverified**. [ComfyUI/model_management.py:563–565][comfycudnn]

**3. Python at build/setup time**

| Purpose | Requirement and evidence |
|---|---|
| **Current SAM graph preparation** | `tools/export_sam31.py` needs Python, PyTorch, and ComfyUI’s modules/checkpoint. Export uses `torch.onnx.export`; documented exporter packages are `onnx` and `onnxscript`. This is the setup dependency for the current ORT SAM route. [export_sam31.py:53–75][export53], [export_sam31.py:264][export264], [EXPORT.md:27–38][exportdoc27] |
| Optional export verification | `--verify` imports Python `onnxruntime` and compares against PyTorch; verification is skipped if that package is absent. This is separate from the Rust worker’s ORT runtime dependency. [export_sam31.py:304–306][export304] |
| SAM reference/smoke generation | `tools/sam31_smoke.py` runs ComfyUI’s own SAM node as ground truth. Development verification, not native application startup. [dev-environment-windows.md:129][smokedoc] |
| qir host setup recipe | `scripts/setup-host.sh` creates a Python 3.12 oracle environment and installs torch/diffusers/transformers; also obtains cuDNN through the `nvidia-cudnn-cu13` wheel. These are that setup recipe’s tooling choices. [setup-host.sh:26–41][setup26] |
| Oracle comparisons / remote helper | Python oracle and comparison scripts are development tools; `host.sh` also uses `python3` for JSON transport. [qir/README.md:328–329][qir328], [host.sh:33–34][host33] |
| TensorRT experiments | Separate benchmark setup creates a Python environment; `run_variants.sh` runs Python ONNX export, TensorRT building, and benchmarking. Not part of the native candle worker. [tensorrt/setup.sh:12–20][trt12], [run_variants.sh:10–13][trtvariants] |
| ComfyUI benchmark setup | Installs Python/PyTorch, ComfyUI requirements, and optional attention/custom-node packages. [bench/comfyui/setup.sh:23–41][comfysetup23] |
| Ancillary packaging | Icon regeneration, macOS DMG asset generation, and Flatpak Cargo-source generation have Python utilities. They are separate from generative model inference. [icon-px.py:4][icon4], [DMG README:31–35][dmg31], [Flatpak manifest:5–6][flatpak5] |

**The qir converters are Rust, not Python:** `convert-fp8` dispatches to Rust safetensors conversion; `scripts/convert.sh` invokes the Rust binary’s `prequantize-text` and `prequantize-convrot` commands. The inspected CUDA build script invokes CUDA compilation and configures linking; it does not invoke Python. [main.rs:615–617][convert615], [convert.rs:1–17][convert1], [convert.sh:26–30][convertsh26], [qir/build.rs:15][qirbuild15]

Python is also a **runtime** dependency whenever ComfyUI is used: the sidecar resolves a Python interpreter and launches `main.py`. Copying cuDNN DLLs out of a torch installation does not itself require Python at application runtime. [sidecar.rs:170–203][sidecar170], [sidecar.rs:300–309][sidecar300]

**4. Windows DLL inventory and effect of candle SAM**

The packaging script explicitly requires this AI runtime payload:

| DLL group | Exact packaged names | After replacing SAM with candle |
|---|---|---|
| CUDA | `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`, `curand64_10.dll` | **Remain** for native diffusion. In particular, qir custom kernels explicitly link `cudart`; it is not solely an ORT dependency. [runtime.ps1:41–42][runtime41], [qir/build.rs:53–70][qirbuild53] |
| ONNX Runtime | `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` | **Can disappear from the editor’s runtime payload** once SAM and its runtime path no longer use ORT. Retained `gn-sam`/SAM3 tools would still require ORT independently. [runtime.ps1:43–44][runtime43], [sam3.rs:103–117][sam3open] |
| cuDNN 9 | `cudnn64_9.dll`, `cudnn_graph64_9.dll`, `cudnn_ops64_9.dll`, `cudnn_cnn64_9.dll`, `cudnn_adv64_9.dll`, `cudnn_heuristic64_9.dll`, `cudnn_engines_precompiled64_9.dll`, `cudnn_engines_runtime_compiled64_9.dll`, `cudnn_engines_tensor_ir64_9.dll`, `cudnn_ext64_9.dll` | **Can disappear if candle SAM also avoids cuDNN**, and optional qir cuDNN remains disabled. “Runs in candle” alone does not guarantee this. [runtime.ps1:45–48][runtime45], [qir/Cargo.toml:36–50][qircudnn] |

Additional loader details:

- Desktop and CLI build scripts delay-load `cublas64_{major}.dll`, `cublasLt64_{major}.dll`, and `curand64_10.dll` when native diffusion is enabled. Both also delay-load `windows.data.pdf.dll` for printing; that Windows component is unrelated to SAM. [desktop build.rs:19–33][desktop19], [CLI build.rs:19–29][cli19]
- Native worker preflight checks only those three cuBLAS/cuRAND names. It is **not a complete transitive DLL inventory**. [native.rs:471–475][native471]
- ORT’s Windows CUDA preload list additionally attempts **`cufft64_12.dll`**, which `runtime.ps1` does not package. Preload failures are deferred to provider initialization. Whether this installed ORT/SAM execution actually requires cuFFT is **unverified**. [ort/cuda.rs:332][ortcuda332], [runtime.rs:100–108][rt100]
- The inspected cudarc dependency links CUDA driver and NVRTC libraries conditionally. Their exact presence in a shipped executable’s import graph, and NVRTC’s resolved Windows DLL filename, are **unverified**; neither is covered by the packaging list above. [cudarc/build.rs:193–208][cudarc193], [candle-core/Cargo.toml:172–176][candle172]

The worker already supports launching the **same executable** with `--engine`; a second Python worker is unnecessary for covered native features. Removing Python from **all** runtime features additionally requires replacing the ComfyUI-only templates and accounting for the standalone enhancer routing. [native.rs:132][native132], [G:4581][g4581]

[g50]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:50
[g55]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:55
[g380]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:380
[g386]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:386
[g421]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:421
[g484]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:484
[g519]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:519
[g534]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:534
[g555]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:555
[g565]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:565
[g852]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:852
[g894]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:894
[g915]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:915
[g954]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:954
[g960]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:960
[g963]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:963
[g1136]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:1136
[g1491]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:1491
[g1697]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:1697
[g1715]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:1715
[g1761]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:1761
[g2064]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2064
[g2142]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2142
[g2144]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2144
[g2148]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2148
[g2224]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2224
[g2593]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2593
[g2843]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:2843
[g3136]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3136
[g3205]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3205
[g3515]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3515
[g3772]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3772
[g3914]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3914
[g3977]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:3977
[g4035]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4035
[g4467]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4467
[g4496]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4496
[g4581]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4581
[g4608]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4608
[g4709]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_cmds.rs:4709
[s37]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:37
[s239]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:239
[s414]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:414
[s570]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:570
[s613]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:613
[s778]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:778
[s870]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/select_ml_cmds.rs:870
[d44]: /C:/Users/5090/Documents/genai-native/crates/gn-diffusion/src/lib.rs:44
[d59]: /C:/Users/5090/Documents/genai-native/crates/gn-diffusion/src/lib.rs:59
[d78]: /C:/Users/5090/Documents/genai-native/crates/gn-diffusion/src/lib.rs:78
[d303]: /C:/Users/5090/Documents/genai-native/crates/gn-diffusion/src/lib.rs:303
[p26]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:26
[p58]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:58
[p70]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:70
[p75]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/engine.rs:75
[prefs990]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/prefs.rs:990
[remove214]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/retouch_cmds/remove.rs:214
[caf359]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/edit_menu_cmds.rs:359
[caf710]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/edit_menu_cmds.rs:710
[caf728]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/edit_menu_cmds.rs:728
[spot845]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/retouch_cmds.rs:845
[spot854]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/retouch_cmds.rs:854
[up112]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_upscale.rs:112
[enhance5]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/enhance-edit.json:5
[style3]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/krea2-turbo-image-style.json:3
[style25]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/krea2-turbo-image-style.json:25
[style29]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/krea2-turbo-image-style.json:29
[kedit3]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/krea2-turbo-edit.json:3
[kedit51]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows/krea2-turbo-edit.json:51
[loras59]: /C:/Users/5090/Documents/PhotoshopEX/crates/engine/src/generate_loras.rs:59
[manifest38]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/Cargo.toml:38
[userpatch]: /C:/Users/5090/scoop/persist/rustup/.cargo/config.toml:5
[pinned57]: /C:/Users/5090/scoop/persist/rustup/.cargo/git/checkouts/genai-native-971b22a75c3cc248/f2bf496/crates/gn-diffusion/src/lib.rs:57
[ort24]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/Cargo.toml:24
[rt75]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:75
[rt83]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:83
[rt100]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:100
[rt131]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:131
[rt136]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:136
[rt168]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/runtime.rs:168
[sam31imports]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:31
[sam31sessions]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam31.rs:214
[sam3open]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/sam3.rs:103
[graph40]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/graph.rs:40
[native465]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/src/native.rs:465
[gnsam108]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/bin/gn-sam.rs:108
[gnsam153]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/bin/gn-sam.rs:153
[gnsam202]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/src/bin/gn-sam.rs:202
[harness206]: /C:/Users/5090/Documents/genai-native/crates/gn-harness/src/engines.rs:206
[harness268]: /C:/Users/5090/Documents/genai-native/crates/gn-harness/src/engines.rs:268
[qircudnn]: /C:/Users/5090/Documents/genai-native/crates/qir/Cargo.toml:36
[qirbuild122]: /C:/Users/5090/Documents/genai-native/crates/qir/build.rs:122
[conv21]: /C:/Users/5090/Documents/genai-native/crates/qir/src/cudnn_conv.rs:21
[vae195]: /C:/Users/5090/Documents/genai-native/crates/qir/src/model/vae.rs:195
[diffcuda]: /C:/Users/5090/Documents/genai-native/crates/gn-diffusion/Cargo.toml:17
[comfycudnn]: /C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/comfy/model_management.py:563
[export53]: /C:/Users/5090/Documents/genai-native/tools/export_sam31.py:53
[export264]: /C:/Users/5090/Documents/genai-native/tools/export_sam31.py:264
[export304]: /C:/Users/5090/Documents/genai-native/tools/export_sam31.py:304
[exportdoc27]: /C:/Users/5090/Documents/genai-native/crates/gn-perception/docs/EXPORT.md:27
[smokedoc]: /C:/Users/5090/Documents/PhotoshopEX/docs/comfy/dev-environment-windows.md:129
[setup26]: /C:/Users/5090/Documents/genai-native/crates/qir/scripts/setup-host.sh:26
[qir328]: /C:/Users/5090/Documents/genai-native/crates/qir/README.md:328
[host33]: /C:/Users/5090/Documents/genai-native/crates/qir/scripts/host.sh:33
[trt12]: /C:/Users/5090/Documents/genai-native/crates/qir/bench/tensorrt/setup.sh:12
[trtvariants]: /C:/Users/5090/Documents/genai-native/crates/qir/bench/tensorrt/run_variants.sh:10
[comfysetup23]: /C:/Users/5090/Documents/genai-native/crates/qir/bench/comfyui/setup.sh:23
[icon4]: /C:/Users/5090/Documents/PhotoshopEX/packaging/icon-px.py:4
[dmg31]: /C:/Users/5090/Documents/PhotoshopEX/packaging/macos/dmg/README.md:31
[flatpak5]: /C:/Users/5090/Documents/PhotoshopEX/packaging/linux/flatpak/ai.storyteller.photocraft.yml:5
[convert615]: /C:/Users/5090/Documents/genai-native/crates/qir/src/main.rs:615
[convert1]: /C:/Users/5090/Documents/genai-native/crates/qir/src/convert.rs:1
[convertsh26]: /C:/Users/5090/Documents/genai-native/crates/qir/scripts/convert.sh:26
[qirbuild15]: /C:/Users/5090/Documents/genai-native/crates/qir/build.rs:15
[sidecar170]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/src/sidecar.rs:170
[sidecar300]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/src/sidecar.rs:300
[runtime41]: /C:/Users/5090/Documents/PhotoshopEX/packaging/windows/runtime.ps1:41
[runtime43]: /C:/Users/5090/Documents/PhotoshopEX/packaging/windows/runtime.ps1:43
[runtime45]: /C:/Users/5090/Documents/PhotoshopEX/packaging/windows/runtime.ps1:45
[qirbuild53]: /C:/Users/5090/Documents/genai-native/crates/qir/build.rs:53
[desktop19]: /C:/Users/5090/Documents/PhotoshopEX/apps/photocraft/build.rs:19
[cli19]: /C:/Users/5090/Documents/PhotoshopEX/apps/photocraft-cli/build.rs:19
[native471]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/src/native.rs:471
[native132]: /C:/Users/5090/Documents/PhotoshopEX/crates/genai/src/native.rs:132
[ortcuda332]: /C:/Users/5090/scoop/persist/rustup/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.15/src/ep/cuda.rs:332
[cudarc193]: /C:/Users/5090/scoop/persist/rustup/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.10/build.rs:193
[candle172]: /C:/Users/5090/scoop/persist/rustup/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-core-0.11.0/Cargo.toml:172
