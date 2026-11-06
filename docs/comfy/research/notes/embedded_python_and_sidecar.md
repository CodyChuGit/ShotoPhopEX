# Shipping the Python stack inside the app vs. a managed sidecar (state as of 2026-10-16)

Scope: how a Rust desktop image editor (MIT OR Apache-2.0) can offer built-in generative AI without
asking the user to install ComfyUI. Options researched: (A) embed CPython + PyTorch in-process via
PyO3 / python-build-standalone; (B) an app-managed sidecar (private ComfyUI or a minimal headless
PyTorch worker) installed on first run and reached over loopback; (C) ComfyUI's own desktop
packaging; (D) precedents: Krita AI Diffusion, Invoke, Amuse (ONNX/DirectML), Draw Things (Swift/Metal).

Research caveat: several primary pages rate-limited or 404'd during this session (gnu.org returned
HTTP 429 three times; raw.githubusercontent.com and jsdelivr returned 404 for the Krita plugin's
`ai_diffusion/server.py` and `resources.py`; web.archive.org is blocked for this tool). Where a
claim rests on a secondary quotation of a primary source, that is said explicitly.

---

## Key question 1: Does embedding PyTorch in-process via PyO3 work on Windows with CUDA, and what breaks?

### Takeaway
It is technically possible (PyO3 supports embedding a dynamically linked CPython and PyTorch is just
an extension module), but nothing in the PyO3 documentation covers numpy/torch embedding, the
documented failure modes (wrong `python3X.dll` on the search path, `OnceLock` deadlocks under the
GIL) are real, and any CUDA/driver access violation inside torch kills the host process. Every
shipping consumer product surveyed here (Krita plugin, Invoke, ComfyUI Desktop) chose a separate
process instead; the only vendor found that discusses in-process LibTorch specifically says GPU
crashes taking down the whole app were the recurring problem that drove it to process isolation.

### Cited Findings
- PyO3 embeds in two linking modes; dynamic linking against `python39.dll`-style shared libraries is "much easier", while static embedding has no "first-class support" and "On Windows static linking is almost never done, so Python distributions don't usually include a static library." For dynamic embedding you "will have to consider including the Python shared library in your distribution" and set `PATH` on Windows; PyO3 points to PyOxidizer for single-file shipping. — [PyO3 guide: Building and distribution](https://pyo3.rs/main/building-and-distribution)
- "The `auto-initialize` feature is deliberately disabled when embedding the interpreter statically." `#[pymodule]` modules are not importable from an embedded interpreter unless registered via `append_to_inittab` before initialization; PyPy cannot be embedded. — [PyO3 guide: Building and distribution](https://pyo3.rs/main/building-and-distribution)
- Windows `STATUS_DLL_NOT_FOUND` / `STATUS_ENTRYPOINT_NOT_FOUND`: "This happens on Windows when linking to the python DLL fails or the wrong one is linked." The DLL (`python3X.dll`, `python3.dll` for abi3, `python3t.dll` for abi3t) must be findable by the Windows DLL search order; PyO3 suggests Dependency Walker for diagnosis. — [PyO3 FAQ](https://pyo3.rs/main/faq)
- GIL deadlock pattern: "`OnceLock`, `LazyLock`, and their thirdparty predecessors use blocking to ensure only one thread ever initializes them", which deadlocks when a thread holding the GIL blocks on an initializer another GIL-waiting thread owns; PyO3's `PyOnceLock` / `OnceExt` are the fix. — [PyO3 FAQ](https://pyo3.rs/main/faq)
- The PyO3 FAQ and distribution guide contain nothing on numpy/torch, CUDA, or host crashes; the embedding story is documented only for the interpreter itself. — [PyO3 FAQ](https://pyo3.rs/main/faq); [PyO3 guide](https://pyo3.rs/main/building-and-distribution)
- Importing torch on Windows has its own DLL-loading history: a PyTorch commit "Fix dll loading process in newer Python on Windows (#17191)" and a later fix for a `__declspec(dllimport)` mismatch that made `import torch` crash on Windows. — [PyTorch commit #17191](https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/428b666814cea4b15860bf3485a613fadd5bfdee); [PyTorch commit #97193](https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/17567e5b29a77dee137dfc365e9ed3f1cf358950)
- A Windows debug-build `import torch` fails with a fatal "GIL is released" error unless run under `python_d.exe`, illustrating how tightly torch's import couples to the exact interpreter build. — [PyTorch forums](https://discuss.pytorch.org/t/windows-debug-build-gil-error-when-importing-torch/216021)
- Real-world evidence that a torch CUDA probe is a *native* crash, not a Python exception: ComfyUI Desktop v0.8.28 on an AMD-only Windows PC died with "Windows fatal exception: access violation" with the stack inside `torch\cuda\__init__.py` `is_available`; a forum reply (2026-04-16, community, not official) attributes it to "the Desktop app bundles a CUDA-only PyTorch" hitting "a native crash instead of a graceful fallback". — [forum.comfy.org, v0.8.28 AMD thread](https://forum.comfy.org/t/windows-desktop-v0-8-28-wont-start-on-amd-gpu/4379)
- A commercial LibTorch C++ SDK (blace.ai) markets process isolation precisely because, in its words, GPU-related crashes "could take down the entire application" with in-process LibTorch; it isolates AI workloads in separate processes so "crashes/leaks don't kill your main app" (vendor claim). — [PyTorch forums, blace.ai announcement](https://discuss.pytorch.org/t/blace-ai-new-c-inference-sdk-model-hub/222361)
- Torch-TensorRT docs note TorchScript's "Python-less interpreter" and that device/context mismatches "could lead the program to crash" without safe mode, i.e. native inference stacks crash the process on mismatch. — [Torch-TensorRT runtime docs](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/runtime.html)
- PyTorch now ships free-threaded `cp313t` wheels for cu128 Windows (`torch-2.11.0+cu128-cp313-cp313t-win_amd64.whl`), so the GIL is no longer strictly mandatory, but these are a separate ABI (`python3t.dll` per PyO3). — [download.pytorch.org cu128 torch index](https://download.pytorch.org/whl/cu128/torch/); [PyO3 FAQ](https://pyo3.rs/main/faq)

### Inferences
- In-process embedding on Windows means: ship `python3XX.dll` + stdlib + a multi-GB site-packages next to `photocraft.exe`, control the DLL search order (torch's own `lib/*.dll`, CUDA runtime DLLs and the Python DLL all must be found before anything else on `PATH`), hold the GIL across every call into torch (or adopt the cp313t free-threaded ABI, which PyO3 supports but which halves the ecosystem of prebuilt wheels), and accept that a CUDA access violation in `torch.cuda.is_available()` or a kernel launch terminates PhotoCraft with the user's unsaved document. That last point directly contradicts upstream's "never crash" rule in AGENTS.md.
- The GIL also means the editor's UI thread cannot share the interpreter with a long sampler loop without releasing the GIL at every step; in practice one dedicated Python thread plus a channel is the same architecture as a sidecar, minus the crash isolation.
- Nobody with a shipping consumer product does in-process torch; the uniform industry choice (below) is a subprocess over loopback. The only in-process precedent is a native (non-Python) runtime (Draw Things, Amuse).

### Gaps
- No GitHub/Reddit thread was found that specifically documents PyO3 + torch + CUDA embedding on Windows end to end (searches surfaced only the PyO3 FAQ and torch's own import bugs). Treat the "works in principle" statement as inferred from PyO3's embedding docs, not from a verified precedent.
- Startup cost of an embedded interpreter importing torch (typically seconds) was not measured or sourced in this session.

---

## Key question 2: What do Krita AI Diffusion and ComfyUI Desktop download, how big is it, and how do they handle updates and GPU detection?

### Takeaway
Both are "download-on-first-run" managers: the Krita plugin installs a private ComfyUI into
`%APPDATA%\krita\ai_diffusion\server` with a backend the user picks (CUDA / ROCm / XPU / Apple / CPU)
and warns that "10GB is a minimum, 50GB+ not unusual"; ComfyUI Desktop (now "Comfy-Desktop",
AGPL-3.0-or-later) ships a ~15–20 MB bootstrap Python and then a relocatable Python with prebuilt
PyTorch GPU wheels, budgets "at least 4.85 GB" per install before models, and updates in place with
snapshots. Neither does driver-level GPU detection beyond torch's own probe, and Desktop's
CUDA-only torch has crashed on AMD-only machines.

### Cited Findings
**Krita AI Diffusion (Acly/krita-ai-diffusion, GPL-3.0 plugin)**
- Install flow: "Open the plugin docker, click Configure to start the local server installation or connect to one"; "The plugin uses ComfyUI as backend"; alternatively "install it manually or use an existing installation." — [README](https://github.com/Acly/krita-ai-diffusion/blob/main/README.md) (as quoted in search results)
- Disk: "AI models can be very large, 10GB is a minimum, 50GB+ not unusual depending on optional packages." Backends offered in the installer: NVIDIA "RTX 2060 or better" via CUDA; AMD "ROCm-enabled graphics cards" via ROCm; "Intel Arc Graphics cards, 8GB or better" via XPU; "Apple Silicon only (M1-M5)"; CPU "very slow". The user chooses the install path; "choose your backend before you click install". Windows Store/Steam Krita and Linux Flatpak may block package installs. — [docs.interstice.cloud/installation](https://docs.interstice.cloud/installation)
- The README (Oct 2026) lists NVIDIA CUDA (Windows/Linux), AMD via DirectML on Windows "(limited, 12+ GB VRAM recommended)" or ROCm on Linux via custom ComfyUI, Apple Silicon community-supported; recommends ≥6 GB VRAM. — [README via search](https://github.com/Acly/krita-ai-diffusion/blob/main/README.md)
- Release v1.53.0 (2026): "managed server was upgraded to CUDA 13, PyTorch 2.13 and latest ComfyUI"; Nunchaku support removed, requiring model re-downloads for Flux 1 / Flux 1 Kontext / Qwen / Z-Image Nunchaku variants. This is how PyTorch/CUDA churn reaches users: a plugin release re-pins the whole managed stack. — [newreleases.io, v1.53.0](https://newreleases.io/project/github/Acly/krita-ai-diffusion/release/v1.53.0)
- Default install location on Windows is `C:\Users\<user>\AppData\Roaming\krita\ai_diffusion\server` (Czech review). — [hwcooling.net](https://www.hwcooling.net/krita-s-ai-diffusion-zdarma-kdyz-se-vam-photoshop-nevyplati/3/)
- Users on ~1 MB/s links report the managed install failing and finishing manually over two days. — [GitHub discussion #704](https://github.com/Acly/krita-ai-diffusion/discussions/704)
- Secondary estimates of the managed footprint range from "~10GB" (user comment) to "easily past 30GB" once ControlNets, upscalers and a second base model are added. — [localaimaster.com guide](https://localaimaster.com/blog/krita-ai-diffusion-guide); [HKI Jena wiki](https://asb.hki-jena.de/wiki/books/ai/page/local-ai-image-editinggeneration-using-krita-ai)

**ComfyUI Desktop / Comfy-Desktop (Comfy-Org)**
- The original `Comfy-Org/desktop` repo is "[ARCHIVED] Superseded by" Comfy-Desktop (read-only since 2026-06-26). It bundled uv: on startup it "will install all the necessary python dependencies with uv and start the ComfyUI server"; "The app will automatically update with stable releases of ComfyUI, ComfyUI-Manager (pip), and the uv executable"; Windows was an "NVIDIA NSIS x64 installer", macOS ARM DMG; licence GPL-3.0; bundled resources at `%LOCALAPPDATA%\Programs\ComfyUI`. — [Comfy-Org/desktop](https://github.com/Comfy-Org/desktop)
- The 2024 open-sourcing post explains the choice: "The Desktop application ships with uv, which creates a Python virtual environment and installs the necessary dependencies"; "we also alias pip to uv pip because uv is fast and you can benefit from the local cache it uses." — [blog.comfy.org, Open sourcing v1 Desktop](https://blog.comfy.org/open-sourcing-v1-desktop/)
- Comfy-Desktop (successor): "each install ships a relocatable Python with PyTorch and GPU wheels prebuilt"; "The app ships a minimal (~15–20 MB) standalone Python with `pygit2` baked in" for git operations; in-place updates of "ComfyUI (and custom nodes)", built-in auto-update, snapshots for rollback; platforms Windows, macOS (Apple Silicon), Linux (.deb/AppImage); licence "AGPL v3 or later" with a commercial licence from licensing@comfy.org, bridge-types package MIT. — [Comfy-Org/Comfy-Desktop](https://github.com/Comfy-Org/Comfy-Desktop)
- Windows docs: "Windows 10 or later", x64 or ARM64, "Dedicated GPU (NVIDIA / AMD) recommended for good performance, but not required", "At least 4.85 GB recommended per installation"; NSIS `.exe`; first launch offers Comfy Cloud vs local install; installs under `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Installs`, shared models under `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Shared`, settings in `%APPDATA%\Comfy Desktop`; auto-checks for updates with "Restart & Update" and an "Automatically install Desktop updates" toggle; uninstall removes only the launcher. — [docs.comfy.org Desktop Windows](https://docs.comfy.org/installation/desktop/windows)
- Earlier wiki guidance: Windows Desktop "Requires NVIDIA graphics card (does not support AMD GPU)", macOS requires Apple Silicon. — [comfyui-wiki.com desktop guide](https://comfyui-wiki.com/fr/install/install-comfyui/comfyui-desktop-installation-guide)
- GPU detection failure modes (user reports, 2026-03 to 2026-04): v0.8.28 crashed in `torch.cuda.is_available()` on an RX 6900 XT; v0.8.27 on an AMD+NVIDIA PC installed a CPU-only torch that the user replaced with cu128 wheels by hand. — [forum.comfy.org v0.8.28](https://forum.comfy.org/t/windows-desktop-v0-8-28-wont-start-on-amd-gpu/4379); [forum.comfy.org v0.8.27](https://forum.comfy.org/t/windows-desktop-v0-8-27-would-not-start-on-my-custom-amd-nvidia-pc/4360)
- ComfyUI's official Windows *portable* builds ship separate NVIDIA / AMD / Intel variants with bundled Python; the NVIDIA variant bundles PyTorch with CUDA 13.0, the AMD build is "experimental" for RDNA 3/3.5/4. — [comfyui-wiki.com Windows install](https://comfyui-wiki.com/install/install-comfyui/install-comfyui-on-windows)

**Common building blocks: uv and python-build-standalone**
- uv's managed Pythons come from Astral's `python-build-standalone`; distributions are "self-contained, highly-portable, and performant", stored under `UV_PYTHON_INSTALL_DIR` (e.g. `.../uv/python/cpython-3.12-...`, a junction on Windows); available versions are "frozen for each uv release"; `python-downloads = manual` / `--no-python-downloads` control fetching. — [uv docs: Python versions](https://docs.astral.sh/uv/concepts/python-versions/)
- python-build-standalone release `20261013` (published 2026-10-14): `cpython-3.11.17+20261013-x86_64-pc-windows-msvc-install_only.tar.gz` is 48,191,837 bytes (≈48 MB); 3.12/3.13 Windows assets exist but their sizes were beyond the readable part of the API response. — [GitHub API, python-build-standalone latest release](https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest)

### Inferences
- The standard recipe in 2026 is: tiny native launcher (Electron/Tauri/Rust) + bundled `uv.exe` + `uv python install` (≈50 MB tarball) + `uv pip install` of a pinned requirements set from a GPU-specific torch index + ComfyUI git checkout + model downloads. Both Krita and Comfy re-pin the whole stack per release rather than letting users float on torch/CUDA versions.
- "GPU detection" in these products is really "the user picks a backend" (Krita) or "the installer picks CUDA by default" (Desktop, NVIDIA-only on Windows historically); real detection would need DXGI/NVML/`nvidia-smi` before Python ever starts, which is a gap PhotoCraft could fill cheaply from Rust.
- The user-visible cost of the sidecar approach is dominated by models, not by the runtime: ≈5 GB runtime (Desktop's 4.85 GB figure) vs 10–50 GB models (Krita's figure).

### Gaps
- The Krita plugin's installer source (`ai_diffusion/server.py`, `resources.py`) could not be fetched (404 from raw.githubusercontent.com and jsdelivr); whether it uses uv, which python-build-standalone build, and the exact torch index per backend are unverified here. Its v1.53.0 release note (CUDA 13 / PyTorch 2.13) is the best dated evidence of the pins.
- Comfy-Desktop's wheel-selection logic for the "relocatable Python with PyTorch and GPU wheels prebuilt" was not found in docs; only the README sentence and user forum reports exist.
- No official download size for either installer binary (Krita plugin zip, Comfy-Desktop NSIS) was found.

---

## Key question 3: Is a GPL-3.0 ComfyUI launched as a separate process by an MIT app a licensing problem?

### Takeaway
ComfyUI is GPL-3.0 (its LICENSE file is the GNU GPL v3, 29 June 2007). The FSF's GPL FAQ treats
programs that communicate by fork/exec, pipes, sockets or command-line arguments as separate
programs and side-by-side distribution as "mere aggregation", so an MIT/Apache app that *spawns*
ComfyUI and talks HTTP/WebSocket to it is, on the FAQ's own test, not a derivative work — with the
FAQ's caveat that "intimate" exchange of complex internal data structures could change that.
Embedding ComfyUI's Python modules in-process (importing `comfy.*` into an interpreter PhotoCraft
hosts) is the pattern the FAQ says *does* make one program.

### Cited Findings
- ComfyUI's licence: "GNU General Public License, version 3 (GPLv3)", dated 29 June 2007. — [comfyanonymous/ComfyUI LICENSE](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE)
- ComfyUI Desktop (archived) was GPL-3.0; Comfy-Desktop is "AGPL v3 or later" dual-licensed with a commercial option. — [Comfy-Org/desktop](https://github.com/Comfy-Org/desktop); [Comfy-Org/Comfy-Desktop](https://github.com/Comfy-Org/Comfy-Desktop)
- Krita AI Diffusion's own plugin is GPL-3.0 and talks to ComfyUI over its HTTP/WebSocket API, so the most successful precedent is itself copyleft and cannot be used to argue the MIT case either way. — [krita-ai-diffusion README](https://github.com/Acly/krita-ai-diffusion/blob/main/README.md)
- GPL FAQ, "MereAggregation" (quoted in an EPICS mailing-list thread because gnu.org rate-limited this session): the test for whether two parts are one program "depends both on the mechanism of communication (exec, pipes, rpc, function calls within a shared address space, etc.) and the semantics of the communication (what kinds of information are interchanged)"; "mere aggregation" is "putting them side by side on the same CD-ROM or hard disk", in which case a GPL on one program "has no effect on the other"; but "if the semantics of the communication are intimate enough, exchanging complex internal data structures, that too could be a basis to consider the two parts as combined into a larger program." — [GNU GPL FAQ #MereAggregation](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation), as quoted by [EPICS tech-talk 2014](https://epics.anl.gov/tech-talk/2014/msg01688.php)
- GPL FAQ on plug-ins: "if the program uses fork and exec to invoke plug-ins, then the plug-ins are separate programs, so the license of the plug-in makes no requirements about the main program"; by contrast dynamically linked plug-ins that "make function calls to each other and share data structures" are a single program. — [GNU GPL FAQ #GPLPlugins](https://www.gnu.org/licenses/gpl-faq.html#GPLPlugins), as quoted by [EPICS tech-talk 2014](https://epics.anl.gov/tech-talk/2014/msg01688.php)
- Comfy-Org itself chose to relicense its desktop shell as AGPL + commercial, which signals that the ecosystem's reference packaging is deliberately copyleft; a permissive app must therefore not vendor Desktop's code. — [Comfy-Org/Comfy-Desktop](https://github.com/Comfy-Org/Comfy-Desktop)

### Inferences
- PhotoCraft's existing design (fork's CLAUDE.md: `crates/genai` talks to a ComfyUI server over HTTP, workflow JSON templates) sits squarely on the "separate programs" side: exec + sockets, with JSON workflow graphs and PNG bytes as the exchanged data. Spawning that same server from a private, app-managed directory does not change the communication mechanism or semantics, so the FAQ analysis is the same as for the user-installed server. Shipping ComfyUI's source alongside the installer (or downloading it on first run) is "mere aggregation" provided the GPL text and source offer accompany the ComfyUI copy and the app's own licence stays separate.
- Risk surfaces: (1) custom nodes written by the fork and installed into ComfyUI's `custom_nodes/` import `comfy.*` and *are* derivative of ComfyUI — they must be GPL-3.0 (ship them as a separate GPL repo, as Krita does with its own `comfyui-tooling-nodes`); (2) a "minimal headless worker" that imports ComfyUI's `comfy/` package for samplers/model loading would also be GPL; (3) a download-on-first-run installer avoids distributing ComfyUI in the PhotoCraft installer at all, which removes the aggregation question but not the custom-node question.
- The FAQ is the FSF's reading, not case law; a lawyer's review before any published build is prudent (the fork's upstream.md brand checklist should add a licence checklist).

### Gaps
- gnu.org returned HTTP 429 for every fetch; the FAQ wording above is from a 2014 mailing-list quotation and should be re-verified against the live page (anchors `#MereAggregation`, `#GPLPlugins`, `#NFUseGPLPlugins`, `#GPLInProprietarySystem`) before being cited in a published document.
- No legal opinion or court case specifically about GPL ComfyUI + permissive frontends was found.

---

## Key question 4: What is the minimum headless PyTorch worker (diffusers + tiny HTTP server) compared to full ComfyUI, in size?

### Takeaway
The runtime floor is set by PyTorch's CUDA wheel, not by ComfyUI: a cu128 Windows torch wheel is
≈3.3 GB (torch 2.7.1 measurement) versus 109 MB for the CPU-only PyPI Windows wheel, and ComfyUI
Desktop budgets 4.85 GB for the whole environment. A diffusers worker saves the ComfyUI code,
frontend and custom-node ecosystem (tens of MB) but keeps the ≈3–4 GB torch+CUDA footprint, so the
size win is small; the real differences are attack surface, update cadence and model/node
compatibility.

### Cited Findings
- `torch-2.7.1+cu128-cp310-cp310-win_amd64.whl` = 3,273,064,440 bytes (≈3.27 GB); `torchvision-0.22.1+cu128` = 7.6 MB; `torchaudio-2.7.1+cu128` = 4.65 MB (Git LFS sizes on a Hugging Face mirror, not download.pytorch.org). — [huggingface.co/socstev/data commit](https://huggingface.co/socstev/data/commit/730ccbd7e8963d0efd04bdb6d55eff994756bd68)
- The official cu128 Windows index currently tops out at torch 2.11.0 for cp312, cp313 and cp313t; the index lists filenames and sha256 but no sizes. — [download.pytorch.org/whl/cu128/torch](https://download.pytorch.org/whl/cu128/torch/)
- PyPI's default torch 2.9.0 wheels: `win_amd64` cp312/cp313 = 109.3 MB each (CPU-only on Windows), `manylinux_2_28_x86_64` cp312 = 899.7 MB (Linux pulls CUDA via separate nvidia-* wheels). — [pypi.org torch 2.9.0 files](https://pypi.org/project/torch/2.9.0/#files)
- CUDA 12.9 Windows wheels were skipped for torch 2.9.1 because "CUDA 12.9 was broken on Windows ... so needed to fall back to 12.8". — [PyTorch forums](https://discuss.pytorch.org/t/torch-2-9-1-cu129-cp312/224340)
- Older data point on how much of a CUDA wheel set is NVIDIA runtime: in 2022 the CU11 nvidia-cublas wheel was 317.1 MB, nvidia-cuda-runtime 557.1 MB, torch itself 776–890 MB. — [PyTorch forums, CUDA dependency bloat](https://discuss.pytorch.org/t/cuda-dependency-make-docker-image-bloated/164784)
- InvokeAI's own Python package is 9.0 MB (sdist, 6.8.0) / 8.1 MB wheel (5.4.2); the launcher is "~150MB max" Electron. — [pypi.org InvokeAI](https://pypi.org/project/invokeai); [invoke-ai/launcher](https://github.com/invoke-ai/launcher)
- ComfyUI Desktop: "At least 4.85 GB recommended per installation" (runtime, excluding models). — [docs.comfy.org](https://docs.comfy.org/installation/desktop/windows)
- python-build-standalone Windows `install_only` CPython ≈48 MB compressed. — [GitHub API latest release](https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest)

### Inferences
- Minimum Windows CUDA worker ≈ 48 MB Python + ≈3.3 GB torch cu128 + ~15 MB torchvision/torchaudio + diffusers/transformers/safetensors/accelerate (hundreds of MB installed) + a few MB server ≈ 3.5–4 GB before models. Full ComfyUI ≈ the same torch plus ComfyUI's repo and requirements ≈ Desktop's 4.85 GB. Difference ≈ 1 GB or less; the Krea 2 / Qwen-Image-Edit / Z-Image checkpoints the fork targets dwarf both.
- A diffusers worker would, however, have to reimplement what ComfyUI already gives the fork for free: model-set memory management, workflow graphs the user can inspect, and the Qwen-Image-Layered / SAM 3.1 / enhancer nodes already wired in `crates/genai/workflows`. Re-doing these in diffusers trades a GPL dependency for a large maintenance surface.
- Because CUDA-major changes (12.8 → 13) ship as whole new 3 GB wheels, any sidecar design should keep torch in a content-addressed cache (uv does this) so an update re-downloads only changed wheels, and should pin a specific `(torch, cuda)` pair per app release exactly as Krita v1.53.0 does.

### Gaps
- No size for `torch-2.11.0+cu128-...-win_amd64.whl` or for cu130 Windows wheels was found; the 3.27 GB figure is for 2.7.1. Expect similar or larger for 2.11/cu130 but this is unverified.
- No measured on-disk size of a bare diffusers environment was found in this session; the ≈3.5–4 GB estimate is arithmetic on the cited pieces.
- ComfyUI's own installed size (repo + non-torch requirements) was not separately sourced; it is inferred from Desktop's 4.85 GB minus torch.

---

## Key question 5: Apps using ONNX/DirectML (Amuse) or Core ML/Metal (Draw Things): sizes, trade-offs, relevance for a Windows-first RTX 5090 setup

### Takeaway
Amuse (TensorStack, .NET + ONNX Runtime/DirectML, now community-maintained after TensorStack
discontinued it) and Draw Things (Swift, custom NNC engine with Metal FlashAttention, GPL-3.0 core)
prove that a consumer app can bundle inference with no Python at all and start instantly; the cost
is a private model-conversion pipeline (ONNX-DirectML or `.ckpt` q6p quantised files) and lagging
support for new architectures. For an NVIDIA 5090 on Windows, DirectML is slower and more
constrained than CUDA and gives up the ComfyUI model ecosystem, so these are precedents for the *UX
bar*, not a recommended runtime.

### Cited Findings
**Amuse**
- Amuse models are distributed as "Onnx DirectML Optimized" conversions (e.g. StableDiffusion-XL-amuse, Tempest-XL-amuse) on TensorStack's Hugging Face. — [huggingface.co/TensorStack](https://huggingface.co/TensorStack/StableDiffusion-XL-amuse)
- Amuse 3.1 hardware guidance (AMD doc quoted on a PCGH forum, secondary): recommended Ryzen AI 300 with ≥24 GB RAM or Ryzen 8040 with 32 GB; Radeon RX 7000/9000; FLUX.1 minimum "24GB VRAM, 32GB RAM, and 40GB of disk space". — [extreme.pcgameshardware.de](https://extreme.pcgameshardware.de/threads/663936)
- Amuse 3.2.0 (heise, 2026): development "had originally been discontinued by manufacturer TensorStack-AI and the sources published"; an open-source project continues it; all components migrated to .NET 10; updated to OnnxRuntime 1.23.0; AMD-specific optimisations such as "AMD XDNA Super Resolution" removed; a brief test on a weak AMD system gave "mixed" results. — [heise.de](https://heise.de/-11245759)
- TensorStack's inference libraries (OnnxStack etc.) are .NET NuGet packages; OnnxStack v0.39.0 used OnnxRuntime 1.18.0. — [nuget.org TensorStack profile](https://www-0.nuget.org/profiles/TensorStack); [OnnxStack releases](https://github.com/TensorStack-AI/OnnxStack/releases?page=1)
- DirectML runs on any DirectX 12 GPU (AMD, Intel, NVIDIA, Qualcomm). — [microsoft/DirectML issue #30](https://github.com/microsoft/directml/issues/30)
- The Krita plugin's DirectML path is described as "limited, 12+ GB VRAM recommended", i.e. the PyTorch-DirectML route is a second-class backend even in its best-known consumer deployment. — [krita-ai-diffusion README](https://github.com/Acly/krita-ai-diffusion/blob/main/README.md)

**Draw Things**
- The community repo is Swift (`swift build --product draw-things-cli`, Bazel), "re-implementation of image generation models, samplers", licensed "GNU General Public License version 3 (GPL-v3)" with alternative licences available from the author; macOS requires Xcode; the Linux gRPC server ships as Docker for RTX 20xx–40xx and A100–H100; models are `.ckpt` files with quantisation tags such as `flux_2_klein_4b_q6p.ckpt`; a "Server-Offload" feature lets the app use a self-hosted gRPC server. — [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community)
- Engine: s4nnc is "a Swift interface to the libnnc C runtime" (author Liu Liu). — [liuliu/s4nnc](https://github.com/liuliu/s4nnc)
- Draw Things replaced Apple's MPS kernels with Metal FlashAttention (open-source Metal shaders by Philip Turner) in Aug 2023: v1.20230807.0 made generation "43–120% faster" without MFA vs with; vs Core ML on ANE it was slightly slower on base M1/M2 (≈15.2 s vs 12.8 s, 25 steps 512²) but 20–40% faster than Core ML-on-GPU on M1 Pro and up, and avoids Core ML's first-time model loading/conversion cost. — [engineering.drawthings.ai, Metal FlashAttention](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18)
- Hugging Face's Core ML guide: Core ML models for Swift apps are the "compiled" variant; PyTorch MPS is for Python and "isn't suitable for native apps". — [diffusers Core ML docs](https://huggingface.co/docs/diffusers/v0.39.0/optimization/coreml.md)

### Inferences
- Both no-Python precedents own their entire model pipeline (conversion, quantisation, kernel work). That is why they can start in under a second and never show a "installing PyTorch" screen, and also why Draw Things needs a gRPC server for CUDA and Amuse lost its AMD features when the vendor left. For a one-person fork targeting bleeding-edge ComfyUI-native models (Krea 2, Qwen-Image-2.1, Z-Image) this pipeline cost is prohibitive; the ComfyUI ecosystem is the asset.
- Relevance to a 5090: DirectML has no FP8/NVFP4 path comparable to CUDA's, which matters because the fork's model policy (devlog: NVFP4) depends on CUDA-specific quantisation kernels. ONNX Runtime's CUDA/TensorRT execution providers exist but would again need per-model exports.
- The UX elements worth copying from Amuse/Draw Things/Krita rather than their runtimes: instant launch with inference loading lazily, a single "Install AI features (≈X GB)" prompt with a backend auto-pick, model packs chosen by workload, and a visible "server starting / ready" state.

### Gaps
- Amuse's installer size and current official system-requirements page were not found (amuse-ai.com now redirects to an unrelated site; the open-source repo URL was not surfaced by search).
- Draw Things' app bundle size and any Windows plans were not found; the community repo page does not state them.

---

## Key question 6 (cross-cutting): precedents' installer mechanics, startup, update story, security surface, and why Krita / Invoke / Amuse chose what they chose

### Takeaway
Krita, Invoke and Comfy all converged on "native launcher + bundled uv + managed Python venv +
separate server process over loopback", each for a stated reason: Comfy for uv's speed and cache,
Invoke for a signed cross-platform launcher that owns the venv and self-updates, Krita because it
is itself a plugin that must not touch Krita's bundled Python. Amuse and Draw Things chose native
runtimes because they control their model formats. None embed Python in the host process.

### Cited Findings
- Invoke launcher: "The launcher is an electron application with React UI" (chosen over Tauri for one rendering engine everywhere); "We bundle uv with the build and then call it to install python, create the app venv, and run the app"; "The Launcher checks for updates once on startup and prompts you if one is available" and "will update itself"; Apache-2.0; "Electron uses about 10x more disk space than something like Tauri, but we're still only talking ~150MB max" against "many GB of models"; Windows/macOS builds are signed, Linux is not; troubleshooting = Repair Mode, clear uv cache, change install location. — [invoke-ai/launcher](https://github.com/invoke-ai/launcher)
- Invoke docs: the launcher "is the official launcher to install, update and manage your invoke installation", ships as Windows EXE / macOS DMG (arm64) / Linux AppImage, can adopt an existing install, and warns third-party launchers (Stability Matrix, LynxHub) are unsupported. — [invoke.ai/start-here/installation](https://invoke.ai/start-here/installation)
- Comfy Desktop's stated rationale for uv: speed and a shared local cache; `pip` aliased to `uv pip`. — [blog.comfy.org](https://blog.comfy.org/open-sourcing-v1-desktop/)
- Comfy Desktop auto-updates ComfyUI, ComfyUI-Manager and the uv executable itself; updater state in `%LOCALAPPDATA%\comfyui-electron-updater`. — [Comfy-Org/desktop](https://github.com/Comfy-Org/desktop)
- Comfy-Desktop keeps a snapshot per install so updates can be rolled back. — [Comfy-Org/Comfy-Desktop](https://github.com/Comfy-Org/Comfy-Desktop)
- Krita's plugin lets the user pick any folder for "the large AI models and backend components", and the Windows Store / Steam / Flatpak sandboxes can block its installer, which is the practical reason it installs into user-writable AppData rather than next to Krita. — [docs.interstice.cloud](https://docs.interstice.cloud/installation); [Scribd guide (low quality)](https://www.scribd.com/document/1059520662/First-Test)
- ComfyUI's Windows portable builds bundle Python and are split by vendor (NVIDIA CUDA 13.0 / AMD experimental / Intel). — [comfyui-wiki.com](https://comfyui-wiki.com/install/install-comfyui/install-comfyui-on-windows)
- uv's version list is "frozen for each uv release", so shipping a pinned uv also pins the Python builds the app can fetch, giving reproducible installs. — [uv docs](https://docs.astral.sh/uv/concepts/python-versions/)

### Inferences
- Startup: a sidecar adds a one-time cold start (interpreter + `import torch` + ComfyUI node discovery, typically tens of seconds on first run after install, then seconds) that the app can hide by launching the server at app start or on first generative command; an embedded interpreter pays the same import cost on the UI process and additionally blocks the GIL while doing it.
- Security surface: every precedent binds an HTTP/WebSocket server to loopback; the risks are (a) other local processes or browser pages reaching `127.0.0.1:8188` (ComfyUI has no auth by default), (b) custom-node code execution, (c) a port collision with a user's own ComfyUI on 8188. Mitigations for an app-managed instance: bind to `127.0.0.1` on an ephemeral port chosen by the parent, pass it to the child, keep the frontend disabled or unexposed, install only the fork's vetted custom nodes, and kill the child with the parent (Windows job object). None of the surveyed products documents such hardening; this is an area where PhotoCraft can exceed the precedents.
- Why Krita chose a managed ComfyUI: it is a plugin inside a Qt app with its own bundled Python, so embedding torch in Krita's interpreter was never an option, and a private ComfyUI gives it the node ecosystem (its own `comfyui-tooling-nodes`) and the "point me at your existing server" escape hatch — the same two properties the fork already relies on.
- Why Invoke chose a launcher: it is a full web app with its own pipeline (not ComfyUI), so the launcher is purely an installer/updater; Electron was accepted at ~150 MB because models dominate.
- Why Amuse chose ONNX/DirectML: it was built by/for AMD on Windows where CUDA is unavailable and ROCm-on-Windows was immature; once AMD/TensorStack stepped back, that choice left it without a maintainer and without new-architecture support, which is the main caution for a native-runtime strategy.
- RAM/disk cost summary for the user (sidecar): ≈5 GB disk for runtime + 10–50 GB models; RAM = a second process holding torch and the loaded model's CPU-side weights (several GB) in addition to the editor, which is why ComfyUI's model offload settings matter on 32 GB machines; an in-process design would hold the same memory in the editor's address space without isolation.

### Gaps
- No measured cold-start numbers (seconds from spawn to `/system_stats` ready) were found for Krita's managed server or Comfy-Desktop; the fork's own `docs/comfy/benchmarks.md` would be the place to measure this locally.
- No documented loopback-hardening (ephemeral port, token auth) in any surveyed product; absence noted, not confirmed as a non-feature.
- CUDA runtime redistribution terms (NVIDIA CUDA Toolkit EULA, which permits redistributing the runtime libraries listed in its attachment) were not fetched in this session and should be checked before bundling `cudart`/`cublas`/`cudnn` DLLs directly instead of via PyTorch's wheels, which already carry them.
