# stable-diffusion.cpp survey (leejet, commit b13a482, MIT)

Read-only survey of `C:\Users\5090\Documents\_eval\stable-diffusion.cpp` for plan §3.3 (day-0 lane)
and §5.3 (benchmark). Paths below are relative to that checkout; `file:line` cites the source read.
Date: 2026-10-18. Nothing was built or run.

## 1. Build

**Top-level options** (`CMakeLists.txt:83-99`): `SD_CUDA`, `SD_VULKAN`, `SD_BUILD_SHARED_LIBS`
(default OFF = static), `SD_BUILD_SHARED_GGML_LIB`, `SD_BUILD_EXAMPLES` (ON when standalone),
`SD_USE_UPSTREAM_GGML` (OFF; ON disables INT8 convrot, FP8 GGUF and SageAttention,
`cmake/ggml.cmake:24-27`, `docs/build.md:33-45`). `SD_CUDA=ON` just sets `GGML_CUDA=ON`
(`CMakeLists.txt:108-111`) and links `CUDA::cuda_driver` publicly (`:341-346`). Building with
patched ggml also gives native SageAttention with no extra option (`docs/sage_attention.md:22-36`).

**CUDA archs** (`ggml/src/ggml-cuda/CMakeLists.txt:21-70`): if `CMAKE_CUDA_ARCHITECTURES` is unset
and `GGML_NATIVE` is on (the default off-CI, `ggml/CMakeLists.txt:123`) it uses `native`; otherwise
a list that adds `120a-real` only when `CUDAToolkit_VERSION >= 12.8` (`:53-65`, comment `:32`:
"120 == Blackwell, needs CUDA v12.8"). Plain `120` is rewritten to `120a` (`:87-104`). So **sm_120
needs CUDA >= 12.8**; CUDA 13 drops the `50/61/70-virtual` entries (`:43-45`). Blackwell-specific
MMA paths exist (`ggml-cuda/common.cuh:61,377-380`); NVFP4 MMQ kernels are compiled
(`template-instances/mmq-instance-nvfp4.cu`).

**Flash attention is both**: build option `GGML_CUDA_FA` (ON, `ggml/CMakeLists.txt:206`; OFF defines
`GGML_CUDA_NO_FA`, `ggml-cuda/CMakeLists.txt:158-160`) compiles the kernels; the per-context string
`sd_ctx_params_t.flash_attn_mode` = `on|off|auto[,module=…]` selects them at runtime
(`include/stable-diffusion.h:254`, `docs/backend.md:262-333`). `auto` enables FA on supported CUDA MMA
configs without timing (`backend.md:292-294`). `--fa`/`--diffusion-fa` are deprecated aliases
(`backend.md:320-322`). `GGML_CUDA_GRAPHS` defaults OFF here (`ggml/CMakeLists.txt:117-118,210`).

**Exact configure lines** (from the release CI, `.github/workflows/build.yml:349-350,397`):

```powershell
# Windows, VS 2022 x64 dev shell, CUDA 12.8+ (13.x also matches the >=12.8 branch)
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=/bigobj -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON `
  -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120a-real `
  -DCMAKE_CUDA_FLAGS='-Xcudafe "--diag_suppress=177" -Xcudafe "--diag_suppress=550"'
cmake --build build --config Release
```
```sh
# Linux (gcc/clang + nvcc), same options; $ORIGIN rpath as in build.yml:88
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON \
  -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build build
```
For a 5090-only dev box `-DCMAKE_CUDA_ARCHITECTURES=120a-real` (or leave unset with `GGML_NATIVE=ON`)
cuts build time; a shipped build should use the CI list `61;70;75;80;86;89;90;100;120`.
Add `-DSD_WEBP=OFF -DSD_WEBM=OFF` to skip the example-only image codecs (`docs/build.md:53-63`); the
library itself does not need them. MSVC compiles the lib with `/EHsc-` because "ggml backends can
throw C++ exceptions through their C API" (`CMakeLists.txt:299-303`), see §6.

**Outputs** (all in `build/bin`, `CMakeLists.txt:23-24`): `stable-diffusion.dll` / `libstable-diffusion.so`
(target `stable-diffusion`, `:216,287`), `sd-cli(.exe)`, `sd-server(.exe)`; with `GGML_BACKEND_DL`
the ggml pieces are `ggml.dll`, `ggml-base.dll`, `ggml-cpu-<variant>.dll`, `ggml-cuda.dll` (MODULE
libs loaded at runtime, `ggml/src/CMakeLists.txt:381-392`), plus CUDA's `cudart64_*`, `cublas64_*`,
`cublasLt64_*` (CI ships them separately, `build.yml:425-432`). Export macro: compile the DLL with
`SD_BUILD_SHARED_LIB` + `SD_BUILD_DLL`; consumers define `SD_BUILD_SHARED_LIB` only
(`stable-diffusion.h:4-20`). A CMake package + `.pc` are installed (`CMakeLists.txt:397-419`).

**Prebuilt**: README points at GitHub releases (`README.md:136`); the CI release job publishes
`sd-master-<hash>-bin-win-cuda12-x64.zip` (CUDA 12.8.1, arch list includes 120) and
`cudart-sd-bin-win-cu12-x64.zip` (`build.yml:338-448`). Linux zips are CPU and Vulkan only; Linux
CUDA ships as Docker images `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` (`build.yml:183-273`),
so a Linux CUDA shared lib must be built by us.

## 2. Public C API (`include/stable-diffusion.h`)

**Context**: `sd_ctx_params_t` (`:200-256`) → `sd_ctx_params_init` (`:515`; defaults in
`src/stable-diffusion.cpp:326-355`: `n_threads` = physical cores, `rng_type = CUDA_RNG`,
`auto_fit = true`, `batched_cfg = true`, `conditioning_cache_size = 4`) → `new_sd_ctx` (`:518`, returns
NULL on failure) → `free_sd_ctx` (`:519`). Fields we use: `diffusion_model_path`, `llm_path`,
`llm_vision_path` (mmproj for GGUF TEs when editing), `vae_path`, `taesd_path`, `wtype` (`sd_type_t`,
`SD_TYPE_COUNT` = keep file type, `:101-148,225`), `tensor_type_rules`, `lora_apply_mode`
(`:166-171`), `flash_attn_mode`, `conv_direct_mode`, `sage_attn`, `backend` / `params_backend`
(strings like `"cuda0"`, `"cpu"`, `"diffusion=cuda0,te=cpu"`, `backend.md:11-57`), `max_vram`
(string, GiB budget), `enable_mmap`, `eager_load`, `model_args` (e.g. `qwen_image_zero_cond_t=true`),
`linear_scale`/`attn_scale` (NaN workaround, `docs/troubleshooting.md:12-48`). There is **no
offload flag**: `--offload-to-cpu` is the CLI prepending `*=cpu` to `params_backend`
(`examples/common/common.cpp:931-933`, `backend.md:426-438`). `sd_ctx_params_to_str` returns a
`malloc`ed string (`:516`). LoRAs are **per request**, not per context (`sd_lora_t{is_high_noise,
multiplier, path}`, `:361-365`, `sd_img_gen_params_t.loras/lora_count`, `:396-397`); the CLI
resolves `<lora:name:w>` tags into that array (`common.cpp:2335,2713-2724`).

**Generation**: `sd_img_gen_params_t` (`:395-425`) → `sd_img_gen_params_init` (`:531`; defaults
`src/stable-diffusion.cpp:514-535`: 512x512, `strength 0.75`, `seed -1` = random, `batch_count 1`,
`clip_skip -1`) → `generate_image(ctx, &p, &images_out, &n)` (`:533-536`). Covers all our tasks:
- txt2img: `prompt`, `negative_prompt`, `width`, `height`, `seed`, `batch_count` (seed+i per image,
  `src/pipeline/image.cpp:865`), `sample_params`.
- img2img: `init_image` (`sd_image_t{w,h,channel,data}`, 8-bit interleaved, `:265-270`) + `strength`.
- inpaint: `mask_image` (1 channel, 0/255, rounded at `image.cpp:194-196`); a missing mask is all
  ones (`image.cpp:209-210`); for flow models the mask becomes `denoise_mask` latent blending
  (`image.cpp:213,415`); dedicated inpaint checkpoints get the concat path (`image.cpp:339-380`).
  Mask must match init geometry (`docs/image_preprocessing.md:103-106`).
- edit with references: `ref_images`/`ref_images_count` + `ref_image_args` (preset string,
  `:402-404`, `docs/edit.md:28-101`); `image_preprocess.rules` for resize/crop (`:272-275`).
- sampling: `sd_sample_params_t` (`:299-310`): `guidance.txt_cfg` (CFG), `img_cfg`,
  `distilled_guidance`, `slg`, `scheduler` (`:65-84`), `sample_method` (`:38-61`), `sample_steps`,
  `eta`, `flow_shift` (INFINITY = model default), `custom_sigmas[]` (what `--sigmas` fills,
  `common.cpp:2742-2743`), `extra_sample_args`. `SCHEDULER_COUNT`/`SAMPLE_METHOD_COUNT` = ask
  the model (`sd_get_default_sample_method/scheduler`, `:528-529`).
- also: `vae_tiling_params` (`:173-184`), `cache` (EasyCache etc., `:334-359`), `hires`
  (`:381-393`), `circular_x/y`, `qwen_image_layers`.
- string<->enum helpers `str_to_sample_method`, `str_to_scheduler`, `str_to_sd_type` … (`:495-510`).

**Callbacks** (process-global, `:464-484`): `sd_set_log_callback(cb(level,text,data))`,
`sd_set_progress_callback(cb(step,steps,time_s_per_step,data))`, `sd_set_preview_callback(cb, mode,
interval, denoised, noisy, data)` + `sd_get_preview_info()`, `sd_set_backend_eval_callback`.
Without a log callback nothing is printed (`src/core/util.cpp:662`); without a progress callback the
lib prints a bar to **stdout** (`util.cpp:596-614`), so always install both.

**Cancellation exists**: `sd_cancel_generation(ctx, SD_CANCEL_ALL | SD_CANCEL_NEW_LATENTS |
SD_CANCEL_RESET)` (`:538-547`) sets an atomic (`src/pipeline/diffusion_engine.h:260`,
`diffusion_engine.cpp:336-344`) that is polled per sampler step (`diffusion_engine.cpp:2528`), per
batch item (`image.cpp:852-862`) and around encode/decode (`image.cpp:561-767,920-965`); it is reset at
the start of each `generate_image` (`image.cpp:803`). Cancelled runs return `false`. Safe to call
from another thread (that is the only way to use it, since `generate_image` blocks).

**Ownership**: `images_out` is one `calloc`ed array of `sd_image_t` whose `data` buffers are
`malloc`ed by the library (`image.cpp:631-643`); free **only** with `free_sd_images(arr, n)`
(`:641-643`, documented CRT-mismatch warning). Input image buffers stay owned by the caller for the
duration of the call. `sd_get_model_version_name` returns a static string (`:525-526`).
`sd_list_devices` fills a caller buffer (`:635-639`). Other entry points: `generate_video`,
upscaler (`:558-576`), ADetailer (`:578-597`), `convert`/`convert_with_components` (`:599-618`,
offline GGUF quantisation), imatrix (`:627-630`), `sd_version`/`sd_commit` (`:632-633`).

## 3. Model support: Krea 2, Qwen-Image-2.1, Qwen Image Edit

Version is detected from tensor names at load (`src/model_loader.cpp:479-482` for Krea2,
`txtfusion.projector.weight`); Raw and Turbo are the **same architecture** (`Krea2Config::detect_from_weights`,
`src/model/diffusion/krea2.hpp:29-134` only sizes blocks/heads), Turbo is just steps/CFG.

| Model | DiT (`diffusion_model_path`) | Text encoder (`llm_path`) | VAE (`vae_path`) | Sources (docs) |
|---|---|---|---|---|
| Krea 2 Raw/Turbo | `krea/Krea-2-*` safetensors, `realrebelai/KREA-2_GGUFs` (BASE/TURBO), or ComfyUI `krea2_turbo_int8_convrot.safetensors` | Qwen3-VL 4B: Comfy-Org/Krea-2 `text_encoders` safetensors or `Qwen/Qwen3-VL-4B-Instruct-GGUF` | `wan_2.1_vae.safetensors` (Comfy-Org Wan 2.1 repack) | `docs/krea2.md:3-17`, `docs/int8_convrot.md:102-106` |
| Qwen-Image-2.1 | `Comfy-Org/Qwen-Image-2.1/diffusion_models` (bf16 / `qwen_image_2.1_int8_convrot.safetensors`) or `leejet/Qwen-Image-2.1-GGUF` | Qwen3-VL-8B: Comfy-Org safetensors (BF16 or INT8 convrot) or `Qwen/Qwen3-VL-8B-Instruct-GGUF` + `mmproj-…-F16.gguf` via `llm_vision_path` for editing | `qwen_image_2.1_vae_bf16.safetensors` (own VAE; not the Qwen-Image/Wan one) | `docs/qwen_image_2.1.md:7-17` |
| Qwen Image Edit / 2509 / 2511 | `Comfy-Org/Qwen-Image-Edit_ComfyUI/split_files/diffusion_models` or QuantStack/unsloth GGUF | Qwen2.5-VL 7B: Comfy-Org `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` or mradermacher GGUF (+ mmproj) | `qwen_image_vae.safetensors` | `docs/qwen_image_edit.md:5-19` |

- **ComfyUI repack files load directly**: every doc above lists Comfy-Org `split_files` as the
  safetensors source; ComfyUI `int8_tensorwise` + `convrot` safetensors load without conversion
  (`docs/int8_convrot.md:3,102-106`). FP8 safetensors load (§4). Safetensors bf16 loads as-is.
- **Formats**: `.safetensors`, `.gguf`, `.ckpt/.pt` (`README.md:103-106`); `wtype` can requantise
  at load time (`docs/quantization_and_gguf.md:1-10`) or pre-convert with `sd-cli -M convert`.
- **Krea 2 defaults**: FLUX_FLOW prediction, `default_flow_shift = 1.15` (`diffusion_engine.cpp:1528-1546`),
  default sampler `euler` (DiT, `src/pipeline/request.cpp:50-51`), default scheduler `discrete`
  (`request.cpp:79`); Turbo recipe `--steps 8 --cfg-scale 1` (`int8_convrot.md:105`), Raw uses
  default 20 steps + CFG (docs give no CFG; `txt_cfg` default 7.0 is SD-era, check Krea's card).
  Krea 2 editing exists only through community edit LoRAs (`ref_image_args` preset
  `krea2_ostris_edit` default, `diffusion_engine.cpp:3219-3223`, `docs/edit.md:19,56-58`).
- **Qwen-Image-2.1 defaults**: FLOW_PRED, `default_flow_shift = 3.0` (`diffusion_engine.cpp:1517-1518`;
  the doc says the resolution-dependent schedule is automatic, `qwen_image_2.1.md:23`), sampler
  `euler`, scheduler **`flux`** (`request.cpp:68-69`). Docs: `--cfg-scale 6.0 --sampling-method euler`,
  dims divisible by 32; Turbo: `--sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0` (preset =
  `1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0`, `common.cpp:1522-1523`;
  via API pass `custom_sigmas`). Editing: `ref_images` + vision weights; preset `qwen`
  (VLM yes, RoPE `increase`, area resize, `edit.md:91`). Prefix KV cache for text/ref tokens is on by
  default (`model_args qwen_image_2_1_prefix_cache[_type]`, `qwen_image_2.1.md:55-81`; f16 cache
  ~2 GiB per condition at 4096 prefix tokens). RGBA output supported via prompt (`:83-89`).
  TAEQI2.1 for fast decode/preview (`docs/taesd.md:41-51`).
- **Qwen Image Edit**: `--cfg-scale 2.5 --sampling-method euler --flow-shift 3`, `-r <image>`;
  2511 needs `model_args qwen_image_zero_cond_t=true` (`qwen_image_edit.md:26,35,42-45`).
- **LoRA**: generic LoRA/LoHa/LoKr with name conversion (`src/model/adapter/lora.hpp`, Qwen 2.1
  fused-MLP special case `:202`); on quantised or INT8 weights it is applied at runtime
  (`docs/lora.md:15-26`, `int8_convrot.md:96`), so no re-merge cost per request.
- **VRAM / resolution**: no per-model numbers in docs; the mechanism is `params_backend cpu` +
  automatic segmented execution + `max_vram` (`docs/performance.md:100-172`). Every doc example uses
  `--offload-to-cpu`, which the author says costs no speed (`performance.md:100-102`); on a 32 GB 5090
  we should instead keep params on `cuda0` and measure both.

## 4. Precision / quantisation

- GGUF types: everything in `sd_type_t` (`stable-diffusion.h:101-148`): F32/F16/BF16, Q4_0…Q8_0,
  K-quants Q2_K…Q6_K, IQ*, TQ*, MXFP4 (39), **NVFP4 (40)**, Q1_0/Q2_0, F8_E4M3 (43), F8_E5M2 (44).
  So NVFP4 is **declared** and the patched ggml-cuda has NVFP4 MMQ kernels
  (`ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu`, `common.cuh:377-380`), but no doc
  mentions NVFP4 model files or a conversion path for diffusion models; treat it as unverified
  (needs a live `convert --type nvfp4` test; ComfyUI's NVFP4 checkpoints are a different layout).
- FP8: safetensors `F8_E4M3`/`E5M2` tensors are decoded at load (`model_loader.cpp:40-87,1313-1314`);
  with patched ggml FP8 can stay FP8 in GGUF/`wtype`; upstream-ggml mode converts to F16 and rejects
  FP8 GGUF (`docs/build.md:36-40`).
- INT8 convrot (ComfyUI `int8_tensorwise`, H256): native CUDA path on CC ≥ 7.5 using cuBLAS
  i8×i8→i32 (`docs/int8_convrot.md:89-98`, `common.cuh:382-389`); LoRA applied on the side.
- SageAttention 2++ on SM89+/CUDA ≥ 12.8 (`docs/sage_attention.md:9-20`), `sage_attn = true`.
- Offload: `params_backend` per module = `cpu` | `disk` | device; auto-fit planner
  (`backend.md:130-216`); `max_vram` budget; `enable_mmap`.

## 5. Performance notes in docs

No absolute s/it numbers for Krea 2 or Qwen-Image-2.1 anywhere in `docs/`; no ComfyUI speed
comparison (ComfyUI appears only as RNG reference: `--rng cpu` = "comfyui RNG", `README.md:127-129`,
useful for seed parity in the harness). What is documented: FA memory savings (`performance.md:1-27`,
e.g. flux 768² −600 MB), `--flash-attn auto` is heuristic ("compare on and off", `:8-14`),
`--conv-direct` probe (`:29-40`), batched CFG default (`:42-50`), VAE tiling with automatic OOM
retries (`:52-98`), `--offload-to-cpu` "without reducing generation speed" (`:100-102`), segmented
execution and prefetch (`:136-172`), Qwen 2.1 prefix cache (`qwen_image_2.1.md:55-81`), TAESD/TAEQI2.1
(`taesd.md`), step caches EasyCache/UCache/DBCache/TaylorSeer (`docs/caching.md`). The benchmark
protocol the docs themselves recommend: same seed/size/sampler, time steps after the first, report
load and first-step init separately (`sage_attention.md:56-61`).

## 6. Risks for a Rust FFI

- **Global state**: log/progress/preview/eval callbacks are static globals (`util.cpp:376-387,
  658-659,707-716`), one set per process, not per context. Fine in a dedicated worker; do not
  share the DLL with anything else that sets them.
- **Thread safety**: a context has a `recursive_mutex` + `executing_` flag; a second
  `generate_image` on the same context while one runs **returns false immediately** (try_lock,
  `diffusion_engine.h:216-254`) rather than queuing. Two contexts in one process are independent
  but share ggml backends only through their own managers (`backend.md:395-413`). Callbacks fire on
  the calling thread. `sd_cancel_generation` is the only call meant for a second thread; ControlNet
  hot-swap is explicitly not (`stable-diffusion.h:490`).
- **Aborts**: `GGML_ABORT`/`GGML_ASSERT` call `abort()` (e.g. `image.cpp:115`,
  `src/core/ggml_extend_backend.cpp`); CUDA OOM usually surfaces as `false` + log, but driver faults
  and asserts kill the process. Confirms plan §3.4: run in the worker, never in the editor.
- **Exceptions**: C++ exceptions cross the ggml C API by design (`CMakeLists.txt:301-302`, built
  `/EHsc-`); `build_core_runners` catches `std::exception` (`src/pipeline/model_builders.cpp:85,528`)
  but the public functions are not blanket `noexcept`-wrapped, so an uncaught throw unwinds into
  Rust = UB/abort. Mitigation is the same: process boundary; keep FFI calls on one thread.
- **stdout**: without a progress callback the lib writes `\r…` bars to stdout (`util.cpp:596-597`);
  always set both callbacks before `new_sd_ctx` (CLI does, `examples/cli/main.cpp:651,691`).
- **Memory**: runners release workspaces at the end of each `generate_image` (`ExecutionScope`
  → `end_runners`, `diffusion_engine.h:249-253`); conditioning cache keeps 4 entries by default
  (`conditioning_cache_size`, the CLI sets 0, `main.cpp:648`); Qwen 2.1 prefix caches are freed per
  sampling run. Weight replicas on GPU persist by design (that is the warm model). Returned images
  are CRT-`malloc`ed; `free_sd_images` only.
- **API stability**: README says "API and command-line option may change frequently"
  (`README.md:13-14`); structs are plain C with fields appended over time (`backend.md:258-260,
  323-328` describe appended fields), no size/version field, so **pin the commit, regenerate
  bindings with bindgen from the pinned header, and never hand-write struct layouts**. Deprecated
  CLI flags show the churn (`--fa`, `--vae-conv-direct`, old CPU fields removed from the C API,
  `backend.md:437`).
- **diffusion-rs**: README lists `newfla/diffusion-rs` as the Rust binding (`README.md:182`); no
  statement in-tree about which commit it tracks, and given the field churn above a generated
  binding against our pinned commit is safer than a crate dependency.
- Licence: MIT (`LICENSE`); ggml MIT (`ggml/LICENSE`, both copied into release zips, `build.yml:108-109`).

## 7. Proposed minimal Rust FFI surface and benchmark commands

Worker-side wrapper (`gn-sdcpp`), bindgen over the pinned `stable-diffusion.h`, exposing to the
engine trait only:

1. `sdcpp::init_callbacks(log: fn, progress: fn)` → `sd_set_log_callback`, `sd_set_progress_callback`
   (once per process, before any context).
2. `sdcpp::devices() -> Vec<String>` → `sd_list_devices` (health check, picks `cuda0`).
3. `sdcpp::Context::open(ModelSet{dit, te, te_vision?, vae, taesd?}, Opts{wtype, backend,
   params_backend, max_vram, flash_attn_mode, sage_attn, model_args})` → `sd_ctx_params_init` +
   `new_sd_ctx`; keep `sd_get_model_version_name` for the badge.
4. `Context::defaults() -> (SampleMethod, Scheduler)` → `sd_get_default_sample_method/scheduler`.
5. `Context::generate(Request{prompt, negative, w, h, seed, steps, cfg, sampler, scheduler,
   flow_shift, sigmas, init: Option<Rgb8>, strength, mask: Option<Gray8>, refs: Vec<Rgb8>,
   ref_image_args, loras: Vec<(path, weight)>, vae_tiling}) -> Result<Vec<Rgba8>>` →
   `sd_img_gen_params_init` + `generate_image`, copy pixels out, `free_sd_images`.
6. `Context::cancel()` → `sd_cancel_generation(SD_CANCEL_ALL)` from the control thread.
7. `Context::close()` → `free_sd_ctx` (Drop).
8. `sdcpp::version() -> (String, String)` → `sd_version`, `sd_commit` (recorded on every run).
9. Optional later: `sd_set_preview_callback` (live previews, `PREVIEW_TAE` with TAEQI2.1), and
   `convert_with_components` for an in-app "quantise to GGUF" action.

Benchmark commands (plan §5.3; 1024×1024, fixed seed, params on GPU; repeat with
`--offload-to-cpu` and `--flash-attn off` as the two ablations; `--rng cpu` matches ComfyUI noise):

```powershell
# Krea 2 Turbo, GGUF Q8_0 (realrebelai/KREA-2_GGUFs/TURBO), Qwen3-VL-4B GGUF, Wan 2.1 VAE
.\build\bin\sd-cli.exe --diffusion-model models\Krea-2-Turbo-Q8_0.gguf `
  --llm models\Qwen3-VL-4B-Instruct-Q8_0.gguf --vae models\wan_2.1_vae.safetensors `
  -p "<harness prompt>" -W 1024 -H 1024 --steps 8 --cfg-scale 1 --sampling-method euler `
  --seed 42 --rng cpu --backend cuda0 --params-backend cuda0 --flash-attn on -v -o out\krea2_turbo_q8.png
# same with the official safetensors: --diffusion-model models\krea2_turbo_int8_convrot.safetensors

# Qwen-Image-2.1, GGUF Q8_0 (leejet/Qwen-Image-2.1-GGUF), Qwen3-VL-8B GGUF, 2.1 VAE
.\build\bin\sd-cli.exe --diffusion-model models\qwen_image_2.1-Q8_0.gguf `
  --llm models\Qwen3VL-8B-Instruct-Q8_0.gguf --vae models\qwen_image_2.1_vae_bf16.safetensors `
  -p "<harness prompt>" -W 1024 -H 1024 --steps 20 --cfg-scale 6.0 --sampling-method euler `
  --seed 42 --rng cpu --backend cuda0 --params-backend cuda0 --flash-attn on -v -o out\qwen21_q8.png
# Turbo variant: add --sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0
```

Read s/it from the progress line (`util.cpp:608-614`), discard step 1, and log `--flash-attn`
decisions from the `[INFO] Using flash attention` line (`performance.md:16-19`). Downloads to
approve, per plan: Krea 2 Turbo Q8_0 (~12 GB) and Qwen3-VL-4B GGUF (~4 GB); Qwen-Image-2.1 Q8_0
and Qwen3-VL-8B GGUF are the second pair.
