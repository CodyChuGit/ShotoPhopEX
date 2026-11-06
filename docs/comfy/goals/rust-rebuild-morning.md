# Rust rebuild: morning summary (2026-10-28)

For Cody, after the overnight run. The details and every commit are in
[`rust-rebuild.md`](rust-rebuild.md) §5; nothing is pushed (PhotoshopEX and genai-native are
local only).

## What works now

**The -Native build is one app with no Python, no ComfyUI and no ONNX Runtime.**

- `packaging/windows/package.ps1 -Arch x64 -Native -NoMsi` builds
  `dist\release\photocraft-0.11.1-windows-x64-native-portable.zip`: **557 MB** (target was under
  700). Its `runtime\` folder holds only NVIDIA's libraries (cuBLAS, cuBLASLt, cuRAND, cudart;
  581 MB unpacked). ONNX Runtime and cuDNN are gone from the folder and from the build itself
  (`cargo tree` shows no `ort` in the -Native graph).
- **Clean-profile run** (`docs/comfy/bench/scripts/clean-profile-native.ps1`): the zip unpacked
  fresh, a new empty profile (APPDATA, LOCALAPPDATA, USERPROFILE, TEMP), only Windows' own PATH,
  ComfyUI neither running nor configured, models read from ComfyUI's `models` folder as they
  are. **All 11 generative cases pass**, each in its own process from cold:

  | Case | Model | Time (cold) |
  |---|---|---|
  | Generate Image | Krea 2 Turbo | 21–25 s |
  | Select by text / by point | SAM 3.1 | 2.1–2.3 s / 1.4–1.5 s |
  | Generative Fill | Qwen-Image-2.1 | 32–34 s |
  | Remove (generative) | 2.1 edit | 15 s |
  | Generative Edit / Expand / Angles | 2.1 edit / expand | 12–14 s each |
  | Remove Background | 2.1 matte | 13 s |
  | Generative Upscale 2× | 2.1 edit tiles | 82–83 s |
  | Split into Layers (3) | Qwen-Image-Layered | 62–63 s |

  **No python.exe started. The worker loaded 63 modules and none was onnxruntime, cudnn,
  python or torch.** Spot-checked pictures look right (denim jacket fill, winter scarf edit,
  expanded canvas, the split).
- **Select by text is as fast as ONNX Runtime**: SAM 3.1 now runs in f16 on the GPU by default,
  273–278 ms by text on a new picture through the app, against ONNX Runtime's 249–346 ms (f32
  was 508–515 ms). Points 28 ms. Every mask Codex could compare is at IoU 0.993 or better against
  ComfyUI's own f32 SAM 3.1.
- **Split into Layers** is native under `auto` at parity: 57 s against ComfyUI's 58 s at 640 px.
  Its RoPE kernel is now compiled into the app (the overnight version compiled it at run time with
  NVRTC, which the package does not ship).

Queue status: R0 ✅ R1 ✅ R2 ✅ R3 ✅ (640 px) R4 ✅ R5 ✅, R6 waiting on you.

## What needs you

1. **The licence question (R6).** Qwen-Image-2.1 is research-only (non-commercial). The
   clean-profile run above had "Allow Research-Only Models" on. With it off (the shipped default),
   prompted fill, edit, expand, the removals, angles and upscale still default to
   Qwen-Image-Edit-2511 on ComfyUI, so a permissive build without ComfyUI cannot do them. I left
   the defaults alone. What I did add: such a request now says both ways out ("turn on Allow
   Research-Only Models, or start ComfyUI") instead of only "cannot reach the ComfyUI server".
   The choices as I see them:
   - keep 2.1 opt-in (today), and accept that the permissive build needs ComfyUI for those
     features;
   - port Qwen-Image-Edit-2511 (Apache-2.0) natively. It shares the 60-block DiT, the Qwen2.5-VL
     encoder and the Wan VAE with the Layered port already done. Its vision tower and edit
     pipeline are new, so the work is sizeable but known;
   - or make 2.1 the default in your own local builds only.
2. **SAM 3.1 precision.** I made f16 the default because it meets the speed gate. f32 matches
   ComfyUI to within 3 pixels; f16 changes up to about 650 pixels on one picture. To go back:
   `PHOTOCRAFT_SAM_PRECISION=f32`. It is not a Preferences setting yet (one would need its label
   in all 20 languages); say if you want one. Codex's caveat: 3 of its 12 test pictures had no
   object to find, so 9 positives back the result.
3. **Push and bump.** PhotoshopEX still pins genai-native `f2bf496`; everything since builds
   through the local cargo path patch. A clean checkout elsewhere needs genai-native pushed and
   the rev in `crates/genai/Cargo.toml` bumped (to `766ecb0` or later), your call.
4. **The PC crash at about 23:47.** Kernel-Power 41 with bugcheck 0 (a hard power-off or hang,
   no GPU driver or hardware event logged). Running at the time: my -Native build (CPU and nvcc)
   and the documentation-pass session's 1 MP Layered benchmark on the GPU. Everything after it
   ran one job at a time with no further crash.

## What's left

- **Remove Background with an empty prompt** (2.1 matte) kept nothing on about half the random
  seeds (3 of 5); with a subject named ("the man") 3 of 3 were consistent. This is the route's own
  behaviour, not a rebuild regression. A fix could name the subject first with the 8B chat model,
  or fall back to SAM 3.1.
- **Split at your 1 MP 3:4 size**: native 179 s against ComfyUI's 140 s (480×640: 44 s against
  36 s). The documentation-pass session's Codex job was on it when the PC crashed.
- **R4b**: ComfyUI's int8 repacks (the documentation-pass session; the CPU half landed,
  genai-native `bf68795`, the loader hooks uncommitted in genai-native's tree).
- The clean-profile check used a fresh profile inside your account, not a second Windows
  account (creating an account is a system change I don't make). Same check from a new account:
  unzip `dist\release\…-native-portable.zip` there and run `clean-profile-native.ps1 -Zip <it>`.
- The Krea 2 style reference still needs its LoRA path natively (R2 remainder).

## Update, 2026-10-28 evening

- **Defaults** (Cody's call): engine native, research-only models on; no default path goes to
  2511 or ComfyUI (PhotoshopEX `df40a2bb`). The ComfyUI engine is an opt-in.
- **Speed vs ComfyUI at 864×1152, warm, same template** (`benchmarks.md` §0): split 125 vs
  125 s; image Krea 2 3.8 vs 4.0 s; image 2.1 3.9 vs 4.2 s; fill 3.2 vs 3.3 s; edit 5.1 vs
  5.6 s; image from a ref 5.2 vs 5.6 s; Remove Background 5.1 vs 5.5 s; select by text 0.3 vs
  0.6 s; upscale 2× 40.6 vs 39.0 s (the last, within 4 %). The fixes: genai-native `f66aeb6`
  (split), `03fb72e` (2.1 picture routes: the text encoder stays resident, a conditioning
  cache), `3da83bd` (LoRAs merged into the INT8 weights once, flash prefix), PhotoshopEX
  `3b6f3ca5` (upscale's app side). Every change kept the outputs within 1/255 (most within
  0.006/255; the split byte-identical).
- **The -Native zip rebuilt** (558 MB) and walked from a fresh profile, ComfyUI not configured:
  **11 of 11**; the worker loaded no onnxruntime, cudnn, python or torch module. The watcher saw
  one python.exe with no readable path, consistent with a Codex job's sandboxed fixture
  commands running at the same time (the app can start Python only through the ComfyUI
  sidecar, which was not configured).
- Remove Background with no prompt now names the subject first (`49ecf6d3`).
- **2511 natively** (a benchmark alternative, not a default): spec `90e005a`, vision tower
  `7049357`; the text encoder's image splicing and the DiT step in progress.
- **Krea 2 style reference natively**: implemented (Codex), every CPU gate but one strict
  absolute cap; the GPU comparison against ComfyUI is next.
- Needs Cody: four `python.exe` processes (ComfyUI's Python, since 16:26, PIDs 16508, 3372,
  6984, 5216; most likely an orphaned ComfyUI from a Codex comparison) hold GPU memory; I was
  not permitted to stop them.
