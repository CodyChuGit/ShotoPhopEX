## New in 0.1.1

- **ShotoPhopEX for Mac (Apple Silicon, M1 and newer).** A `.dmg` with the generative tools running on Apple's GPU through MLX, compiled into the app: no Python, Homebrew or Xcode. Tuned on an M1 MacBook Air with 16 GB.
  - Generate Image (MOSS Turbo, or FLUX 3 Image Klein for higher quality), Generative Edit, Fill, Remove, Expand, Split into Layers, Clown Pass, Select Subject / by text / by point, Remove Background and Prompt Enhance.
  - Not on the Mac yet: Generative Angles, Upscale, and the big Krea 2 and Qwen-Image-2.1 models.
  - **Help › Get AI Models** offers a **Mac pack** sized for 16 GB.
- **FLUX 3 Image Klein** (Mac) for text-to-image, edits, fills and expands, from mflux's 4-bit MLX files under the FLUX Kommunity License.
- **Licences up front:** Get AI Models lists each model's licence and downloads nothing until you accept them. The Windows and Linux downloads now come with third-party notices and an SBOM (`.spdx.json`).

## Highlights

- **Generative tools, built in:** Generative Fill, Remove, Expand, Split into Layers, Angles, Upscale, Generate Similar and text-to-image, all running natively on your GPU (no Python, no ComfyUI).
- **Models:** Krea 2, Qwen-Image-2.1, MOSS-Image (fast, low-VRAM) and, on a Mac, FLUX 3 Image Klein. Each model keeps its own licence; Qwen-Image-2.1 is under the Qwen Research License (no commercial use).
- **Memory profiles:** the first launch matches models and settings to your VRAM or Mac memory.
- **Object Selection and Select Subject** with SAM 3.1: click an object to select it whole, Next Match for its parts, or drag a box.
- **Live preview** while generating, and a Generations library that keeps every result.

## Install

- **Mac:** open the `.dmg` and drag ShotoPhopEX to Applications. Apple Silicon (M1 or newer) and macOS 14 or later; 16 GB of memory recommended. The app is not notarized, so the first time macOS says it was **Not Opened**: click **Done**, then **System Settings › Privacy & Security › Open Anyway**.
- **Windows:** the MSI installer or the portable zip. Files are not code-signed, so on first run click **More info › Run anyway** when SmartScreen says "Windows protected your PC".
- **Linux:** the portable tar.gz (beta; not yet tested on a GPU).
- On Windows and Linux, generative features need an NVIDIA RTX 30-series card or newer. Model weights are not included: use **Help › Get AI Models**.
- Check downloads against `SHA256SUMS`.
