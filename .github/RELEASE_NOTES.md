## New in 0.1.2

- **24 GB cards work now (RTX 3090, 4090).** A lean memory mode loads the smaller text encoders and frees memory between models, so every generative tool runs on 24 GB without running out of memory.
- **Face Swap, Body Swap and Pose** in Generative Edit: put a person from another picture into this one, or repose someone to match a pose picture (Qwen-Image-2.1 with the BFS and PoseStudio LoRAs).
- **Better Angles:** Generative Angles always uses Qwen-Image-Edit-2511 with the Multiple Angles LoRA, and its default settings now give more believable hidden sides.
- **New models:** H3 Image (MiniMax, 32 GB cards), FLUX 3 Image Klein on 8 to 16 GB NVIDIA cards (Edit and Expand), and NVFP4 Krea 2 Turbo on RTX 50-series cards.
- **Quality:** MOSS-Image uses a realism recipe by default, and Krea 2 has a finer detail schedule on Quality Best and Balanced.
- **Select Subject, High Quality:** a soft matte from Qwen-Image-2.1 for hair and fur. Cut Out now keeps the whole named object, and the background is fully transparent.
- **Mac fixes:** better Select Subject and click selection, a Fill that sometimes came back as a flat grey square is caught and redone, and an experimental 8 GB Mac mode.
- **Based on PhotoCraft 0.23.1**, with the installer and app now using the ShotoPhopEX name everywhere. Release files are named `shotophopex-*`; the in-app updater finds them.
- **Known issues:** Generative Fill with Qwen-Image-2.1 on Quality Best can give a flat fill (use Fast or Krea 2). Cut Out with no prompt can keep small pieces next to the subject; naming the subject avoids it. Angles is not on the Mac yet.

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
