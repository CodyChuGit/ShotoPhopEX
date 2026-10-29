# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""Overnight 2026-10-27 Q6: Generative Upscale live. A stock picture halved (the ground truth is the
original), then enlarged 2x back by the app's CLI two ways: plain bicubic (image.imageSize) and
generate.upscale on the native engine (2.1 edit, the Restore LoRA at 0.5 when installed). Prints
each one's PSNR against the original and writes 200 % crops side by side. argv: the CLI, the
picture, the output folder. Run with ComfyUI's Python (numpy, Pillow), -I, a worker attached
through PHOTOCRAFT_NATIVE_ENGINE.
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

RUNTIME = ";".join([
    "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib",
    "C:/Users/5090/scoop/apps/cuda/current/bin/x64",
    "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib",
])
PREFS = {
    "integrations.engine": "native",
    "integrations.allowResearchModels": True,
    "integrations.runtimePath": RUNTIME,
    "integrations.modelsPath": "C:/Users/5090/Documents/_eval/native-models",
}


def psnr(a, b):
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    cli, picture, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    truth = Image.open(picture).convert("RGB")
    w, h = truth.size[0] // 2 * 2, truth.size[1] // 2 * 2
    truth = truth.crop((0, 0, w, h))
    small = os.path.join(out, "small.png")
    truth.resize((w // 2, h // 2), Image.LANCZOS).save(small)
    setp = ["--cmd", "prefs.set", "--params", json.dumps({"values": PREFS})]
    runs = {
        "bicubic": ["--cmd", "image.imageSize", "--params", json.dumps({"width": w, "height": h, "resample": "bicubic"})],
        "generative": ["--cmd", "generate.upscale", "--params", json.dumps({"scale": 2, "seed": 21})],
    }
    results = {}
    for name, cmd in runs.items():
        dest = os.path.join(out, f"{name}.png")
        t0 = time.time()
        r = subprocess.run([cli, "run", small] + setp + cmd + ["--out", dest], capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0 and os.path.exists(dest)
        line = [l for l in (r.stdout or "").splitlines() if "upscale" in l or "tiles" in l][-1:]
        print(f"{name}\t{time.time() - t0:.1f}s\t{'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-400:]}\t{' '.join(line)[:300]}", flush=True)
        if ok:
            results[name] = Image.open(dest).convert("RGB")
            print(f"{name}\tPSNR vs original {psnr(truth, results[name]):.2f} dB\t{results[name].size}")
    # 200 % crops of the middle: original, bicubic, generative.
    cx, cy, s = w // 2 - 160, h // 2 - 120, 320
    tiles = [truth] + [results[k] for k in ("bicubic", "generative") if k in results]
    sheet = Image.new("RGB", (len(tiles) * s * 2, int(s * 1.5)))
    for i, im in enumerate(tiles):
        sheet.paste(im.resize(truth.size).crop((cx, cy, cx + s, cy + int(s * 0.75))).resize((s * 2, int(s * 1.5)), Image.NEAREST), (i * s * 2, 0))
    sheet.save(os.path.join(out, "crops.png"))


if __name__ == "__main__":
    main()
