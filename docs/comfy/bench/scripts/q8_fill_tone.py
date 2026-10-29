# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""Overnight 2026-10-27 Q8: the seam tone match on Generative Fill (toneMatch), Qwen-Image-2.1 on the
native worker: six fills on stock pictures, the same seed with it off and on, and a sheet of
crops around each selection (original, off, on). argv: the CLI, the output folder. Run with
ComfyUI's Python (Pillow), -I, a worker attached through PHOTOCRAFT_NATIVE_ENGINE.
"""

import json
import os
import subprocess
import sys
import time

from PIL import Image, ImageDraw

STOCK = r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
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
# (picture, x, y, w, h as fractions, prompt, seed)
CASES = [
    ("st-dog.jpg", 0.62, 0.72, 0.25, 0.12, "a red ball on the grass", 31),
    ("st-cup.jpg", 0.55, 0.55, 0.25, 0.3, "a green apple on the table", 32),
    ("st-boat.jpg", 0.15, 0.15, 0.25, 0.25, "a seagull flying", 33),
    ("st-sign.jpg", 0.1, 0.5, 0.3, 0.35, "a small wooden door in the wall", 34),
    ("st-beach.jpg", 0.3, 0.7, 0.3, 0.2, "a sandcastle", 35),
    ("st-car.jpg", 0.05, 0.1, 0.3, 0.3, "a hot air balloon in the sky", 36),
]


def main():
    cli, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    setp = ["--cmd", "prefs.set", "--params", json.dumps({"values": PREFS})]
    rows = []
    for pic, fx, fy, fw, fh, prompt, seed in CASES:
        src = os.path.join(STOCK, pic)
        im = Image.open(src).convert("RGB")
        W, H = im.size
        x, y, w, h = int(fx * W), int(fy * H), int(fw * W), int(fh * H)
        made = []
        for tone in (False, True):
            dest = os.path.join(out, f"{pic[:-4]}-tone{int(tone)}.png")
            p = {"prompt": prompt, "seed": seed, "template": "qwen-2.1/fill", "variations": 1, "enhance": False, "toneMatch": tone}
            sel = ["--cmd", "select.rect", "--params", json.dumps({"x": x, "y": y, "width": w, "height": h})]
            t0 = time.time()
            r = subprocess.run([cli, "run", src] + setp + sel + ["--cmd", "generate.fill", "--params", json.dumps(p), "--out", dest], capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok = r.returncode == 0 and os.path.exists(dest)
            print(f"{pic}\ttone={tone}\t{time.time() - t0:.1f}s\t{'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-300:]}", flush=True)
            made.append(Image.open(dest).convert("RGB") if ok else None)
        m = int(0.35 * max(w, h))
        box = (max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m))
        rows.append([im.crop(box)] + [(i.crop(box) if i is not None else Image.new("RGB", (box[2] - box[0], box[3] - box[1]))) for i in made])
    cell = 420
    sheet = Image.new("RGB", (cell * 3, cell * len(rows)), "black")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            t = im.copy()
            t.thumbnail((cell - 4, cell - 4))
            sheet.paste(t, (c * cell, r * cell))
        ImageDraw.Draw(sheet).text((4, r * cell + 4), CASES[r][0] + "  original | off | on", fill=(255, 255, 0))
    sheet.save(os.path.join(out, "sheet.png"))


if __name__ == "__main__":
    main()
