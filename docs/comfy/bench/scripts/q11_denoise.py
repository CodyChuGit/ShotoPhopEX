# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""Overnight 2026-10-27: a fill's `denoise` live on the 2511 Lightning fill through ComfyUI. The top
hat of st-hat.jpg asked to become a red one at denoise 1, 0.75 and 0.6 (the same seed), and a
sheet of the crops. argv: the CLI, the output folder. Run with ComfyUI's Python, -I.
"""

import json
import os
import subprocess
import sys
import time

from PIL import Image

STOCK = r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
cli, out = sys.argv[1], sys.argv[2]
template = sys.argv[3] if len(sys.argv) > 3 else "qwen-edit-2511/fill-lightning-8"
levels = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0, 0.75, 0.6]
os.makedirs(out, exist_ok=True)
src = os.path.join(STOCK, "st-hat.jpg")
x, y, w, h = 1201, 422, 205, 174  # the hat (st-objects.json)
prefs = {"integrations.engine": "comfy"}
crops = [Image.open(src).convert("RGB")]
for d in levels:
    dest = os.path.join(out, f"hat-{template.split(chr(47))[-1]}-{d}.png")
    p = {"prompt": "a red top hat", "seed": 41, "template": template, "variations": 1, "enhance": False, "denoise": d}
    args = [cli, "run", src, "--cmd", "prefs.set", "--params", json.dumps({"values": prefs}), "--cmd", "select.rect", "--params", json.dumps({"x": x, "y": y, "width": w, "height": h}),
            "--cmd", "generate.fill", "--params", json.dumps(p), "--out", dest]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and os.path.exists(dest)
    print(f"denoise {d}\t{time.time() - t0:.1f}s\t{'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-300:]}", flush=True)
    if ok:
        crops.append(Image.open(dest).convert("RGB"))
box = (x - 120, y - 80, x + w + 120, y + h + 160)
cell = (box[2] - box[0], box[3] - box[1])
sheet = Image.new("RGB", (cell[0] * len(crops), cell[1]))
for i, im in enumerate(crops):
    sheet.paste(im.crop(box), (i * cell[0], 0))
sheet.save(os.path.join(out, f"denoise-sheet-{template.split(chr(47))[-1]}.png"))
