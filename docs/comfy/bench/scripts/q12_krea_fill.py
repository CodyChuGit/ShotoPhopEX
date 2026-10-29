# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""Overnight 2026-10-27: Generative Fill with Krea 2 Turbo (krea2-turbo/fill, noise-mask inpainting)
against the 2511 Lightning fill, through ComfyUI, on the six fill cases of q8_fill_tone.py (same
seeds). A sheet of crops: original, 2511, Krea 2. argv: the CLI, the output folder. Run with
ComfyUI's Python (Pillow), -I.

Measured and withdrawn the same night: Krea 2 paints a separate photo into the selection's box
at denoise 1, 0.8 and 0.65 (a lawn, a window, a day sky in a night street), so the template is
not shipped; it is kept as krea2-turbo-fill.measured.json beside this script. To rerun, put it
back in crates/genai/workflows and template.rs's BUILTIN.
"""

import json
import os
import subprocess
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q8_fill_tone import CASES, STOCK  # noqa: E402

TEMPLATES = ["qwen-edit-2511/fill-lightning-8", "krea2-turbo/fill"]
# argv[3]: Krea 2 denoise values to try instead of the 2511 comparison, e.g. "0.8,0.65".
DENOISE = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else []


def main():
    cli, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    setp = ["--cmd", "prefs.set", "--params", json.dumps({"values": {"integrations.engine": "comfy"}})]
    rows = []
    for pic, fx, fy, fw, fh, prompt, seed in CASES:
        src = os.path.join(STOCK, pic)
        im = Image.open(src).convert("RGB")
        W, H = im.size
        x, y, w, h = int(fx * W), int(fy * H), int(fw * W), int(fh * H)
        made = []
        runs = [("krea2-turbo/fill", d) for d in DENOISE] if DENOISE else [(t, None) for t in TEMPLATES]
        for t, d in runs:
            dest = os.path.join(out, f"{pic[:-4]}-{t.replace('/', '_')}{'' if d is None else f'-{d}'}.png")
            p = {"prompt": prompt, "seed": seed, "template": t, "variations": 1, "enhance": False}
            if d is not None:
                p["denoise"] = d
            sel = ["--cmd", "select.rect", "--params", json.dumps({"x": x, "y": y, "width": w, "height": h})]
            t0 = time.time()
            r = subprocess.run([cli, "run", src] + setp + sel + ["--cmd", "generate.fill", "--params", json.dumps(p), "--out", dest], capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok = r.returncode == 0 and os.path.exists(dest)
            print(f"{pic}\t{t}\t{time.time() - t0:.1f}s\t{'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-300:]}", flush=True)
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
        ImageDraw.Draw(sheet).text((4, r * cell + 4), CASES[r][0] + "  original | 2511 | Krea 2", fill=(255, 255, 0))
    sheet.save(os.path.join(out, "sheet.png" if not DENOISE else "sheet-denoise.png"))


if __name__ == "__main__":
    main()
