# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""Overnight 2026-10-27 Q4: LoRA parity, the native engine against ComfyUI, through the app's CLI.

Each case runs with and without a LoRA on one route (argv: native|comfy, the CLI, the output
folder); `compare` (argv: compare, the output folder) prints the PSNR between the two routes per
case and setting. The native route attaches to a running worker (PHOTOCRAFT_NATIVE_ENGINE, from
native-worker.ps1); the ComfyUI route applies the LoRA as a LoraLoaderModelOnly after the
template's model loader (`comfy::with_loras`). Run with ComfyUI's Python (numpy, Pillow), -I.
"""

import json
import math
import os
import subprocess
import sys
import time

STOCK = r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
RUNTIME = ";".join([
    "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib",
    "C:/Users/5090/scoop/apps/cuda/current/bin/x64",
    "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib",
])
MODELS = "C:/Users/5090/Documents/_eval/native-models"
RESTORE = ("Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, "
           "keeping the same composition, people and colors.")
CASES = [
    ("winter", "edit", "st-glasses.jpg", "make it winter", 11),
    ("red-cup", "edit", "st-cup.jpg", "turn the cup red", 12),
    ("restore", "edit", "st-hat.jpg", RESTORE, 13),
    ("portrait", "image", None, "a portrait photograph of a woman in a red coat by a window", 14),
]
LORAS = {
    "edit": [None, ("qwen-image-2.1-consistency.safetensors", 1.0), ("qwen-image-2.1-restore.safetensors", 1.0)],
    "image": [None, ("bfs_head_v1.1_qwen_2.1.safetensors", 1.0)],
}


def tag(lora):
    return "none" if lora is None else lora[0].split(".safetensors")[0]


def run(route, cli, out):
    prefs = {
        "integrations.engine": "native" if route == "native" else "comfy",
        "integrations.allowResearchModels": True,
        "integrations.runtimePath": RUNTIME if route == "native" else "",
        "integrations.modelsPath": MODELS if route == "native" else "",
    }
    setp = ["--cmd", "prefs.set", "--params", json.dumps({"values": prefs})]
    for cid, kind, image, prompt, seed in CASES:
        for lora in LORAS[kind]:
            p = {"prompt": prompt, "seed": seed, "enhance": False}
            if lora is not None:
                p["loras"] = [{"file": lora[0], "strength": lora[1]}]
            dest = os.path.join(out, f"{cid}-{tag(lora)}-{route}.png")
            if kind == "edit":
                p["template"] = "qwen-2.1/edit"
                p["wholePicture"] = True
                args = [cli, "run", os.path.join(STOCK, image)] + setp + ["--cmd", "generate.edit", "--params", json.dumps(p)]
            else:
                p.update({"template": "qwen-2.1/image", "width": 1024, "height": 1024, "target": "layer"})
                args = [cli, "run", "--new", json.dumps({"width": 1024, "height": 1024})] + setp + ["--cmd", "generate.image", "--params", json.dumps(p)]
            args += ["--out", dest]
            t0 = time.time()
            r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok = r.returncode == 0 and os.path.exists(dest)
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] if not ok else []
            print(f"{route}\t{cid}\t{tag(lora)}\t{time.time() - t0:.1f}s\t{'ok' if ok else 'FAILED ' + ' '.join(tail)[:300]}", flush=True)


def psnr(a, b):
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        return float("nan")
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def compare(out):
    print("case\tsetting\tnative vs comfy dB\tlora vs none (native) dB\tlora vs none (comfy) dB")
    for cid, kind, _, _, _ in CASES:
        for lora in LORAS[kind]:
            f = lambda route, t=tag(lora): os.path.join(out, f"{cid}-{t}-{route}.png")
            g = lambda route: os.path.join(out, f"{cid}-none-{route}.png")
            if not (os.path.exists(f("native")) and os.path.exists(f("comfy"))):
                print(f"{cid}\t{tag(lora)}\tmissing")
                continue
            effect = "" if lora is None else f"{psnr(f('native'), g('native')):.1f}\t{psnr(f('comfy'), g('comfy')):.1f}"
            print(f"{cid}\t{tag(lora)}\t{psnr(f('native'), f('comfy')):.1f}\t{effect}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        os.makedirs(sys.argv[3], exist_ok=True)
        run(sys.argv[1], sys.argv[2], sys.argv[3])
