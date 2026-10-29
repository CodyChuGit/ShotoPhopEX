# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
"""The native worker's LoRA clear (2026-10-27): an edit with a LoRA, then the same without one, on a
running worker (PHOTOCRAFT_NATIVE_ENGINE). The second failed with CUDA_ERROR_INVALID_CONTEXT in
0.3 s. argv: the CLI, the output folder. Run with ComfyUI's Python, -I.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4_lora_parity import MODELS, RUNTIME, STOCK  # noqa: E402

cli, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
prefs = {"integrations.engine": "native", "integrations.allowResearchModels": True, "integrations.runtimePath": RUNTIME, "integrations.modelsPath": MODELS}
for i, loras in enumerate([[{"file": "qwen-image-2.1-consistency.safetensors", "strength": 1.0}], []]):
    p = {"prompt": "make it winter", "seed": 11, "enhance": False, "template": "qwen-2.1/edit", "wholePicture": True}
    if loras:
        p["loras"] = loras
    args = [cli, "run", os.path.join(STOCK, "st-glasses.jpg"), "--cmd", "prefs.set", "--params", json.dumps({"values": prefs}), "--cmd", "generate.edit", "--params", json.dumps(p), "--out", os.path.join(out, f"repro-{i}.png")]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(i, [l["file"] for l in loras], f"{time.time() - t0:.1f}s", "ok" if r.returncode == 0 else "FAILED", flush=True)
    if r.returncode != 0:
        with open(os.path.join(out, f"repro-{i}.err.txt"), "w", encoding="utf-8") as e:
            e.write((r.stderr or "") + (r.stdout or ""))
