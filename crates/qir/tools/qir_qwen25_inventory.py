"""Inspect Layered headers without loading checkpoint payloads."""
import collections
import json
import pathlib
import struct

ROOT = pathlib.Path(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models")

def header(path):
    with open(path, "rb") as f:
        return json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))

if __name__ == "__main__":
    for name in ["diffusion_models/qwen_image_layered_fp8mixed.safetensors",
                 "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                 "vae/qwen_image_layered_vae.safetensors"]:
        h = header(ROOT / name)
        tensors = {k: v for k, v in h.items() if k != "__metadata__"}
        print(name, len(tensors), collections.Counter(v["dtype"] for v in tensors.values()))
        print([(k, v) for k, v in tensors.items() if any(s in k for s in
              ["scaled_fp8", "layers.0.", "img_in.", "norm.weight", "timestep"])][:32])
        print([(k, v) for k, v in tensors.items() if "addition" in k or ("transformer_blocks.0." in k and "mod" in k)])
