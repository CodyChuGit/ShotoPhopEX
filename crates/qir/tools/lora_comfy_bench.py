"""Replay the native harness's six PNG tiles through installed ComfyUI."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--lora", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
base = "http://127.0.0.1:8188"

def request(path, data=None, content_type="application/json"):
    payload = json.dumps(data).encode() if isinstance(data, dict) else data
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

names = []
for i in range(6):
    filename = f"qir-lora-runtime-{i}.png"
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + (args.inputs / f"input-{i}.png").read_bytes()
            + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode())
    names.append(request("/upload/image", body, f"multipart/form-data; boundary={boundary}")["name"])

prompt = "Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, keeping the same composition, people and colors."
for i in range(12):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": names[i % 6]}},
        "5": {"class_type": "QwenImage21Cache", "inputs": {"model": ["13" if args.lora else "1", 0], "device": "auto", "dtype": "default"}},
        "6": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "negative_prompt": "", "resolution": 0, "images.image_1": ["4", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["6", 2], "seed": 42 + i, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "qir-lora-runtime"}},
    }
    if args.lora:
        graph["13"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "qwen-image-2.1-restore.safetensors", "strength_model": 0.5}}
    started = time.perf_counter()
    queued = request("/prompt", {"prompt": graph, "client_id": "qir-lora-bench"})
    prompt_id = queued["prompt_id"]
    while True:
        history = request("/history/" + prompt_id)
        if prompt_id in history:
            history = history[prompt_id]
            break
        time.sleep(0.1)
    if history["status"]["status_str"] != "success":
        raise RuntimeError(history)
    record = {"run": i, "seed": 42 + i, "seconds": time.perf_counter() - started, "history": history}
    (args.out / f"run-{i}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("run", "seed", "seconds")}), flush=True)
