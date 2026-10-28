"""Materialize Comfy-style patched INT8 weights for a native parity experiment."""
import argparse
import json
from pathlib import Path
import zlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--lora", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--strength", type=float, default=0.5)
args = parser.parse_args()
with safe_open(args.base, framework="pt", device="cpu") as source:
    weights = {name: source.get_tensor(name) for name in source.keys()}
with safe_open(args.lora, framework="pt", device="cpu") as source:
    adapters = {name: source.get_tensor(name) for name in source.keys()}
count = 0
for name in sorted(adapters):
    if not name.endswith(".lora_A.weight"):
        continue
    module = name.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
    qkey, skey = module + ".weight_i8", module + ".col_scale"
    q = weights[qkey].cuda().view(torch.int8)
    n, k = q.shape
    # Native's rotation basis differs only by reversing each group of quantized columns.
    q = q.view(n, k // 256, 256).flip(-1).reshape(n, k)
    params = TensorWiseINT8Layout.Params(scale=weights[skey].cuda().reshape(n, 1), orig_dtype=torch.bfloat16,
                                        orig_shape=(n, k), is_weight=True, convrot=True, convrot_groupsize=256)
    quantized = QuantizedTensor(q, "TensorWiseINT8Layout", params)
    a = adapters[name].cuda()
    b = adapters[name.replace(".lora_A.", ".lora_B.")].cuda()
    weight = quantized.dequantize()
    weight += args.strength * torch.mm(b, a)
    seed = zlib.crc32(("diffusion_model." + module).encode())
    patched = quantized.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed)
    weights[qkey] = patched._qdata.view(n, k // 256, 256).flip(-1).reshape(n, k).view(torch.uint8).cpu()
    weights[skey] = patched._params.scale.reshape(n).cpu()
    count += 1
    print(json.dumps({"module": module, "patched": count}), flush=True)
    del q, quantized, weight, patched, a, b
save_file(weights, args.out)
print(json.dumps({"modules": count, "output": str(args.out), "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
