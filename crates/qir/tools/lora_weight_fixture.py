"""Generate small native-weight oracles with the installed Comfy Kitchen backend."""
import argparse
import zlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--lora", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
result = {}
modules = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.out", "transformer_blocks.0.img_mlp.gate_up"]
with safe_open(args.base, framework="pt", device="cpu") as base, safe_open(args.lora, framework="pt", device="cpu") as adapter:
    for i, module in enumerate(modules):
        q = base.get_tensor(module + ".weight_i8")[:16].cuda()
        scales = base.get_tensor(module + ".col_scale")[:16].cuda()
        n, k = q.shape
        comfy_q = q.view(torch.int8).view(n, k // 256, 256).flip(-1).reshape(n, k)
        params = TensorWiseINT8Layout.Params(scale=scales.reshape(n, 1), orig_dtype=torch.bfloat16,
                                            orig_shape=(n, k), is_weight=True, convrot=True, convrot_groupsize=256)
        quantized = QuantizedTensor(comfy_q, "TensorWiseINT8Layout", params)
        original = quantized.dequantize()
        up = adapter.get_tensor("diffusion_model." + module + ".lora_B.weight")[:16].cuda()
        down = adapter.get_tensor("diffusion_model." + module + ".lora_A.weight").cuda()
        patched = original + 0.5 * torch.mm(up, down)
        seed = zlib.crc32(("diffusion_model." + module).encode())
        expected = quantized.requantize_from_float(patched, scale="recalculate", stochastic_rounding=seed)
        expected_q = expected._qdata.view(n, k // 256, 256).flip(-1).reshape(n, k).view(torch.uint8)
        tensors = {"base_q": q, "base_s": scales, "base_w": original, "patched": patched,
                   "q": expected_q, "s": expected._params.scale.reshape(n), "a": down, "b": up}
        result.update({f"{i}.{name}": tensor.cpu().contiguous() for name, tensor in tensors.items()})
        result[f"{i}.seed"] = torch.tensor([seed], dtype=torch.int64)
save_file(result, args.out)
