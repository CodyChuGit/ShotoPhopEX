"""Exercise installed ComfyUI's dynamic patch materialization on a real module."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

COMFY = Path("C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
sys.path.insert(0, str(COMFY))
import torch
from safetensors import safe_open
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
import comfy.ops
import comfy.utils
from comfy.model_patcher import LowVramPatch
from comfy.weight_adapter.lora import LoRAAdapter

module = "transformer_blocks.0.attn.to_q"
seed_key = "diffusion_model." + module
key = seed_key + ".weight"
with safe_open(COMFY / "models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", framework="pt", device="cuda") as f:
    q = f.get_tensor(module + ".weight")
    scale = f.get_tensor(module + ".weight_scale")
with safe_open(COMFY / "models/loras/qwen-image-2.1-restore.safetensors", framework="pt", device="cuda") as f:
    a = f.get_tensor("diffusion_model." + module + ".lora_A.weight")
    b = f.get_tensor("diffusion_model." + module + ".lora_B.weight")
params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape), is_weight=True, convrot=True, convrot_groupsize=256)
weight = QuantizedTensor(q, "TensorWiseINT8Layout", params)
unpatched = weight.dequantize()
reference = unpatched + 0.5 * torch.mm(b, a)
expected = weight.requantize_from_float(reference, scale="recalculate", stochastic_rounding=comfy.utils.string_to_seed(seed_key))
patch = LowVramPatch(key, {key: [(0.5, LoRAAdapter(set(), (b, a, None, None, None, None)), 1.0, None, None)]})
calls = 0

def apply(w):
    global calls
    calls += 1
    return patch(w)

# Supply the gathered parameters directly; residency and patching use the real implementation.
original_interpret = comfy.memory_management.interpret_gathered_like
comfy.memory_management.interpret_gathered_like = lambda *args: [weight, None]
s = SimpleNamespace(weight=weight, bias=None, seed_key=seed_key, weight_lowvram_function=apply, weight_function=[],
                    _prefetch={"resident": False, "signature": 1, "xfer_dest": None, "cast_geometry": None, "needs_cast": False})
try:
    first, _ = comfy.ops.resolve_cast_module_with_vbar(s, torch.bfloat16, torch.device("cuda"), torch.bfloat16, torch.bfloat16, True)
    first_bytes = first._qdata.clone()
    second, _ = comfy.ops.resolve_cast_module_with_vbar(s, torch.bfloat16, torch.device("cuda"), torch.bfloat16, torch.bfloat16, True)
finally:
    comfy.memory_management.interpret_gathered_like = original_interpret
x = torch.randn((3844, 4096), device="cuda", dtype=torch.bfloat16)
side = torch.nn.functional.linear(x, unpatched) + 0.5 * torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
merged = torch.nn.functional.linear(x, second.dequantize())
torch.cuda.synchronize()
print(json.dumps({
    "module": key,
    "patch_calls_for_two_resolves": calls,
    "resident_quantized": isinstance(second, QuantizedTensor),
    "bytes_match_explicit_bf16_patch_requant": torch.equal(expected._qdata, second._qdata),
    "scales_match": torch.equal(expected._params.scale, second._params.scale),
    "resident_bytes_unchanged": torch.equal(first_bytes, second._qdata),
    "side_vs_merged_mean_abs": (side.float() - merged.float()).abs().mean().item(),
    "seed": comfy.utils.string_to_seed(seed_key),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
}, indent=2))
