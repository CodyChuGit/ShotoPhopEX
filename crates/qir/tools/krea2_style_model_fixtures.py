"""Installed-model CPU oracle: stock EditPlus VAE references, LoRA, DiT and Euler.
Run after krea2_style_fixtures.py and krea2_style_encoder_fixtures.py.
Streams one DiT block at a time; never touches CUDA. Complete tensors stay generated locally.
"""
import os
for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[key] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, gc, ctypes, time
from pathlib import Path
COMFY = Path(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu", "--fp32-vae", "--use-pytorch-cross-attention"]
import torch
import comfy.ops, comfy.sd, comfy.utils
from comfy.latent_formats import Wan21
from comfy.ldm.krea2.model import SingleStreamDiT, SingleStreamBlock
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.model_sampling import ModelSamplingFlux
from comfy.samplers import simple_scheduler
from safetensors import safe_open
from safetensors.torch import load_file, save_file

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
OUT = Path(__file__).resolve().parents[1] / "tests/generated"

def memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
            ["peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]
    c = Counters(); c.cb = ctypes.sizeof(c)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(c), c.cb)
    assert c.peak_working < 24*2**30, c.peak_working
    return c.peak_working / 2**30

with torch.inference_mode():
    vae_path = COMFY / "models/vae/qwen_image_vae.safetensors"
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_path), safe_load=True), device=torch.device("cpu"), dtype=torch.float32)
    pixels = load_file(OUT / "krea2_style_preprocess.safetensors")
    result = {}
    for i in range(2):
        rgb = (pixels[f"vae_input_{i}"] + 1) / 2
        latent = vae.encode(rgb.movedim(1,-1))
        result[f"ref{i}"] = Wan21().process_in(latent).squeeze(2).contiguous()
        print("reference", i, "peak GiB", memory(), flush=True)
    del vae, pixels
    gc.collect()
    save_file(result, str(OUT / "krea2_style_references.safetensors"))
    result["context"] = load_file(OUT / "krea2_style_encoder.safetensors")["conditioning"]
    path = COMFY / "models/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
    lora_path = COMFY / "models/loras/krea2_style_reference.safetensors"
    with safe_open(path, framework="pt", device="cpu") as checkpoint, safe_open(lora_path, framework="pt", device="cpu") as lora:
        def load(module, prefix=""):
            for name, _ in list(module.named_parameters()):
                key = prefix + name
                value = checkpoint.get_tensor(key)
                if value.dtype == torch.float8_e4m3fn:
                    scale = checkpoint.get_tensor(key.removesuffix("weight") + "weight_scale")
                    value = (value.to(torch.bfloat16) * scale.to(torch.bfloat16)).float()
                else: value = value.float()
                stem = "diffusion_model." + key.removesuffix(".weight")
                akey, bkey = stem + ".lora_A.weight", stem + ".lora_B.weight"
                if akey in lora.keys():
                    a, b = lora.get_tensor(akey).float(), lora.get_tensor(bkey).float()
                    adapter = LoRAAdapter(set(), (b,a,None,None,None,None))
                    value = adapter.calculate_weight(value, key, 1.0, 1.0, None, lambda x:x)
                parent, _, leaf = name.rpartition(".")
                setattr(module.get_submodule(parent), leaf, torch.nn.Parameter(value, requires_grad=False))
            return module
        class StreamedBlock(torch.nn.Module):
            def __init__(self, index): super().__init__(); self.index=index
            def forward(self, *args, **kwargs):
                block = SingleStreamBlock(6144,48,4,kvheads=12, device="meta",dtype=torch.float32,operations=comfy.ops.disable_weight_init)
                load(block, f"blocks.{self.index}.")
                output = block(*args, **kwargs)
                print("block", self.index, "peak GiB", memory(), flush=True)
                return output
        model = load(SingleStreamDiT(layers=0,device="meta",dtype=torch.float32,operations=comfy.ops.disable_weight_init))
        model.blocks = torch.nn.ModuleList([StreamedBlock(i) for i in range(28)])
        refs = [result["ref0"], result["ref1"]]
        context = result["context"].flatten(2)
        torch.manual_seed(123)
        x = torch.randn(1,16,32,32)
        result["noise"] = x.clone()
        result["forward"] = model(x, torch.tensor([.625]), context, ref_latents=refs,ref_latents_method="index_timestep_zero")
        class Config: sampling_settings = {}
        sampling = ModelSamplingFlux(Config()); sampling.set_parameters(shift=.5)
        sigmas = simple_scheduler(sampling,2)
        result["sigmas"] = sigmas
        for i in range(2):
            v = model(x,sigmas[i:i+1],context,ref_latents=refs,ref_latents_method="index_timestep_zero")
            x = x + (sigmas[i+1]-sigmas[i])*v
            result[f"latent_{i+1}"] = x.clone()
        save_file({k:v.contiguous() for k,v in result.items()},str(OUT / "krea2_style_model.safetensors"))
        print("saved full installed model oracle; peak GiB",memory(),flush=True)
