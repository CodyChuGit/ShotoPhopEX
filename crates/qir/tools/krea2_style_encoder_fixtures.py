"""Installed Qwen3-VL CPU oracle; stage vision and language weights separately.
Run portable python.exe -I this script after krea2_style_fixtures.py.
Uses ComfyUI's Krea2TEModel and stock EditPlus node, not a recreated chat encoder.
"""
import os
for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, gc, ctypes, time
from pathlib import Path
COMFY = Path(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu", "--use-pytorch-cross-attention"]
import torch
import comfy.ops
from comfy.text_encoders.krea2 import Krea2TEModel, Krea2Tokenizer
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
OUT = Path(__file__).resolve().parents[1] / "tests/generated"
OUT.mkdir(exist_ok=True)

def peak():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
            ["peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]
    c = Counters(); c.cb = ctypes.sizeof(c)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(c), c.cb)
    assert c.peak_working < 24 * 2**30, c.peak_working
    return c.peak_working / 2**30

def load(module, checkpoint, prefix):
    for name, _ in list(module.named_parameters()):
        key = prefix + name
        value = checkpoint.get_tensor(key)
        if value.dtype == torch.float8_e4m3fn:
            scale = checkpoint.get_tensor(key.removesuffix("weight") + "weight_scale")
            value = (value.to(torch.bfloat16) * scale.to(torch.bfloat16)).float()
        elif name != "pos_embed.weight":
            value = value.float()
        parent, _, leaf = name.rpartition(".")
        setattr(module.get_submodule(parent), leaf, torch.nn.Parameter(value, requires_grad=False))
    print("loaded", prefix, "peak GiB", peak(), flush=True)

class Clip:
    def __init__(self, model):
        self.model, self.tok = model, Krea2Tokenizer()
    def tokenize(self, *args, **kwargs): return self.tok.tokenize_with_weights(*args, **kwargs)
    def encode_from_tokens_scheduled(self, tokens):
        out = self.model.encode_token_weights(tokens)
        return [[out[0], {}]]

with torch.inference_mode(), safe_open(COMFY / "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors", framework="pt", device="cpu") as f:
    started = time.perf_counter()
    model = Krea2TEModel(device="meta", dtype=torch.float32, model_options={"custom_operations":comfy.ops.manual_cast})
    clipmodel = model.qwen3vl_4b
    clipmodel.execution_device = torch.device("cpu")
    transformer = clipmodel.transformer
    load(transformer.visual, f, "model.visual.")
    # Cache only small vision features; release the tower before loading the LM.
    pictures = []
    for w,h in [(256,256), (320,240)]:
        i = torch.arange(w*h*4, dtype=torch.int64)
        p = ((i*37+i//7)%256).to(torch.uint8).reshape(h,w,4)
        pictures.append(p[...,:3].float().unsqueeze(0)/255)
    class CacheClip:
        def tokenize(self, prompt, **kw): self.kw=kw; return []
        def encode_from_tokens_scheduled(self, tokens): return [[torch.zeros(1,1,1),{}]]
    capture = CacheClip()
    TextEncodeQwenImageEditPlus.execute(capture, "a ceramic bird", image1=pictures[0], image2=pictures[1])
    features = [transformer.preprocess_embed({"type":"image", "data":v}, torch.device("cpu")) for v in capture.kw["images"]]
    dump = {}
    for i, (merged, extra) in enumerate(features):
        dump[f"vision_{i}"] = merged.contiguous()
        for j,d in enumerate(extra["deepstack"]): dump[f"deepstack_{i}_{j}"] = d.contiguous()
    transformer.visual.to_empty(device="meta")
    gc.collect()
    load(transformer.model, f, "model.")
    # SDClipModel invokes this once per placeholder in prompt order.
    cursor = [0]
    def cached(embed, device):
        result = features[cursor[0]]; cursor[0] += 1
        return result
    transformer.preprocess_embed = cached
    clip = Clip(model)
    result = TextEncodeQwenImageEditPlus.execute(clip, "a ceramic bird", image1=pictures[0], image2=pictures[1])
    # NodeOutput's first output is CONDITIONING.
    dump["conditioning"] = result[0][0][0].reshape(1,-1,12,2560).contiguous()
    save_file(dump, str(OUT / "krea2_style_encoder.safetensors"))
    print("saved full encoder tensors; seconds", time.perf_counter()-started, "peak GiB", peak(), flush=True)
