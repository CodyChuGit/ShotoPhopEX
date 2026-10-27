"""CPU-only ComfyUI reference; stream one language block to bound RAM.

Run with the portable python.exe -I. No vision weights or CUDA allocations.
Reference uses ComfyUI blocks in f32, with the checkpoint's FP8 weights
dequantized in bf16 first (the same rounding as qir::fp8).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import pathlib
import json
import time
import gc
import ctypes

COMFY = pathlib.Path(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI")
TOKENIZER_ONLY = "--tokenizer-only" in sys.argv
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy.text_encoders import llama, qwen_image
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.ops

torch.set_num_threads(8)
OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
PATH = COMFY / "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

def peak_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
            ["peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(c), c.cb)
    assert ok, ctypes.get_last_error()
    return c.peak_working

def tensor(f, key):
    w = f.get_tensor(key)
    if w.dtype == torch.float8_e4m3fn:
        scale = f.get_tensor(key.removesuffix("weight") + "scale_weight")
        w = w.to(torch.bfloat16) * scale.to(torch.bfloat16)
    return w.float()

def cut(ids):
    starts = [i for i, t in enumerate(ids) if t == 151644][:2]
    end = starts[-1] if starts else -1
    if len(ids) > end + 3 and ids[end + 1:end + 3] == [872, 198]:
        end += 3
    return end if end >= 0 else len(ids) - 1

def record(out, meta, key, x, full=False):
    x = x.detach().float().contiguous()
    flat = x.flatten()
    idx = torch.arange(4096, dtype=torch.int64) * (flat.numel() - 1) // 4095
    out[key] = x if full and x.numel() * 4 < 1_500_000 else flat[idx]
    meta[key] = dict(shape=list(x.shape), full=list(out[key].shape) == list(x.shape),
                     mean=x.mean().item(), std=x.std(correction=0).item(), min=x.min().item(), max=x.max().item())

@torch.inference_mode()
def main():
    start = time.perf_counter()
    tokenizer = qwen_image.QwenImageTokenizer()
    cases = [("positive", "A red cat."), ("negative", "")]
    ids = [torch.tensor([[t[0] for t in tokenizer.tokenize_with_weights(p)["qwen25_7b"][0]]]) for _, p in cases]
    out, meta = {}, {"precision": "CPU f32 ComfyUI blocks; FP8 -> bf16 * bf16 scale -> f32", "prompts": {}}
    # Transformers' fast equivalent is a portable tokenizer.json for Rust.
    from transformers import Qwen2TokenizerFast
    fast = Qwen2TokenizerFast.from_pretrained(COMFY / "comfy/text_encoders/qwen25_tokenizer", local_files_only=True)
    shared = pathlib.Path(r"C:\Users\5090\Documents\_eval\weights\krea2-turbo\tokenizer\tokenizer.json")
    from tokenizers import Tokenizer
    shared_tok = Tokenizer.from_file(str(shared)) if shared.is_file() else None
    if shared_tok is not None:
        a = json.loads(fast.backend_tokenizer.to_str())
        b = json.loads(shared_tok.to_str())
        meta["shared_tokenizer"] = dict(path=str(shared), same_vocab=a["model"]["vocab"] == b["model"]["vocab"],
            same_merges=a["model"]["merges"] == b["model"]["merges"], same_pretokenizer=a["pre_tokenizer"] == b["pre_tokenizer"])
        meta["shared_tokenizer"]["number_example"] = dict(text="123456", qwen25=fast.encode("123456", add_special_tokens=False),
            qwen3=shared_tok.encode("123456", add_special_tokens=False).ids)
        probes = ["", "A red cat.", "123456", "cafe\u0301", "中文的图像", "  spaces\n\n", "# ## ###", "<|image_pad|>", "<tool_call>", "<|fim_prefix|>", "<|video_pad|>"]
        meta["tokenizer_probes"] = [dict(text=t, ids=fast.encode(t, add_special_tokens=False),
            qwen3_ids=shared_tok.encode(t, add_special_tokens=False).ids) for t in probes]
        meta["shared_tokenizer"]["same_normalizer"] = a["normalizer"] == b["normalizer"]
        meta["shared_tokenizer"]["extra_qwen3_tokens"] = sorted(set(t["content"] for t in b["added_tokens"]) - set(t["content"] for t in a["added_tokens"]))
    for (name, prompt), tokens in zip(cases, ids):
        expected = tokens[0].tolist()
        template = tokenizer.llama_template.format(prompt)
        assert fast.encode(template, add_special_tokens=False) == expected
        if shared_tok is not None:
            assert shared_tok.encode(template, add_special_tokens=False).ids == expected
        meta["prompts"][name] = dict(text=prompt, ids=expected, prefix_cut=cut(expected))
    if TOKENIZER_ONLY:
        existing = json.loads((OUT / "qwen25_text.json").read_text())
        existing.update({k: meta[k] for k in ["shared_tokenizer", "tokenizer_probes"]})
        (OUT / "qwen25_text.json").write_text(json.dumps(existing, indent=2) + "\n")
        print(json.dumps(meta["shared_tokenizer"], indent=2))
        return
    with safe_open(PATH, framework="pt", device="cpu") as f:
        # Gather embedding rows before casting: never retain the entire f32 embedding.
        emb = f.get_tensor("model.embed_tokens.weight")
        states = [emb[t].float().clone() for t in ids]
        del emb
        cfg = llama.Qwen25_7BVLI_Config()
        freqs = [llama.precompute_freqs_cis(128, torch.arange(t.shape[1]).unsqueeze(0), 1e6) for t in ids]
        masks = [torch.full((t.shape[1], t.shape[1]), torch.finfo(torch.float32).min / 4).triu_(1) for t in ids]
        for li in range(28):
            prefix = f"model.layers.{li}."
            with torch.device("meta"):
                layer = llama.TransformerBlock(cfg, li, dtype=torch.float32, ops=comfy.ops.manual_cast)
            weights = {k[len(prefix):]: tensor(f, k) for k in f.keys()
                       if k.startswith(prefix) and (k.endswith(".weight") or k.endswith(".bias"))}
            layer.load_state_dict(weights, assign=True, strict=True)
            del weights
            for ci, ((name, _), x) in enumerate(zip(cases, states)):
                # TransformerBlock aliases its input as an output buffer; clone taps below.
                states[ci] = layer(x=x, attention_mask=masks[ci], freqs_cis=freqs[ci],
                                   optimized_attention=optimized_attention_for_device(torch.device("cpu"), mask=True, small_input=True))[0]
                if li in [0, 13, 27]:
                    record(out, meta, f"{name}.layer{li}", states[ci])
            del layer
            gc.collect()
            print(f"layer {li}: {time.perf_counter()-start:.2f}s peak {peak_bytes()/2**30:.2f} GiB", flush=True)
            assert peak_bytes() < 25 * 2**30
        norm = llama.RMSNorm(3584, eps=1e-6, dtype=torch.float32)
        norm.load_state_dict({"weight": tensor(f, "model.norm.weight")})
        for (name, _), x in zip(cases, states):
            record(out, meta, f"{name}.conditioning", norm(x)[:, meta["prompts"][name]["prefix_cut"]:], full=True)
    meta["seconds"] = time.perf_counter() - start
    meta["peak_working_bytes"] = peak_bytes()
    save_file(out, OUT / "qwen25_text.safetensors")
    (OUT / "qwen25_text.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ["prompts"]}), flush=True)

if __name__ == "__main__":
    main()
