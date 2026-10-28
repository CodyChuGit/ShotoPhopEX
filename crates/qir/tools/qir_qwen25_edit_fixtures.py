"""Regenerate EditPlus CPU fixtures with portable python.exe -I.

Small sampled outputs and exact layout JSON are checked in. The full embeddings,
LM taps and conditioning are regenerated in qwen25_edit_full.safetensors.
Only installed ComfyUI modules implement the reference operations.
"""
import os
for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import pathlib
import json
import gc
import time
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qir_qwen25_fixtures as base
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy.text_encoders import llama, qwen_image, qwen_vl
from comfy import sd1_clip
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device

torch.set_num_threads(8)

class Embedding(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, ids, out_dtype=None):
        return torch.nn.functional.embedding(ids, self.weight).to(out_dtype)

class Splicer:
    special_tokens = {"pad": 151643}
    def __init__(self, embedding, pictures):
        self.transformer = self
        self.embedding = embedding
        self.pictures = iter(pictures)
    def get_input_embeddings(self):
        return self.embedding
    def preprocess_embed(self, embed, device):
        return next(self.pictures)

@torch.inference_mode()
def main():
    start = time.perf_counter()
    out, full, meta = {}, {}, {"prompts": {}, "precision": "ComfyUI CPU f32, FP8 dequantized through bf16"}
    meta["mrope_inv_freq_bits"] = (1.0 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))).view(torch.int32).tolist()
    probe = torch.randn(1, 8, 3584, generator=torch.Generator().manual_seed(2511))
    probe_norm = llama.RMSNorm(3584, eps=1e-6, dtype=torch.float32)
    probe_norm.load_state_dict({"weight": torch.linspace(0.5, 1.5, 3584)})
    out["rms.input"] = probe
    out["rms.weight"] = probe_norm.weight.detach().clone()
    out["rms.output"] = probe_norm(probe)
    with safe_open(base.PATH, framework="pt", device="cpu") as f:
        with torch.device("meta"):
            visual = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=3584,
                dtype=torch.float32, ops=comfy.ops.manual_cast)
        weights = {k.removeprefix("visual."): base.tensor(f, k) for k in f.keys()
                   if k.startswith("visual.") and k.endswith((".weight", ".bias"))}
        visual.load_state_dict(weights, assign=True, strict=True)
        del weights
        pictures = []
        # The VL sizes after EditPlus's 384^2 area resize, before rounding to 28.
        for i, (h, w) in enumerate([(384, 384), (272, 543)]):
            image = (torch.arange(h*w*3).reshape(1, h, w, 3) % 251).float() / 250
            pixels, grid = qwen_vl.process_qwen2vl_images(image)
            rows = visual(pixels, grid).clone()
            pictures.append((rows, grid))
            full[f"picture{i}"] = rows
            meta[f"picture{i}"] = {"grid": grid[0].tolist(), "input_hw": [h, w]}
            print(f"picture {i}: {list(rows.shape)}, peak {base.peak_bytes()/2**30:.2f} GiB", flush=True)
        del visual, pixels, image
        gc.collect()
        tokenizer = qwen_image.QwenImageTokenizer()
        template = tokenizer.llama_template_images.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        embedding = Embedding(f.get_tensor("model.embed_tokens.weight"))
        states, masks, freqs = [], [], []
        for name, count, prompt in [("one", 1, "Make the sky blue."), ("two", 2, "Use Picture 2's colors in Picture 1.")]:
            text = "".join(f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(count)) + prompt
            tokens = tokenizer.tokenize_with_weights(text, images=[None]*count, llama_template=template)["qwen25_7b"][0]
            ids = [151655 if isinstance(t[0], dict) else t[0] for t in tokens]
            splicer = Splicer(embedding, pictures[:count])
            x, mask, _, info = sd1_clip.SDClipModel.process_tokens(splicer, [[t[0] for t in tokens]], torch.device("cpu"))
            # Capture position_ids from the actual installed Qwen25 forward, no copied algorithm.
            model = llama.Qwen25_7BVLI.__new__(llama.Qwen25_7BVLI)
            torch.nn.Module.__init__(model)
            with patch.object(llama.BaseLlama, "forward", lambda self, *a, **kw: kw["position_ids"]):
                pos = model(None, attention_mask=mask, embeds=x, embeds_info=info)
            expanded = ids.copy()
            for e in info:
                expanded[e["index"]:e["index"]+1] = [151655]*e["size"]
            # Let the actual TE wrapper choose the cut and slice the mask.
            te = qwen_image.QwenImageTEModel.__new__(qwen_image.QwenImageTEModel)
            torch.nn.Module.__init__(te)
            with patch.object(sd1_clip.SD1ClipModel, "encode_token_weights",
                              lambda self, *a, **kw: (x, None, {"attention_mask": mask})):
                sliced, _, extra = te.encode_token_weights({"qwen25_7b": [tokens]})
            cut = x.shape[1] - sliced.shape[1]
            meta["prompts"][name] = dict(text=prompt, pictures=count, ids=ids, expanded_ids=expanded,
                spans=[[e["index"], e["size"]] for e in info], positions=pos.tolist(), prefix_cut=cut,
                mask=mask[0].tolist())
            full[f"{name}.mask"] = extra.get("attention_mask", torch.ones(sliced.shape[:2])).float().contiguous()
            full[f"{name}.embeddings"] = x.clone()
            base.record(out, meta, f"{name}.embeddings", x)
            states.append(x)
            s = x.shape[1]
            masks.append(torch.full((s,s), torch.finfo(torch.float32).min/4).triu_(1))
            freqs.append(llama.precompute_freqs_cis(128, pos, 1e6, rope_dims=[16,24,24]))
        del embedding, splicer
        gc.collect()
        cfg = llama.Qwen25_7BVLI_Config()
        for li in range(28):
            prefix = f"model.layers.{li}."
            with torch.device("meta"):
                layer = llama.TransformerBlock(cfg, li, dtype=torch.float32, ops=comfy.ops.manual_cast)
            weights = {k[len(prefix):]: base.tensor(f, k) for k in f.keys()
                       if k.startswith(prefix) and k.endswith((".weight", ".bias"))}
            layer.load_state_dict(weights, assign=True, strict=True)
            del weights
            for ci, name in enumerate(["one", "two"]):
                states[ci] = layer(x=states[ci], attention_mask=masks[ci], freqs_cis=freqs[ci],
                    optimized_attention=optimized_attention_for_device(torch.device("cpu"), mask=True, small_input=True))[0]
                if li in [0,13,27]:
                    key = f"{name}.layer{li}"
                    full[key] = states[ci].clone()
                    base.record(out, meta, key, states[ci])
            del layer
            gc.collect()
            print(f"layer {li}: {time.perf_counter()-start:.1f}s peak {base.peak_bytes()/2**30:.2f} GiB", flush=True)
            assert base.peak_bytes() < 24 * 2**30
        norm = llama.RMSNorm(3584, eps=1e-6, dtype=torch.float32)
        norm.load_state_dict({"weight": base.tensor(f, "model.norm.weight")})
        for name, x in zip(["one", "two"], states):
            key = f"{name}.conditioning"
            x = norm(x)[:, meta["prompts"][name]["prefix_cut"]:].contiguous()
            full[key] = x
            base.record(out, meta, key, x)
    meta["seconds"] = time.perf_counter()-start
    meta["peak_working_bytes"] = base.peak_bytes()
    save_file(full, base.OUT / "qwen25_edit_full.safetensors")
    save_file(out, base.OUT / "qwen25_edit.safetensors")
    (base.OUT / "qwen25_edit.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"done: {meta['seconds']:.1f}s, peak {base.peak_bytes()/2**30:.2f} GiB")

if __name__ == "__main__":
    main()
