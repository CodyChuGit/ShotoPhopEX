"""CPU ComfyUI oracle for stock Krea style semantics and complete preprocessing tensors.
Run portable python.exe -I crates/qir/tools/krea2_style_fixtures.py.
Small synthetic-model fixtures are intended for version control; large preprocessing tensors are regenerated.
This is an architectural regression gate, not an installed-model accuracy claim.
"""
import os
for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
COMFY = Path(r"C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu", "--use-pytorch-cross-attention"]
import torch
import comfy.ops
import comfy.utils
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
from safetensors.torch import save_file
import math, json

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
OUT = Path(__file__).resolve().parents[1] / "tests/fixtures"
LARGE = Path(__file__).resolve().parents[1] / "tests/generated"
LARGE.mkdir(exist_ok=True)

class CaptureClip:
    def tokenize(self, prompt, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        return []
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 1), {}]]

class CaptureVae:
    def __init__(self): self.inputs = []
    def encode(self, x):
        self.inputs.append(x)
        return torch.zeros(1, 16, x.shape[1] // 8, x.shape[2] // 8)

def picture(w, h):
    # Integer arithmetic gives identical pixels in Rust, including a nonconstant alpha.
    i = torch.arange(w * h * 4, dtype=torch.int64)
    return ((i * 37 + i // 7) % 256).to(torch.uint8).reshape(h, w, 4)

with torch.inference_mode():
    captured = {}
    clip, vae = CaptureClip(), CaptureVae()
    pictures = [picture(256, 256), picture(320, 240)]
    from PIL import Image
    for i,pixels in enumerate(pictures): Image.fromarray(pixels.numpy(), "RGBA").save(LARGE / f"krea2_style_ref{i+1}.png")
    TextEncodeQwenImageEditPlus.execute(clip, "a ceramic bird", vae,
        *[p[..., :3].float().unsqueeze(0) / 255 for p in pictures])
    for i, (vl, rgb) in enumerate(zip(clip.kwargs["images"], vae.inputs)):
        captured[f"vae_input_{i}"] = (rgb.movedim(-1, 1) * 2 - 1).contiguous()
        patches, grid = process_qwen2vl_images(vl, patch_size=16, image_mean=[.5]*3, image_std=[.5]*3)
        captured[f"patches_{i}"] = patches.contiguous()
        captured[f"grid_{i}"] = grid.contiguous()
    save_file(captured, str(LARGE / "krea2_style_preprocess.safetensors"))
    (OUT / "krea2_style_template.json").write_text(json.dumps(clip.kwargs["llama_template"].format(clip.prompt)))

    torch.manual_seed(429)
    model = SingleStreamDiT(features=64, tdim=32, txtdim=32, heads=2, kvheads=1,
        multiplier=4, layers=2, txtlayers=3, txtheads=1, txtkvheads=1,
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    for name, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * (.03 if name.endswith(("scale", "lin")) else .07))
    save_file(model.state_dict(), str(OUT / "krea2_style_weights.safetensors"))
    fixture = {}
    fixture["x"] = torch.randn(1, 16, 32, 32)
    fixture["context"] = torch.randn(1, 7, 3, 32) * .2
    fixture["ref0"] = torch.randn(1, 16, 8, 8) * .2
    fixture["ref1"] = torch.randn(1, 16, 7, 10) * .2
    context = fixture["context"].flatten(2)
    t = torch.tensor([.625])
    fixture["image"] = model(fixture["x"], t, context)
    # Legacy krea2edit wiring with stock ComfyUI modules: [text, reference, target],
    # aligned positions, shared timestep, target-to-reference attention boost four.
    from comfy.ldm.flux.layers import timestep_embedding
    target, target_ids, hh, ww = model.process_img(fixture["x"])
    reference, ref_ids, rh, rw = model.process_img(fixture["ref0"], index=1)
    ref_ids[...,1] += max(hh-rh,0)/2
    ref_ids[...,2] += max(ww-rw,0)/2
    ctx = model.txtmlp(model.txtfusion(fixture["context"]))
    prefix = ctx.shape[1] + reference.shape[1]
    combined = torch.cat((ctx,model.first(reference),model.first(target)),dim=1)
    ids = torch.cat((torch.zeros(1,ctx.shape[1],3),ref_ids,target_ids),dim=1)
    rope = model.pe_embedder(ids)
    temb = model.tmlp(timestep_embedding(t,32).unsqueeze(1))
    tvec = model.tproj(temb)
    mask = torch.zeros(1,1,combined.shape[1],combined.shape[1])
    mask[:,:,prefix:,ctx.shape[1]:prefix] = math.log(4)
    for block in model.blocks: combined = block(combined,tvec,rope,mask)
    final = model.last(combined[:,prefix:],temb)
    fixture["edit"] = final.reshape(1,hh,ww,16,2,2).permute(0,3,1,4,2,5).reshape(1,16,hh*2,ww*2)
    # Use ComfyUI's LoRA weight adapter to merge every block projection.
    from comfy.weight_adapter.lora import LoRAAdapter
    lora = {}
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and name.endswith("weight") and p.ndim == 2:
            stem = name.removesuffix(".weight")
            a = torch.randn(2, p.shape[1]) * .025
            b = torch.randn(p.shape[0], 2) * .025
            lora[stem + ".lora_down.weight"] = a
            lora[stem + ".lora_up.weight"] = b
            # Equivalent to LoraLoaderModelOnly strength 1 with absent alpha.
            adapter = LoRAAdapter(loaded_keys=set(), weights=(b, a, None, None, None, None))
            p.copy_(adapter.calculate_weight(p, stem, strength=1.0, strength_model=1.0, offset=None, function=lambda x:x))
    save_file(lora, str(OUT / "krea2_style_lora.safetensors"))
    refs = [fixture["ref0"], fixture["ref1"]]
    for count in (1, 2):
        fixture[f"style_{count}"] = model(fixture["x"], t, context,
            ref_latents=refs[:count], ref_latents_method="index_timestep_zero")
    torch.manual_seed(123)
    noise = torch.randn_like(fixture["x"])
    fixture["noise"] = noise.clone()
    from comfy.model_sampling import ModelSamplingFlux
    from comfy.samplers import simple_scheduler
    class Config: sampling_settings = {}
    sampling = ModelSamplingFlux(Config())
    sampling.set_parameters(shift=.5)
    sigmas = simple_scheduler(sampling, 2)
    fixture["sigmas"] = sigmas
    for i in range(2):
        velocity = model(noise, sigmas[i:i+1], context, ref_latents=refs,
            ref_latents_method="index_timestep_zero")
        noise = noise + (sigmas[i+1] - sigmas[i]) * velocity
    fixture["latent_2"] = noise
    save_file({k:v.contiguous() for k,v in fixture.items()}, str(OUT / "krea2_style.safetensors"))
    print("Wrote CPU architecture and preprocessing oracles", flush=True)
