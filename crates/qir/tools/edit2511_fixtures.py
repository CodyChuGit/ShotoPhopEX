"""CPU ComfyUI oracle for 2511 preparation and header-only adapter maps.

Run portable python_embeded/python.exe -I crates/qir/tools/edit2511_fixtures.py.
Large complete tensors are regenerated in target-edit2511/fixtures. Checked-in
small fixtures contain deterministic samples plus complete crop and mask tensors.
"""
import os
for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import pathlib
import json
import struct
from types import SimpleNamespace
COMFY = pathlib.Path(r'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI')
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import torch
from safetensors.torch import save_file
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
from comfy_extras.nodes_flux import FluxKontextImageScale
from comfy.sd import VAE
import comfy.utils
import comfy.lora

torch.set_num_threads(8)
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'target-edit2511/fixtures'
SMALL = ROOT / 'crates/qir/tests/fixtures'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = [
    'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
    'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors',
    'Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors',
    'qwen-image-edit-2511-multiple-angles-lora.safetensors',
]

class Clip:
    def tokenize(self, prompt, **kwargs):
        self.images = kwargs['images']
        return []
    def encode_from_tokens_scheduled(self, tokens):
        return []
class CaptureVae:
    def encode(self, pixels):
        self.pixels = pixels
        return torch.zeros(1)
class CropVae:
    crop_input = True
    output_channels = 3
    def spacial_compression_encode(self):
        return 8

def header(path):
    with path.open('rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        data = json.loads(f.read(n))
    data.pop('__metadata__', None)
    return n + 8, data

class HeaderQwen(comfy.model_base.QwenImage):
    """Expose checkpoint shapes to the actual QwenImage key mapper, with no model allocation."""
    def __init__(self, hdr):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, 'model_config', SimpleNamespace(unet_config={}))
        object.__setattr__(self, 'header_state', {
            'diffusion_model.' + k: torch.empty(v['shape'], device='meta') for k,v in hdr.items()
        })
    def state_dict(self):
        return self.header_state

def lora_maps():
    maps = {}
    for name in NAMES:
        path = COMFY / 'models/loras' / name
        if not path.exists():
            maps[name] = None
            print(name, 'absent')
            continue
        start, hdr = header(path)
        # Meta tensors supply exact shapes to ComfyUI's actual loader, without
        # reading hundreds of MB of adapter payload. Only alpha scalars are read.
        raw = {}
        with path.open('rb') as f:
            for k, v in hdr.items():
                if k.endswith('.alpha'):
                    f.seek(start + v['data_offsets'][0])
                    b = f.read(v['data_offsets'][1] - v['data_offsets'][0])
                    raw[k] = torch.frombuffer(bytearray(b), dtype={'BF16':torch.bfloat16,'F32':torch.float32}[v['dtype']]).reshape(v['shape'])
                else:
                    raw[k] = torch.empty(v['shape'], dtype=torch.bfloat16, device='meta')
        # Both the model key map and adapter pairing come from ComfyUI itself.
        _, dit = header(COMFY / 'models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors')
        mapping = comfy.lora.model_lora_keys_unet(HeaderQwen(dit), {})
        patches = comfy.lora.load_lora(raw, mapping)
        result = {}
        for k, adapter in patches.items():
            up, down, alpha, *_ = adapter.weights
            rank = down.shape[0]
            module = k.removeprefix('diffusion_model.').removesuffix('.weight')
            result[module] = dict(rank=rank, alpha=alpha, scale=1. if alpha is None else alpha/rank)
        assert len(result) == (840 if name == NAMES[3] else 720), (name, len(result))
        maps[name] = result
        print(name, len(result), sorted(set((x['rank'],x['alpha'],x['scale']) for x in result.values())))
    (SMALL / 'edit2511_lora_maps.json').write_text(json.dumps(maps, indent=2) + '\n')

@torch.inference_mode()
def main():
    small = {}
    for case,h,w in [('portrait',83,51), ('landscape',49,87), ('odd',73,77), ('large_portrait',1307,811), ('large_landscape',791,1473)]:
        raw = ((torch.arange(h*w*3).reshape(h,w,3)*37+19)%256).to(torch.uint8)
        pixels = raw.float()/255.
        vae = CaptureVae()
        clip = Clip()
        TextEncodeQwenImageEditPlus.execute(clip, '', vae=vae, image1=pixels[None])
        reference = vae.pixels[0]
        kontext = FluxKontextImageScale.execute(pixels[None])[0][0]
        crop = VAE.vae_encode_crop_pixels(CropVae(), pixels[None])[0]
        red = pixels[...,0]
        mask4 = comfy.utils.reshape_mask(red, (2,16,9,13))
        mask5 = comfy.utils.reshape_mask(red, (2,16,1,9,13))
        stages = dict(vl=clip.images[0][0], reference=reference, kontext=kontext, crop=crop, mask4=mask4, mask5=mask5)
        full = dict(pixels=pixels, **stages)
        save_file({k:v.contiguous() for k,v in full.items()}, OUT / f'{case}.safetensors')
        if case.startswith('large_'):
            print(case, {k:list(v.shape) for k,v in stages.items()})
            continue
        small[f'{case}.pixels'] = pixels
        for key,value in stages.items():
            small[f'{case}.{key}.shape'] = torch.tensor(value.shape)
            # Keep complete crop/masks; regularly spaced samples for megapixel images.
            v = value.flatten()
            small[f'{case}.{key}'] = (v if key in ['crop','mask4','mask5'] else v[torch.arange(4096)*(v.numel()-1)//4095]).contiguous()
        print(case, {k:list(v.shape) for k,v in stages.items()})
    save_file(small, SMALL / 'edit2511_prep_small.safetensors')
    lora_maps()

if __name__ == '__main__':
    main()


