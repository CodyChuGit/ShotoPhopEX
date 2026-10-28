"""ComfyUI CPU vision oracle; portable Python -I, one f32 block at a time.

Full portrait/landscape stage tensors go in target-qwen25-vision/fixtures.
The small preprocessing fixture is suitable for checking in (<2 MB).
"""
import os
for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import pathlib
import gc
import time
import json
import ctypes
COMFY = pathlib.Path(r'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI')
SMALL = '--small' in sys.argv
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import torch
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from comfy.text_encoders import qwen_vl
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.ops
import comfy.utils

torch.set_num_threads(8)
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'target-qwen25-vision/fixtures'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = COMFY / 'models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors'

def peak_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
            ['peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile']]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(c), c.cb)
    return c.peak_working

def tensor(f, key):
    w = f.get_tensor(key)
    if w.dtype == torch.float8_e4m3fn:
        w = w.bfloat16() * f.get_tensor(key.removesuffix('weight') + 'scale_weight').bfloat16()
    return w.float()

def resize(image):
    h,w = image.shape[:2]
    scale = (384**2/(h*w))**0.5
    return comfy.utils.common_upscale(image.permute(2,0,1)[None], round(w*scale), round(h*scale), 'area', 'disabled')[0].permute(1,2,0)

@torch.inference_mode()
def main():
    start = time.perf_counter()
    with torch.device('meta'):
        tower = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, dtype=torch.float32, ops=comfy.ops.manual_cast)
    if SMALL:
        # Asymmetric, non-multiple dimensions exercise bilinear coordinates and patch packing.
        pixels = ((torch.arange(57*85*3).reshape(57,85,3)*37+19)%256).float()/255
        patches,grid = qwen_vl.process_qwen2vl_images(pixels[None])
        index,cu = tower.get_window_index(grid)
        out = dict(pixels=pixels, patches=patches, grid=grid, window_index=index, cu=torch.unique_consecutive(torch.tensor(cu)), resized=resize(pixels))
        for h,w in [(18,26),(16,16)]:
            index,cu = tower.get_window_index(torch.tensor([[1,h,w]]))
            out[f'index_{h}_{w}'] = index
            out[f'cu_{h}_{w}'] = torch.unique_consecutive(torch.tensor(cu))
        save_file({k:v.contiguous() for k,v in out.items()}, ROOT / 'crates/qir/tests/fixtures/qwen25_vision_small.safetensors')
        return
    square = pathlib.Path(r'C:/Users/5090/ComfyUI/photocraft-tests/bench/swap/gen/gen-tshirt.png')
    stock = pathlib.Path(r'C:/Users/5090/ComfyUI/photocraft-tests/bench/stock')
    landscape = next(p for p in sorted(stock.glob('*.png')) if Image.open(p).width > Image.open(p).height)
    portrait = next(p for p in sorted(stock.glob('*.png')) if Image.open(p).height > Image.open(p).width)
    cases = [('portrait',portrait),('landscape',landscape),('square',square)]
    states, outputs, positions, lengths, indices = [],[],[],[],[]
    meta = {'precision':'CPU f32, FP8 -> bf16 * bf16 scale -> f32', 'cases':{}}
    for name,path in cases:
        rgb = torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).float()/255
        pixels = resize(rgb)
        patches,grid = qwen_vl.process_qwen2vl_images(pixels[None])
        idx,cu = tower.get_window_index(grid)
        pos = tower.get_position_embeddings(grid,torch.device('cpu')).reshape(-1,4,40)[idx].reshape(-1,40)
        pos = torch.cat([pos,pos],-1)
        positions.append((pos.cos(),pos.sin()))
        lengths.append(torch.unique_consecutive(torch.tensor(cu)))
        indices.append(idx)
        outputs.append({'pixels':pixels.contiguous(), 'patches':patches, 'grid':grid, 'window_index':idx, 'cu':lengths[-1]})
        meta['cases'][name] = {'path':str(path), 'input_hw':list(rgb.shape[:2]), 'resized_hw':list(pixels.shape[:2]),'grid':grid.tolist()}
    with safe_open(MODEL, framework='pt', device='cpu') as f:
        with torch.device('meta'):
            patch = qwen_vl.VisionPatchEmbed(embed_dim=1280,dtype=torch.float32,ops=comfy.ops.manual_cast)
        patch.load_state_dict({'proj.weight':tensor(f,'visual.patch_embed.proj.weight')},assign=True)
        for out,idx in zip(outputs,indices):
            x = patch(out['patches'])
            out['patch_embed'] = x.clone()
            states.append(x.reshape(-1,4,1280)[idx].reshape(-1,1280))
        del patch
        for i in range(32):
            with torch.device('meta'):
                block = qwen_vl.VisionBlock(1280,3420,16,dtype=torch.float32,ops=comfy.ops.manual_cast)
            prefix = f'visual.blocks.{i}.'
            block.load_state_dict({k[len(prefix):]:tensor(f,k) for k in f.keys() if k.startswith(prefix) and k.endswith(('.weight','.bias'))},assign=True,strict=True)
            for ci,x in enumerate(states):
                cu = torch.tensor([0,x.shape[0]]) if i in [7,15,23,31] else lengths[ci]
                states[ci] = block(x, positions[ci], cu, optimized_attention=optimized_attention_for_device(torch.device('cpu'),mask=False,small_input=True))
                if i in [0,7,31]: outputs[ci][f'block{i}'] = states[ci].clone()
            del block
            gc.collect()
            print(f'block {i}: {time.perf_counter()-start:.1f}s',flush=True)
            assert peak_bytes() < 24 * 2**30
        with torch.device('meta'):
            merger = qwen_vl.PatchMerger(3584,1280,dtype=torch.float32,ops=comfy.ops.manual_cast)
        prefix = 'visual.merger.'
        merger.load_state_dict({k[len(prefix):]:tensor(f,k) for k in f.keys() if k.startswith(prefix) and k.endswith(('.weight','.bias'))},assign=True,strict=True)
        for (name,_),x,out,idx in zip(cases,states,outputs,indices):
            out['merged'] = merger(x)[torch.argsort(idx)]
            save_file({k:v.contiguous() for k,v in out.items()}, OUT / f'{name}.safetensors')
    meta['peak_working_bytes'] = peak_bytes()
    meta['seconds'] = time.perf_counter()-start
    (OUT / 'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__ == '__main__': main()
