"""CPU ComfyUI oracle; never load the complete Layered checkpoint.

Run embedded python.exe -I crates/qir/tools/qwen_image_dit_fixtures.py [--full].
Formula inputs (also in the Rust test): latent=sin(i*.017)*.7 in
[1,16,3,32,32] (two layers plus composite); reference=cos(i*.013)*.4
in [1,16,1,32,32]; text=sin(i*.019)*.3 in [1,12,3584]; time=.625.
Independent block inputs: image=sin(i*.007)*.2 [1,1024,3072],
text=cos(i*.011)*.2 [1,12,3072], temb=sin(i*.003)*.1 [1,3072].
Each tensor records up to 4096 evenly spaced samples, mean/population std/min/max.
--full additionally streams all 60 blocks in bf16, retaining only one block.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], '--cpu', '--use-pytorch-cross-attention']
    import torch
    import comfy.ops
    import comfy_kitchen as ck
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel, QwenImageTransformerBlock
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = args.checkpoint or args.comfy / 'models/diffusion_models/qwen_image_layered_fp8mixed.safetensors'
    if not checkpoint.is_file():
        print(f'Skipping absent {checkpoint}')
        return
    with checkpoint.open('rb') as file:
        length = struct.unpack('<Q', file.read(8))[0]
        header = json.loads(file.read(length))
    base = 8 + length
    dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F8_E4M3': torch.float8_e4m3fn, 'U8': torch.uint8}

    def read(key):
        info = header[key]
        start, end = info['data_offsets']
        with checkpoint.open('rb') as file:
            file.seek(base + start)
            raw = bytearray(file.read(end - start))
        return torch.frombuffer(raw, dtype=dtypes[info['dtype']]).reshape(info['shape'])

    def load(module, prefix='', dtype=torch.float32):
        # Assign one parameter at a time; no state-dict-sized second copy.
        for name, _ in list(module.named_parameters()):
            key = prefix + name
            weight = read(key)
            if weight.dtype == torch.float8_e4m3fn:
                config = json.loads(bytes(read(key.removesuffix('weight') + 'comfy_quant').tolist()))
                assert config == {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True}
                scale = read(key.removesuffix('weight') + 'weight_scale')
                weight = ck.dequantize_per_tensor_fp8(weight, scale, dtype)
            parent, _, leaf = name.rpartition('.')
            setattr(module.get_submodule(parent), leaf, torch.nn.Parameter(weight.to(dtype), requires_grad=False))
        return module

    results = {'checkpoint': str(checkpoint), 'torch': torch.__version__, 'device': 'cpu', 'threads': args.threads, 'tensors': {}}
    def record(name, tensor):
        flat = tensor.float().flatten()
        count = min(4096, flat.numel())
        indices = torch.arange(count, dtype=torch.int64) * (flat.numel() - 1) // max(count - 1, 1)
        results['tensors'][name] = dict(shape=list(tensor.shape), values=flat[indices].tolist(),
                                       mean=flat.mean().item(), std=flat.std(correction=0).item(),
                                       min=flat.min().item(), max=flat.max().item())
        print(f'{name}: {list(tensor.shape)}', flush=True)

    def formula(shape, freq, amp, cosine=False):
        phase = torch.arange(__import__('math').prod(shape), dtype=torch.float32) * freq
        return ((phase.cos() if cosine else phase.sin()) * amp).reshape(shape)

    def shell(dtype):
        return QwenImageTransformer2DModel(num_layers=0, use_additional_t_cond=True,
            default_ref_method='negative_index', dtype=dtype, device='meta', operations=comfy.ops.disable_weight_init)

    started = time.perf_counter()
    with torch.inference_mode():
        model = shell(torch.float32)
        # process_img and EmbedND have no parameters and work directly on CPU inputs.
        latent = formula((1,16,3,32,32), .017, .7)
        reference = formula((1,16,1,32,32), .013, .4, True)
        text = formula((1,12,3584), .019, .3)
        patches, img_ids, _ = model.process_img(latent)
        ref_patches, ref_ids, _ = model.process_img(reference, index=-1)
        img = torch.cat((patches, ref_patches), 1)
        img_ids = torch.cat((img_ids, ref_ids), 1)
        txt_ids = torch.arange(8,20).reshape(1,12,1).repeat(1,1,3)
        rope = model.pe_embedder(torch.cat((txt_ids, img_ids), 1))
        for name, tensor in [('patches', patches), ('reference_patches', ref_patches), ('image_ids', img_ids), ('text_ids', txt_ids), ('rope', rope)]:
            record(name, tensor)
        for name in ['img_in', 'txt_norm', 'txt_in', 'time_text_embed', 'norm_out', 'proj_out']:
            load(getattr(model, name), name + '.')
        record('time_projection', model.time_text_embed.time_proj(torch.tensor([.625])))
        temb = model.time_text_embed(torch.tensor([.625]), img)
        record('time_embedding', temb)
        record('time_embedding_one', model.time_text_embed(torch.tensor([.625]), img, torch.tensor([1])))
        record('image_projection', model.img_in(img))
        record('text_projection', model.txt_in(model.txt_norm(text)))
        block_img = formula((1,1024,3072), .007, .2)
        block_txt = formula((1,12,3072), .011, .2, True)
        block_time = formula((1,3072), .003, .1)
        for i in [0,29,59]:
            t0 = time.perf_counter()
            block = load(QwenImageTransformerBlock(3072,24,128,device='meta',dtype=torch.float32,operations=comfy.ops.disable_weight_init), f'transformer_blocks.{i}.')
            txt_out, img_out = block(block_img, block_txt, None, block_time, rope)
            record(f'block_{i}_image', img_out)
            record(f'block_{i}_text', txt_out)
            results[f'block_{i}_seconds'] = time.perf_counter() - t0
            print(f'block {i}: {results[f"block_{i}_seconds"]:.3f}s', flush=True)
            del block, txt_out, img_out
            gc.collect()
        normalized = model.norm_out(block_img, block_time)
        record('norm_out', normalized)
        record('proj_out', model.proj_out(normalized))
        if args.full:
            # Matching CPU f32 oracle for Rust's device-generic forward.
            fimg = model.img_in(img)
            ftxt = model.txt_in(model.txt_norm(text))
            t0 = time.perf_counter()
            for i in range(60):
                block = load(QwenImageTransformerBlock(3072,24,128,device='meta',dtype=torch.float32,operations=comfy.ops.disable_weight_init), f'transformer_blocks.{i}.')
                ftxt, fimg = block(fimg, ftxt, None, temb, rope)
                del block
                gc.collect()
                print(f'full f32 block {i}: {time.perf_counter()-t0:.3f}s cumulative', flush=True)
            record('full_f32_velocity', model.proj_out(model.norm_out(fimg,temb))[:, :768])
            results['full_f32_seconds'] = time.perf_counter()-t0
            del fimg, ftxt
            # Stream bf16 weights; this is a second oracle, not f32 block fixtures.
            del model, block_img, block_txt, block_time, normalized
            gc.collect()
            model = shell(torch.bfloat16)
            for name in ['img_in','txt_norm','txt_in','time_text_embed','norm_out','proj_out']:
                load(getattr(model, name), name + '.', torch.bfloat16)
            img = model.img_in(img.to(torch.bfloat16))
            text = model.txt_in(model.txt_norm(text.to(torch.bfloat16)))
            temb = model.time_text_embed(torch.tensor([.625]), img)
            rope = rope.to(torch.bfloat16)
            t0 = time.perf_counter()
            for i in range(60):
                block = load(QwenImageTransformerBlock(3072,24,128,device='meta',dtype=torch.bfloat16,operations=comfy.ops.disable_weight_init), f'transformer_blocks.{i}.', torch.bfloat16)
                text, img = block(img, text, None, temb, rope)
                del block
                gc.collect()
                print(f'full bf16 block {i}: {time.perf_counter()-t0:.3f}s cumulative', flush=True)
            record('full_bf16_velocity', model.proj_out(model.norm_out(img,temb))[:, :768])
            results['full_bf16_seconds'] = time.perf_counter()-t0
    results['seconds'] = time.perf_counter()-started
    # Windows process counters include the peak working set, including native allocations.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counters = Counters()
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters))
    assert counters.PeakWorkingSetSize < 25 * 2**30
    results['peak_working_set_bytes'] = counters.PeakWorkingSetSize
    dest = Path(__file__).resolve().parents[1] / 'tests/fixtures/qwen_image_dit.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(results,separators=(',',':')) + '\n')
    print(f'Wrote {dest}: {dest.stat().st_size} bytes, {results["seconds"]:.3f}s, peak RSS {counters.PeakWorkingSetSize/2**30:.3f} GiB', flush=True)


if __name__ == '__main__':
    main()
