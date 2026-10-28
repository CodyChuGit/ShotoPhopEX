"""CPU ComfyUI 2511 oracle. Stream one real checkpoint block at a time.

Run portable python_embeded/python.exe -I this_file.py [--full].
Small geometry fixtures are checked in; checkpoint outputs are regenerated.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--full', action='store_true')
    p.add_argument('--geometry-only', action='store_true')
    args = p.parse_args()
    for name in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '8'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], '--cpu', '--use-pytorch-cross-attention']
    import torch
    import comfy.ops
    import comfy_kitchen as ck
    from safetensors.torch import save_file
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel, QwenImageTransformerBlock
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    print(f"ComfyUI CPU oracle: torch {torch.__version__}, threads=8", flush=True)
    checkpoint = args.checkpoint or args.comfy / 'models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors'
    with checkpoint.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
    assert '__index_timestep_zero__' in header
    assert 'time_text_embed.addition_t_embedding.weight' not in header
    base = length + 8
    dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F8_E4M3': torch.float8_e4m3fn, 'U8': torch.uint8}

    def read(key):
        info = header[key]
        start, end = info['data_offsets']
        with checkpoint.open('rb') as f:
            f.seek(base + start)
            raw = bytearray(f.read(end - start))
        return torch.frombuffer(raw, dtype=dtypes[info['dtype']]).reshape(info['shape'])

    def load(module, prefix=''):
        for name, _ in list(module.named_parameters()):
            key = prefix + name
            weight = read(key)
            if weight.dtype == torch.float8_e4m3fn:
                stem = key.removesuffix('weight')
                config = json.loads(bytes(read(stem + 'comfy_quant').tolist()))
                assert config == {'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True}
                assert read(stem + 'input_scale').item() == 1
                weight = ck.dequantize_per_tensor_fp8(weight, read(stem + 'weight_scale'), torch.float32)
            parent, _, leaf = name.rpartition('.')
            setattr(module.get_submodule(parent), leaf, torch.nn.Parameter(weight.float(), requires_grad=False))
        return module

    def block(i):
        return load(QwenImageTransformerBlock(3072,24,128, device='meta', dtype=torch.float32,
                    operations=comfy.ops.disable_weight_init), f'transformer_blocks.{i}.')

    class StreamedBlock(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
        def forward(self, *inputs, **kwargs):
            # ComfyUI's loop holds its previous block until the next is yielded.
            # Keep only a light proxy there, freeing weights inside each call.
            module = block(self.index)
            result = module(*inputs, **kwargs)
            del module
            gc.collect()
            print(f'full block {self.index}', flush=True)
            return result

    def formula(shape, freq, amp, cosine=False):
        phase = torch.arange(__import__('math').prod(shape), dtype=torch.float32) * freq
        return ((phase.cos() if cosine else phase.sin()) * amp).reshape(shape)

    dest = Path(__file__).resolve().parents[1] / 'tests/fixtures'
    small, real = {}, {}
    def record(group, name, tensor):
        group[name] = tensor.detach().float().contiguous().clone()
        print(name, list(tensor.shape), flush=True)

    with torch.inference_mode():
        model = QwenImageTransformer2DModel(num_layers=0, default_ref_method='index_timestep_zero',
                    use_additional_t_cond=False, device='meta', dtype=torch.float32,
                    operations=comfy.ops.disable_weight_init)
        target = formula((1,16,1,5,6), .017,.7)
        refs = [formula((1,16,1,3,7), .013,.4,True), formula((1,16,1,6,3), .023,.25), formula((1,16,1,1,1), .031,.2)]
        text = formula((1,5,3584), .019,.3)
        time_input = torch.tensor([.625, 0.])
        for name,tensor in [('target',target),('ref1',refs[0]),('ref2',refs[1]),('ref3',refs[2]),('text',text)]:
            record(small,name,tensor)
        packed, ids, _ = model.process_img(target)
        num_target = packed.shape[1]
        for i,ref in enumerate(refs):
            rp, ri, _ = model.process_img(ref,index=i+1)
            packed = torch.cat((packed,rp),1)
            ids = torch.cat((ids,ri),1)
        txt_ids = torch.arange(1,6).reshape(1,5,1).repeat(1,1,3)
        rope = model.pe_embedder(torch.cat((txt_ids,ids),1))
        for name,tensor in [('packed',packed),('image_ids',ids),('text_ids',txt_ids),('rope',rope)]:
            record(small,name,tensor)
        record(small,'time_projection',model.time_text_embed.time_proj(torch.tensor([.625,0.,.125,.999])))
        save_file(small,str(dest/'qwen_edit_2511_geometry.bin'))
        if args.geometry_only:
            return
        for name in ['img_in','txt_norm','txt_in','time_text_embed','norm_out','proj_out']:
            load(getattr(model,name),name+'.')
        temb = model.time_text_embed(time_input,packed)
        record(real,'temb',temb)
        # One different-sized reference, with common boundary tensors for all three blocks.
        one_rope = model.pe_embedder(torch.cat((txt_ids,ids[:,:num_target+8]),1))
        # Independent boundaries like the Layered oracle, stored in full so Rust
        # receives identical inputs. Standard normal streams; timestep stddev .1.
        generator = torch.Generator(device='cpu').manual_seed(0)
        image = torch.randn((1,num_target+8,3072), generator=generator)
        txt = torch.randn((1,5,3072), generator=generator)
        block_time = torch.randn((2,3072), generator=generator) * .1
        record(real,'block_input_time',block_time)
        record(real,'block_input_image',image)
        record(real,'block_input_text',txt)
        for i in (0,29,59):
            module = block(i)
            params = module.img_mod(temb)
            record(real,f'block_{i}_modulation',params)
            for half in (0,1):
                mod, gates = module._modulate(module.img_norm1(image),params.chunk(2,-1)[half],num_target)
                record(real,f'block_{i}_modulated_{half}',mod)
                record(real,f'block_{i}_gates_{half}',torch.cat((gates[0].expand(-1,num_target,-1), gates[1].expand(-1,8,-1)),1))
            txt_out,img_out = module(image,txt,None,block_time,one_rope,timestep_zero_index=num_target)
            record(real,f'block_{i}_image',img_out)
            record(real,f'block_{i}_text',txt_out)
            del module,txt_out,img_out,params
            gc.collect()
        if args.full:
            model.transformer_blocks = torch.nn.ModuleList(StreamedBlock(i) for i in range(60))
            for count in (1,2):
                # Invoke ComfyUI's actual packing, timestep, block loop and crop path.
                result = model(target,torch.tensor([.625]),text,ref_latents=refs[:count])
                record(real,f'velocity_{count}',result)
        save_file(real,str(dest/'qwen_edit_2511_real.safetensors'))
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counters = Counters()
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),ctypes.sizeof(counters))
    assert counters.PeakWorkingSetSize < 24_000_000_000
    print(f'Peak working set: {counters.PeakWorkingSetSize/2**30:.3f} GiB',flush=True)


if __name__ == '__main__':
    main()

