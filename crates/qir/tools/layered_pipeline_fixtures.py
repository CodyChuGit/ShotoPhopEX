"""ComfyUI oracle for R3 steps 5/6. Stream blocks; never load the full DiT.

Without --full, write the small complete CPU sampler fixture. With --full,
capture both CFG branches, every latent, retained slices and float decodes.
--device cuda is for the subsequent GPU comparison, not the CPU bring-up.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    p.add_argument('--models', type=Path)
    p.add_argument('--full', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--height', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--prompt', default='A red cat.')
    p.add_argument('--input', type=Path, help='Optional PNG, already at the requested resolution')
    p.add_argument('--out', type=Path)
    p.add_argument('--compare', type=Path, help='Rust capture safetensors; report max absolute and relative RMS')
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], '--cpu' if args.device == 'cpu' else '--normalvram', '--use-pytorch-cross-attention']
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    import comfy.ops
    import comfy.sample
    import comfy.samplers
    from comfy.model_sampling import ModelSamplingDiscreteFlow, CONST
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel, QwenImageTransformerBlock
    from comfy_extras.nodes_qwen import EmptyQwenImageLayeredLatentImage
    import comfy_extras.nodes_post_processing  # ComfyUI's node import order breaks a circular import.
    from comfy_extras.nodes_latent import LatentCut, LatentCutToBatch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    out = {}
    meta = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, torch=torch.__version__)
    models = args.models or args.comfy / 'models'

    def record(name, x):
        out[name] = x.detach().float().cpu().contiguous().clone()
        print(f'{name}: {list(x.shape)} {time.perf_counter()-started:.3f}s', flush=True)

    class Sampling(ModelSamplingDiscreteFlow, CONST):
        pass

    sampling = Sampling()
    sampling.set_parameters(shift=1.0, multiplier=1.0)
    shell = QwenImageTransformer2DModel(num_layers=0, use_additional_t_cond=True,
        default_ref_method='negative_index', device='meta', dtype=torch.float32, operations=comfy.ops.disable_weight_init)

    with torch.inference_mode():
        if not args.full:
            # Non-square, multiple slices and two seeds expose channel/time transpositions.
            for seed in [7, 4294967303]:
                empty = EmptyQwenImageLayeredLatentImage.execute(48, 32, 2)[0]['samples'].cpu()
                noise = comfy.sample.prepare_noise(empty, seed)
                packed, _, shape = shell.process_img(noise)
                unpacked = packed.view(1, 3, 2, 3, 16, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(shape)
                record(f'noise_{seed}', noise)
                record(f'packed_{seed}', packed)
                record(f'unpacked_{seed}', unpacked)
            sigmas = comfy.samplers.simple_scheduler(sampling, 20)
            record('sigmas', sigmas)
            record('timesteps', sampling.timestep(sigmas))
            record('qwen21_rounded', ((sigmas * 1000).bfloat16() / 1000).bfloat16())
            x = out['noise_7']
            pos = (torch.arange(x.numel()).float() * .017).sin().reshape(x.shape)
            neg = (torch.arange(x.numel()).float() * .013).cos().reshape(x.shape)
            record('positive', pos)
            record('negative', neg)
            interval = sigmas[7:9]
            def denoiser(x, sigma):
                return comfy.samplers.cfg_function(None, sampling.calculate_denoised(sigma, pos, x),
                    sampling.calculate_denoised(sigma, neg, x), 2.5, x, sigma)
            record('euler', sample_euler(denoiser, x, interval, disable=True))
        else:
            assert args.width > 0 and args.height > 0 and args.width % 16 == args.height % 16 == 0
            assert 1 <= args.layers <= 8 and args.steps > 0
            device = torch.device(args.device)
            dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32

            def read(f, key, legacy=False):
                w = f.get_tensor(key).to(device)
                if w.dtype == torch.float8_e4m3fn:
                    scale = f.get_tensor(key.removesuffix('weight') + ('scale_weight' if legacy else 'weight_scale')).to(device)
                    # Match the independently tested encoder's bf16 dequantization.
                    compute = torch.bfloat16 if legacy else dtype
                    w = w.to(compute) * scale.to(compute)
                return w.to(dtype)

            def load(module, path, prefix='', legacy=False):
                with safe_open(path, framework='pt', device='cpu') as f:
                    for name, _ in list(module.named_parameters()):
                        parent, _, leaf = name.rpartition('.')
                        setattr(module.get_submodule(parent), leaf, torch.nn.Parameter(read(f, prefix + name, legacy), requires_grad=False))
                return module

            from comfy.ldm.wan.vae import WanVAE
            from comfy.latent_formats import Wan21
            from PIL import Image
            import numpy as np
            def vae():
                with torch.device('meta'):
                    model = WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True], image_channels=4, conv_out_channels=4)
                return load(model, models / 'vae/qwen_image_layered_vae.safetensors')

            if args.input:
                picture = Image.open(args.input).convert('RGBA')
                assert picture.size == (args.width, args.height)
                rgb = torch.from_numpy(np.array(picture)[..., :3].copy()).float() / 255
            else:
                # Same byte pattern as the Rust smoke test; input alpha is unused by LoadImage.
                rgb = (torch.arange(args.width*args.height*4).reshape(args.height,args.width,4) % 256)[..., :3].float() / 255
            pixels = (rgb.permute(2,0,1)[None,None] * 2 - 1).transpose(1,2)
            pixels = torch.cat((pixels, torch.ones(1,1,1,args.height,args.width)), 1).to(device=device,dtype=dtype)
            model = vae()
            reference = Wan21().process_in(model.encode(pixels).float())
            record('reference', reference)
            del model, pixels
            gc.collect()

            from comfy.text_encoders import llama, qwen_image
            from comfy.ldm.modules.attention import optimized_attention_for_device
            tokenizer = qwen_image.QwenImageTokenizer()
            ids = [torch.tensor([[v[0] for v in tokenizer.tokenize_with_weights(s)['qwen25_7b'][0]]], device=device) for s in [args.prompt, '']]
            text_path = models / 'text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors'
            with safe_open(text_path, framework='pt', device='cpu') as f:
                emb = f.get_tensor('model.embed_tokens.weight')
                states = [emb[t.cpu()].to(device=device,dtype=dtype) for t in ids]
                del emb
                cfg = llama.Qwen25_7BVLI_Config()
                freqs = [llama.precompute_freqs_cis(128, torch.arange(t.shape[1],device=device).unsqueeze(0), 1e6) for t in ids]
                masks = [torch.full((t.shape[1],t.shape[1]),torch.finfo(torch.float32).min/4,device=device).triu_(1) for t in ids]
                for li in range(28):
                    with torch.device('meta'):
                        layer = llama.TransformerBlock(cfg,li,dtype=dtype,ops=comfy.ops.manual_cast)
                    load(layer,text_path,f'model.layers.{li}.',legacy=True)
                    for i in range(2):
                        states[i] = layer(x=states[i],attention_mask=masks[i],freqs_cis=freqs[i],
                            optimized_attention=optimized_attention_for_device(device,mask=True,small_input=True))[0]
                    del layer
                norm = llama.RMSNorm(3584,eps=1e-6,dtype=dtype,device=device)
                norm.load_state_dict({'weight': read(f,'model.norm.weight')})
                texts = []
                for i, tokens in enumerate(ids):
                    values = tokens[0].tolist()
                    cut = [j for j,v in enumerate(values) if v == 151644][:2][-1]
                    if len(values) > cut+3 and values[cut+1:cut+3] == [872,198]:
                        cut += 3
                    texts.append(norm(states[i])[:,cut:].contiguous())
            record('positive_text',texts[0])
            record('negative_text',texts[1])
            del states, norm
            gc.collect()

            checkpoint = models / 'diffusion_models/qwen_image_layered_fp8mixed.safetensors'
            # ComfyUI's own forward performs patching, positions and reference concatenation.
            # Forward hooks materialize only the block currently executing.
            model = QwenImageTransformer2DModel(num_layers=0,use_additional_t_cond=True,
                default_ref_method='negative_index',device='meta',dtype=dtype,operations=comfy.ops.disable_weight_init)
            for name in ['img_in','txt_norm','txt_in','time_text_embed','norm_out','proj_out']:
                load(getattr(model,name),checkpoint,name+'.')
            class StreamBlock(torch.nn.Module):
                def __init__(self, index):
                    super().__init__()
                    self.index = index
                def forward(self,*a,**kw):
                    block = QwenImageTransformerBlock(3072,24,128,device='meta',dtype=dtype,operations=comfy.ops.disable_weight_init)
                    load(block,checkpoint,f'transformer_blocks.{self.index}.')
                    return block(*a,**kw)
            model.transformer_blocks = torch.nn.ModuleList([StreamBlock(i) for i in range(60)])
            empty = EmptyQwenImageLayeredLatentImage.execute(args.width,args.height,args.layers)[0]['samples'].cpu()
            noise = comfy.sample.prepare_noise(empty,args.seed).to(device)
            record('noise',noise)
            sigmas = comfy.samplers.simple_scheduler(sampling,args.steps).to(device)
            current = sampling.noise_scaling(sigmas[0],noise,empty.to(device))
            for i in range(args.steps):
                sigma = sigmas[i:i+1]
                velocities = []
                for label, text in zip(['positive','negative'],texts):
                    velocity = model(current.to(dtype),sampling.timestep(sigma).float(),text,ref_latents=[reference.to(dtype)]).float()
                    record(f'step_{i}.{label}',velocity)
                    velocities.append(velocity)
                def denoiser(x, sigma):
                    pos, neg = [sampling.calculate_denoised(sigma,v,x) for v in velocities]
                    return comfy.samplers.cfg_function(None,pos,neg,2.5,x,sigma)
                current = sample_euler(denoiser,current,sigmas[i:i+2],disable=True)
                record(f'step_{i}.latent',current)
            del model
            gc.collect()
            retained = LatentCut.execute({'samples':current},'t',1,16384)[0]
            retained = LatentCutToBatch.execute(retained,'t',1)[0]['samples']
            record('retained',retained.squeeze(2))
            model = vae()
            for i in range(args.layers):
                decoded = model.decode(Wan21().process_out(retained[i:i+1]).to(dtype)).float().squeeze(2)
                record(f'decoded_{i}',decoded)
            del model

    # .bin contains safetensors; qir ignores the .safetensors extension as model weights.
    target = args.out or Path(__file__).resolve().parents[1] / 'tests/fixtures' / ('layered_pipeline_cpu.bin' if args.full else 'layered_pipeline.bin')
    target.parent.mkdir(parents=True,exist_ok=True)
    save_file(out,target)
    meta['seconds'] = time.perf_counter()-started
    if os.name == 'nt':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)] + [(k,ctypes.c_size_t) for k in ['peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(counters),counters.cb)
        meta['peak_working_bytes'] = counters.peak_working
        assert counters.peak_working < 25_000_000_000
    if args.compare:
        actual = load_file(args.compare)
        assert set(out) == set(actual), (set(out)-set(actual),set(actual)-set(out))
        meta['errors'] = {}
        def errors(actual, reference):
            error = actual - reference
            return dict(max_abs=error.abs().max().item(),relative_rms=(error.square().mean()/reference.square().mean().clamp_min(1e-24)).sqrt().item())
        for key, reference in out.items():
            assert actual[key].shape == reference.shape, key
            assert torch.isfinite(actual[key]).all() and torch.isfinite(reference).all(), key
            meta['errors'][key] = errors(actual[key],reference)
            if key.startswith('decoded_'):
                meta['errors'][key+'.rgb'] = errors(actual[key][:,:3],reference[:,:3])
                meta['errors'][key+'.alpha'] = errors(actual[key][:,3:],reference[:,3:])
                a, b = [((x+1)/2).clamp(0,1).mul(255).to(torch.uint8) for x in [actual[key],reference]]
                meta['errors'][key+'.bytes'] = dict(max_abs=(a.int()-b.int()).abs().max().item(),different=(a!=b).sum().item(),count=a.numel())
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == '__main__':
    main()
