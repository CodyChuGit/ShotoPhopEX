"""Generate compact vision parity fixtures from ComfyUI's own SAM 3.1 on CPU f32.

Run with python_embeded/python.exe -I. The deterministic RGB picture is a sky,
ground, sun and building scene generated with integer arithmetic (no image decoder).
Only statistics and evenly spaced fixed samples are stored, never full features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], '--cpu']
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from comfy import ops
    from comfy.ldm.sam3.sam import ViTDet, FPNScaleConv, PositionEmbeddingSine

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.checkpoint or args.comfy / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    start = time.perf_counter()
    tensors = {}
    records = {}

    def record(name, x, samples=True):
        flat = x.detach().contiguous().flatten().float()
        stats = torch.stack([flat.mean(), flat.std(correction=0), flat.min(), flat.max()])
        tensors[name + '.stats'] = stats
        if samples:
            indices = torch.arange(4096, dtype=torch.int64) * (flat.numel() - 1) // 4095
            tensors[name + '.indices'] = indices
            tensors[name + '.samples'] = flat[indices]
        records[name] = {'shape': list(x.shape), 'layout': 'NHWC' if name.startswith('block') else 'NCHW'}
        print(f'{name}: {list(x.shape)}, mean/std/min/max={stats.tolist()}, elapsed={time.perf_counter()-start:.2f}s', flush=True)

    yy, xx = torch.meshgrid(torch.arange(1008), torch.arange(1008), indexing='ij')
    picture = torch.stack([80+xx*80//1008, 140+yy*60//1008, 220-yy*70//1008])
    ground = yy >= 650
    picture[:, ground] = torch.stack([60+xx*30//1008, 100+xx*70//1008, 35+yy*30//1008])[:, ground]
    sun = (xx-760)**2 + (yy-220)**2 < 110**2
    picture[:, sun] = torch.tensor([255,225,90])[:,None]
    building = (xx>=180)&(xx<480)&(yy>=360)&(yy<820)
    picture[:, building] = torch.tensor([170,75,50])[:,None]
    window = (xx>=250)&(xx<330)&(yy>=440)&(yy<560)
    picture[:, window] = torch.tensor([35,60,85])[:,None]
    pixels = picture.float().unsqueeze(0)/255

    with safe_open(checkpoint, framework='pt', device='cpu') as f, torch.inference_mode():
        prefix = 'detector.backbone.vision_backbone.'
        trunk = ViTDet(device='cpu', dtype=torch.float32, operations=ops.manual_cast)
        trunk.load_state_dict({k[len(prefix+'trunk.'):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix+'trunk.')}, strict=True)
        trunk.eval()
        trunk.patch_embed.register_forward_hook(lambda m,a,x: record('patch', x, False))
        for i in [0,7,31]:
            trunk.blocks[i].register_forward_hook(lambda m,a,x,i=i: record(f'block{i}', x, i!=0))
        features = trunk(pixels)
        for family in ['convs','interactive_convs']:
            for i,scale in enumerate([4.0,2.0,1.0]):
                branch = FPNScaleConv(1024,256,scale,device='cpu',dtype=torch.float32,operations=ops.manual_cast)
                bp = f'{prefix}{family}.{i}.'
                branch.load_state_dict({k[len(bp):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(bp)}, strict=True)
                branch.eval()
                out = branch(features)
                if family == 'interactive_convs' and i==2:
                    out = out + f.get_tensor('tracker.model.interactivity_no_mem_embed').float().reshape(1,256,1,1)
                record(('detector' if family=='convs' else 'interactive')+str(i), out)
                del out, branch
        record('image_pos', PositionEmbeddingSine()(features))

    root = Path(__file__).resolve().parents[1]/'tests/fixtures'
    root.mkdir(parents=True, exist_ok=True)
    metadata = {'torch': torch.__version__, 'dtype': 'f32 CPU; raw F16 weights cast to f32',
                'checkpoint': checkpoint.name, 'picture_sha256': hashlib.sha256(picture.to(torch.uint8).numpy().tobytes()).hexdigest(),
                'stats': 'mean,population_std,min,max', 'indices': 'floor(i*(numel-1)/4095), i=0..4095; contiguous tensor order',
                'records': json.dumps(records), 'seconds': str(time.perf_counter()-start)}
    destination = root/'vision.safetensors'
    save_file(tensors, destination, metadata=metadata)
    assert destination.stat().st_size < 2_000_000
    print(f'Wrote {destination}, {destination.stat().st_size} bytes, {metadata["seconds"]} seconds', flush=True)


if __name__ == '__main__':
    main()
