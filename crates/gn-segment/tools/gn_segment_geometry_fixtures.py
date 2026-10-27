"""ComfyUI f32 CPU geometry parity, run with python_embeded/python.exe -I.

Both sides generate fpn[c,y,x] = 0.5*sin((0.013*c + 0.071*y) + 0.031*x),
in f32 at detector level 2: [1,256,72,72]. Positions use PositionEmbeddingSine
(Rust image_position). Large formula inputs are represented by this metadata;
box inputs and complete token outputs are stored, keeping the fixture below 2 MB.
Pixel corners refer to an original 1200x800 image and are converted to cx,cy,w,h
before encoding. Cases: middle, tiny (3x4 pixels), full, outside, three boxes.
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], '--cpu']
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from comfy import ops
    from comfy.ldm.sam3.detector import GeometryEncoder, SAM3Detector
    from comfy.ldm.sam3.sam import PositionEmbeddingSine
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.checkpoint or args.comfy / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    model = GeometryEncoder(dtype=torch.float32, device='cpu', operations=ops.manual_cast).eval()
    state = {}
    prefix = 'detector.geometry_encoder.'
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name.endswith('in_proj_weight') or name.endswith('in_proj_bias'):
                suffix = 'weight' if name.endswith('weight') else 'bias'
                base = name.rsplit('.', 1)[0]
                for part, value in zip(['q', 'k', 'v'], f.get_tensor(key).float().chunk(3, 0)):
                    state[f'{base}.{part}_proj.{suffix}'] = value
            else:
                state[name] = f.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    c = torch.arange(256, dtype=torch.float32)[:, None, None]
    y = torch.arange(72, dtype=torch.float32)[None, :, None]
    x = torch.arange(72, dtype=torch.float32)[None, None, :]
    fpn = (0.5*torch.sin((0.013*c + 0.071*y) + 0.031*x))[None]
    image = fpn.flatten(2).permute(0, 2, 1)
    pos = PositionEmbeddingSine()(fpn).flatten(2).permute(0, 2, 1)
    cases = {
        'middle': [[300, 200, 900, 600]],
        'tiny': [[601, 399, 604, 403]],
        'full': [[0, 0, 1200, 800]],
        'outside': [[-180, -120, 480, 400]],
        'three': [[300, 200, 900, 600], [601, 399, 604, 403], [-180, -120, 480, 400]],
    }
    tensors = {}
    with torch.inference_mode():
        for name, corners in cases.items():
            corners = torch.tensor(corners, dtype=torch.float32)
            boxes = torch.stack([(corners[:, 0]+corners[:, 2])*0.5/1200,
                                 (corners[:, 1]+corners[:, 3])*0.5/800,
                                 (corners[:, 2]-corners[:, 0])/1200,
                                 (corners[:, 3]-corners[:, 1])/800], -1)[None]
            tensors[f'{name}.boxes'] = boxes.contiguous()
            tensors[f'{name}.tokens'] = model(boxes=boxes, image_features=image).contiguous()
            print(name, tensors[f'{name}.tokens'].shape, flush=True)
        cls = model.norm(model.final_proj(model.cls_embed.weight.view(1, 1, 256)))
        for layer in model.encode:
            cls = SAM3Detector._run_geo_layer(layer, cls, image, pos)
        tensors['cls'] = model.encode_norm(cls).contiguous()
    target = Path(__file__).resolve().parents[1] / 'tests/fixtures/geometry.safetensors'
    save_file(tensors, target, metadata={'formula': __doc__, 'torch': torch.__version__, 'checkpoint': checkpoint.name})
    print('Wrote', target, target.stat().st_size, 'bytes', flush=True)


if __name__ == '__main__':
    main()
