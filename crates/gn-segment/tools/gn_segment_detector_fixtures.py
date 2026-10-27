"""ComfyUI f32 CPU detector parity, using raw checkpoint weights.

Run python_embeded/python.exe -I. Both sides evaluate these formulas in f32:
fpn_k[c,y,x] = 0.5*sin(((0.013*c + 0.071*y) + 0.031*x) + k),
k=0,1,2 at sizes 288,144,72. No tracker no-memory embedding is added.
p[t,c] = 0.3*cos(0.05*c + 0.7*t), eight tokens, final two padding.
Positions are ComfyUI PositionEmbeddingSine(normalize=True), 256 channels.
Samples are floor(i*(200*288*288-1)/4095), i=0..4095 across all masks.
Stats are global mean, population std, min, max. Top-three indices sort logits.
References are cxcywh after each refinement; final public boxes are xyxy.
"""
import argparse
from pathlib import Path
import sys
import time


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
    from comfy.ldm.sam3.detector import Transformer, DotProductScoring, SegmentationHead, box_cxcywh_to_xyxy
    from comfy.ldm.sam3.sam import PositionEmbeddingSine
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.checkpoint or args.comfy / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    kwargs = dict(device='cpu', dtype=torch.float32, operations=ops.manual_cast)
    transformer = Transformer(**kwargs).eval()
    scoring = DotProductScoring(**kwargs).eval()
    segmentation = SegmentationHead(**kwargs).eval()
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        for model, prefix in [(transformer, 'detector.transformer.'), (scoring, 'detector.dot_prod_scoring.'), (segmentation, 'detector.segmentation_head.')]:
            state = {}
            for key in f.keys():
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix):]
                tensor = f.get_tensor(key).float()
                if name.endswith(('in_proj_weight', 'in_proj_bias')):
                    base, suffix = name.rsplit('.', 1)
                    suffix = 'weight' if suffix.endswith('weight') else 'bias'
                    for proj, value in zip(['q_proj','k_proj','v_proj'], tensor.chunk(3, dim=0)):
                        state[f'{base}.{proj}.{suffix}'] = value.contiguous()
                else:
                    state[name] = tensor
            model.load_state_dict(state, strict=True)
    fpn = []
    for k, size in enumerate([288,144,72]):
        c = torch.arange(256, dtype=torch.float32)[:,None,None]
        y = torch.arange(size, dtype=torch.float32)[None,:,None]
        x = torch.arange(size, dtype=torch.float32)[None,None,:]
        fpn.append((0.5*torch.sin(((0.013*c + 0.071*y) + 0.031*x) + k))[None])
    t = torch.arange(8, dtype=torch.float32)[:,None]
    c = torch.arange(256, dtype=torch.float32)[None,:]
    prompt = (0.3*torch.cos(0.05*c + 0.7*t))[None]
    valid = torch.tensor([[True]*6 + [False]*2])
    refs = []
    original_bias = transformer.decoder._compute_box_rpb
    def capture(ref, h, w):
        refs.append(ref.clone())
        return original_bias(ref, h, w)
    transformer.decoder._compute_box_rpb = capture
    start = time.perf_counter()
    with torch.inference_mode():
        pos = PositionEmbeddingSine(num_pos_feats=256, normalize=True)(fpn[2]).flatten(2).transpose(1,2)
        memory = transformer.encoder(fpn[2].flatten(2).transpose(1,2), pos, prompt, valid)
        dec = transformer.decoder(memory, pos, prompt, valid)
        logits = scoring(dec['decoder_output'], prompt, valid)
        masks = segmentation(dec['decoder_output'], fpn, memory, prompt, valid)
        top = logits[0].argsort(descending=True)[:3]
        flat = masks.flatten()
        samples = torch.tensor([i*(flat.numel()-1)//4095 for i in range(4096)])
        tensors = dict(boxes=box_cxcywh_to_xyxy(dec['pred_boxes']), logits=logits,
                       presence=dec['presence'], samples=flat[samples],
                       stats=torch.stack([flat.mean(),flat.std(correction=0),flat.min(),flat.max()]),
                       top_indices=top, top_masks=masks[0,top].contiguous(),
                       references=torch.stack(refs[1:]+[dec['pred_boxes']],0))
    elapsed = time.perf_counter()-start
    print(f'CPU forward {elapsed:.3f}s; logits min/max/std {logits.min().item():.6f}/{logits.max().item():.6f}/{logits.std().item():.6f}; top={top.tolist()}; presence={dec["presence"].tolist()}', flush=True)
    if logits.std().item() < 1e-5:
        raise RuntimeError('Formula inputs are degenerate; select richer inputs before saving')
    dest = Path(__file__).resolve().parents[1] / 'tests/fixtures/detector.safetensors'
    save_file({k:v.contiguous() for k,v in tensors.items()},dest,metadata={'torch':torch.__version__,'checkpoint':checkpoint.name,'formula':__doc__,'cpu_seconds':str(elapsed)})
    assert dest.stat().st_size < 2_000_000
    print('Wrote',dest,dest.stat().st_size,flush=True)


if __name__ == '__main__':
    main()

