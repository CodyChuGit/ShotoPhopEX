"""Generate interactive f32 parity fixtures with ComfyUI's own modules.

Run with python_embeded/python.exe -I. No FPN arrays or checkpoint weights are saved.
For k=0,1,2 (sizes 288,144,72), both sides compute in f32, in this order:
fpn_k[c,y,x] = 0.5 * sin(((0.013*c + 0.071*y) + 0.031*x) + k).
These are the interactive heads' inputs: k=2 already stands for the post-no-memory feature.
Feedback logits: sin(0.017*y + 0.023*x), shape [1,1,1008,1008].
Samples: floor(i*(288*288-1)/4095), i=0..4095, independently per plane.
Stats: mean, population std, min, max. All calculations and stored tensors are f32.
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
    from comfy.ldm.sam3.tracker import SAMPromptEncoder, SAMMaskDecoder
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.checkpoint or args.comfy / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    pe = SAMPromptEncoder(dtype=torch.float32, device='cpu', operations=ops.manual_cast).eval()
    md = SAMMaskDecoder(dtype=torch.float32, device='cpu', operations=ops.manual_cast).eval()
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        for model, prefix in [(pe, 'tracker.model.interactive_sam_prompt_encoder.'), (md, 'tracker.model.interactive_sam_mask_decoder.')]:
            state = {}
            for key in f.keys():
                if key.startswith(prefix):
                    name = key[len(prefix):].replace('mlp.lin1.', 'mlp.0.').replace('mlp.lin2.', 'mlp.2.').replace('norm_final_attn.', 'norm_final.')
                    state[name] = f.get_tensor(key).float()
            model.load_state_dict(state, strict=True)
    fpn = []
    for k, size in enumerate([288,144,72]):
        c = torch.arange(256, dtype=torch.float32)[:,None,None]
        y = torch.arange(size, dtype=torch.float32)[None,:,None]
        x = torch.arange(size, dtype=torch.float32)[None,None,:]
        fpn.append((0.5 * torch.sin(((0.013*c + 0.071*y) + 0.031*x) + k)).unsqueeze(0))
    y = torch.arange(1008, dtype=torch.float32)[:,None]
    x = torch.arange(1008, dtype=torch.float32)[None,:]
    feedback = torch.sin(0.017*y + 0.023*x)[None,None]
    cases = [
        ([[480.,520.]], [1], None, None),
        ([[0.,0.]], [-1], [120.,210.,780.,870.], None),
        ([[315.,420.],[690.,735.]], [1,0], None, feedback),
    ]
    samples = torch.tensor([i*(288*288-1)//4095 for i in range(4096)])
    tensors = {}
    with torch.inference_mode():
        for index,(points,labels,box,mask) in enumerate(cases):
            coords = torch.tensor([points])
            labels = torch.tensor([labels], dtype=torch.int32)
            boxes = None if box is None else torch.tensor([box])
            dense_mask = None if mask is None else torch.nn.functional.interpolate(mask, (288,288), mode='bilinear', align_corners=False, antialias=True)
            sparse,dense = pe(points=(coords,labels), boxes=boxes, masks=dense_mask)
            # ComfyUI exposes single-mask and multimask slices separately; join both
            # calls to retain all four hypotheses without reimplementing the decoder.
            single = md(fpn[2], pe.get_dense_pe(), sparse, dense, high_res_features=fpn[:2], return_all=True)
            multi = md(fpn[2], pe.get_dense_pe(), sparse, dense, high_res_features=fpn[:2], multimask_output=True, return_all=True)
            masks = torch.cat([single[0],multi[0]],1)[0].flatten(1)
            tensors[f'{index}.samples'] = masks[:,samples].contiguous()
            tensors[f'{index}.stats'] = torch.stack([masks.mean(1),masks.std(1,correction=0),masks.min(1).values,masks.max(1).values],1)
            tensors[f'{index}.iou'] = torch.cat([single[1],multi[1]],1).contiguous()
            tensors[f'{index}.object'] = single[3].contiguous()
            if box is None:
                coords = torch.cat([coords,torch.zeros(1,1,2)],1)
                labels = torch.cat([labels,torch.tensor([[-1]],dtype=torch.int32)],1)
            else:
                coords = torch.cat([coords,boxes.reshape(1,2,2)],1)
                labels = torch.cat([labels,torch.tensor([[2,3]],dtype=torch.int32)],1)
            tensors[f'{index}.points'] = coords.contiguous()
            tensors[f'{index}.labels'] = labels.contiguous()
            print(f'{index}: iou={tensors[f"{index}.iou"].tolist()}, object={single[3].tolist()}', flush=True)
    root = Path(__file__).resolve().parents[1] / 'tests/fixtures'
    save_file(tensors,root / 'interactive.safetensors',metadata={'torch':torch.__version__,'checkpoint':checkpoint.name,'formula':__doc__})
    print('Wrote',root / 'interactive.safetensors',flush=True)


if __name__ == '__main__':
    main()
