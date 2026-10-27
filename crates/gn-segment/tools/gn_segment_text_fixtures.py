"""Regenerate step 4 references using ComfyUI's SAM CLIP implementation in CPU f32.

Run with ComfyUI's python_embeded/python.exe -I; no model files are copied.
The tokenizer JSON is the same artifact used by gn-perception (override via CLI).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, default=Path(
        r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path, default=Path(
        r'C:\Users\5090\Documents\_eval\native-models\sam3.1\tokenizer.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    # ComfyUI inspects sys.argv while importing; keep generator flags out of its CLI.
    sys.argv = [sys.argv[0], '--cpu']

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from comfy import clip_model, ops, utils
    from comfy.text_encoders.sam3_clip import SAM3_CLIP_CONFIG, SAM3Tokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = args.checkpoint or args.comfy / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    prefix = 'detector.backbone.language_backbone.'
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        state = {k[len(prefix):]: f.get_tensor(k).float()
                 for k in f.keys() if k.startswith(prefix)}
    resizer = torch.nn.Linear(1024, 256)
    resizer.load_state_dict({k.removeprefix('resizer.'): v
                             for k, v in state.items() if k.startswith('resizer.')})
    # Use ComfyUI's conversion and forward path, rather than reimplementing CLIP.
    converted = utils.transformers_convert(state, 'encoder.', '', 24)
    model = clip_model.CLIPTextModel_(SAM3_CLIP_CONFIG, torch.float32, 'cpu', ops.manual_cast)
    model.load_state_dict({k: v for k, v in converted.items() if not k.startswith('resizer.')}, strict=True)
    model.eval()
    resizer.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    comfy_tokenizer = SAM3Tokenizer()
    prompts = [
        'the cup', 'a red umbrella', 'person',
        'a person holding a red umbrella beside a small wooden table with a cup '
        'and a blue plate near the open window in a bright room with white walls '
        'and green plants on the floor',
    ]
    tensors = {}
    records = []
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            pieces = tokenizer.encode(prompt, add_special_tokens=False).ids[:30]
            ids = [49406] + pieces + [49407]
            mask = [True] * len(ids) + [False] * (32 - len(ids))
            ids += [0] * (32 - len(ids))
            batches = comfy_tokenizer.tokenize_with_weights(prompt)
            comfy_ids = [pair[0] for pair in batches[0]]
            equal = ids == comfy_ids
            print(f'{index}: ComfyUI first-window ids equal gn-perception: {equal}; '
                  f'valid={sum(mask)}, ComfyUI windows={len(batches)}', flush=True)
            if not equal:
                raise ValueError(f'tokenizer mismatch: {ids} != {comfy_ids}')
            token_tensor = torch.tensor([ids], dtype=torch.long)
            mask_tensor = torch.tensor([mask], dtype=torch.bool)
            features = resizer(model(token_tensor, mask_tensor)[0]).contiguous()
            tensors[f'{index}.ids'] = token_tensor.to(torch.int32)
            # Candle's safetensors reader does not support BOOL; store boolean 0/1 as U8.
            tensors[f'{index}.mask'] = mask_tensor.to(torch.uint8)
            tensors[f'{index}.features'] = features
            records.append({'prompt': prompt, 'comfy_ids_equal': equal,
                            'comfy_windows': len(batches), 'valid_tokens': sum(mask)})
    root = Path(__file__).resolve().parents[1] / 'tests/fixtures'
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        'torch': torch.__version__, 'dtype': 'f32 including resizer',
        'checkpoint': checkpoint.name,
        'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'prompts': json.dumps(records),
    }
    save_file(tensors, root / 'text.safetensors', metadata=metadata)
    print(f'Wrote {root / "text.safetensors"}', flush=True)


if __name__ == '__main__':
    main()
