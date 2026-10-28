"""Compare Layered RGBA outputs without resizing; RGB uses the union of visible pixels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('native', type=Path)
parser.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\photocraft-tests\bench\layered'))
parser.add_argument('--native-prefix', default='native')
parser.add_argument('--reference-prefix', default='comfy')
parser.add_argument('--rgb-limit', type=float)
parser.add_argument('--visible-alpha-threshold', type=int, default=127)
parser.add_argument('--iou-limit', type=float, default=0.99)
parser.add_argument('--pad-canvas', action='store_true', help='Top-left placement on a common transparent canvas; no resizing.')
args = parser.parse_args()
passed = True
for i in range(1, 4):
    a = Image.open(args.native / f'{args.native_prefix}-{i}.png').convert('RGBA')
    b = Image.open(args.comfy / f'{args.reference_prefix}-{i}.png').convert('RGBA')
    if args.pad_canvas:
        size = (max(a.width, b.width), max(a.height, b.height))
        def canvas(image):
            out = Image.new('RGBA', size)
            out.paste(image, (0, 0))
            return out
        a, b = canvas(a), canvas(b)
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f'layer {i}: mismatched shapes {a.shape} vs {b.shape}')
    am, bm = a[..., 3] > 127, b[..., 3] > 127
    visible = (a[..., 3] > args.visible_alpha_threshold) | (b[..., 3] > args.visible_alpha_threshold)
    rgb = float(np.abs(a[..., :3] - b[..., :3])[visible].mean()) if visible.any() else 0.0
    union = int(np.logical_or(am, bm).sum())
    iou = float(np.logical_and(am, bm).sum() / union) if union else 1.0
    print(json.dumps(dict(layer=i, rgb_mae=rgb, rgb_mae_all=float(np.abs(a[..., :3] - b[..., :3]).mean()), alpha_mae=float(np.abs(a[..., 3] - b[..., 3]).mean()), alpha_iou=iou, identical=bool(np.array_equal(a, b)))))
    if args.rgb_limit is not None:
        passed &= rgb <= args.rgb_limit and iou >= args.iou_limit
if not passed:
    raise SystemExit('Layered parity gate failed')
