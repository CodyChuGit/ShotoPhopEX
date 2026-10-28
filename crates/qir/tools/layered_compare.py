"""Compare saved 1024px Layered RGBA outputs, including the alpha silhouette."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('native', type=Path)
parser.add_argument('--comfy', type=Path, default=Path(r'C:\Users\5090\ComfyUI\photocraft-tests\bench\layered'))
args = parser.parse_args()
for i in range(1, 4):
    a = np.asarray(Image.open(args.native / f'native-{i}.png').convert('RGBA'), dtype=np.float32)
    b = np.asarray(Image.open(args.comfy / f'comfy-{i}.png').convert('RGBA'), dtype=np.float32)
    assert a.shape == b.shape == (1024, 1024, 4)
    am, bm = a[..., 3] > 127, b[..., 3] > 127
    print(json.dumps(dict(layer=i, rgb_mae=float(np.abs(a[..., :3] - b[..., :3]).mean()), alpha_mae=float(np.abs(a[..., 3] - b[..., 3]).mean()), alpha_iou=float(np.logical_and(am, bm).sum() / max(1, np.logical_or(am, bm).sum())))))
