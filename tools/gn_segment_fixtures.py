"""Regenerate small CPU float32 PyTorch references for gn-segment step 1."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from safetensors.torch import save_file
from torchvision.ops import roi_align


def signal(shape):
    """Avoid RNG version dependence while exercising positive and negative values."""
    t = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    return (torch.sin(t * 0.17) + 0.3 * torch.cos(t * 0.41)).reshape(shape)


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1] / 'crates/gn-segment/tests/fixtures'
    root.mkdir(parents=True, exist_ok=True)
    metadata = {'torch': torch.__version__, 'torchvision': torchvision.__version__}
    features = signal((1, 3, 19, 23))
    boxes = torch.tensor([
        [2.2, 3.7, 19.1, 16.3],  # Inside, with multiple samples per bin.
        [0, 0, 23, 19],          # Touching image edges.
        [-4, -3, 12, 15],        # Partly outside, including samples below -1.
        [19, 16, 27, 23],        # Past the right and bottom edges.
        [7.2, 8.3, 7.25, 8.35],  # Tiny: unaligned ROI size is clamped to one.
        [-1, -1, -1, -1],        # Edge-clamped negative samples.
        [23, 19, 23, 19],        # Samples beyond size must be zero.
        [-8, -8, -4, -4],        # Entirely outside.
    ], dtype=torch.float32)
    rois = torch.cat((torch.zeros((len(boxes), 1)), boxes), dim=1)
    expected = roi_align(features, rois, (7, 7), spatial_scale=1.0,
                         sampling_ratio=-1, aligned=False)
    save_file({'features': features, 'boxes': boxes, 'expected': expected},
              root / 'roi_align.safetensors', metadata=metadata)
    tensors = {}
    cases = [('down35', (2, 3, 28, 35), (8, 10)),
             ('down4', (1, 2, 32, 40), (8, 10)),
             ('up2', (2, 2, 5, 7), (10, 14)),
             ('nonsquare', (1, 3, 17, 29), (6, 11)),
             ('mixed', (1, 2, 9, 5), (3, 13)),
             ('singleton', (1, 2, 1, 9), (4, 1)),
             ('identity', (1, 2, 3, 7), (3, 7))]
    for name, shape, size in cases:
        x = signal(shape)
        tensors[name + '.input'] = x
        tensors[name + '.expected'] = F.interpolate(x, size=size, mode='bilinear',
                                                    align_corners=False, antialias=True)
    save_file(tensors, root / 'resize.safetensors', metadata=metadata)
    x = signal((2, 16, 5, 7))
    weight = torch.linspace(0.3, 1.4, 16)
    bias = torch.linspace(-0.2, 0.4, 16)
    expected = F.layer_norm(x.permute(0, 2, 3, 1), (16,), weight, bias, 1e-5)
    save_file({'input': x, 'weight': weight, 'bias': bias,
               'expected': expected.permute(0, 3, 1, 2).contiguous()},
              root / 'channel_layer_norm.safetensors', metadata=metadata)
    print(f'Wrote {sum(p.stat().st_size for p in root.glob("*.safetensors"))} bytes to {root}')


if __name__ == '__main__':
    main()
