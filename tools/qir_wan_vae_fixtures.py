"""CPU f32 oracle for the single-frame Wan VAE (no sampling).

Run with ComfyUI's python_embeded/python.exe -I tools/qir_wan_vae_fixtures.py.
Override --comfy and --models for another installation. Outputs are small f32
safetensors in crates/qir/tests/fixtures; neither weights nor PNGs are copied.
"""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, default=Path(
        r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI'))
    parser.add_argument('--models', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    # ComfyUI reads argv during import; force its CPU path as well as VAE dtype.
    sys.argv = [sys.argv[0], '--cpu', '--fp32-vae', '--use-pytorch-cross-attention']
    import torch
    from safetensors.torch import save_file
    import comfy.sd
    import comfy.utils
    from comfy.latent_formats import Wan21

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = args.models or args.comfy / 'models' / 'vae'
    output = Path(__file__).resolve().parents[1] / 'crates/qir/tests/fixtures'
    output.mkdir(parents=True, exist_ok=True)
    y, x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing='ij')
    rgb = torch.stack((x, y, (torch.sin(x * 13 + y * 7) + 1) / 2), dim=-1)[None]
    alpha = ((1 - ((x - .5)**2 + (y - .5)**2) * 3).clamp(0, 1))[None, ..., None]
    latent = (torch.sin(torch.arange(16 * 8 * 8, dtype=torch.float32) * .17) * .7).reshape(1, 16, 1, 8, 8)
    for name, channels in [('qwen_image_layered_vae', 4), ('wan_2.1_vae', 3)]:
        path = models / (name + '.safetensors')
        if not path.is_file():
            print(f'Skipping absent {path}')
            continue
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        vae = comfy.sd.VAE(sd=state, device=torch.device('cpu'), dtype=torch.float32)
        assert vae.output_channels == channels
        fmt = Wan21()
        tensors = {'latent': latent.squeeze(2)}
        with torch.inference_mode():
            for case, picture in [('soft', torch.cat((rgb, alpha), dim=-1) if channels == 4 else rgb),
                                  ('opaque', rgb)]:
                # Exercise the public loader's pixel preparation, including RGB alpha padding.
                mean = vae.encode(picture)
                tensors[case + '.input'] = (picture.movedim(-1, 1) * 2 - 1).contiguous()
                tensors[case + '.mean'] = mean.squeeze(2).contiguous()
                tensors[case + '.encoded'] = fmt.process_in(mean).squeeze(2).contiguous()
            # Raw decoder output is the qir API boundary, before ComfyUI clamps/maps to [0,1].
            tensors['decoded'] = vae.first_stage_model.decode(latent).squeeze(2).contiguous()
        dest = output / (name + '.safetensors')
        save_file(tensors, str(dest), metadata={
            'torch': torch.__version__, 'dtype': 'float32', 'device': 'cpu',
            'checkpoint': str(path), 'decode_space': 'raw VAE latent and raw pixels',
            'encode_space': 'pixels [-1,1], posterior mean, Wan21 normalized mean',
        })
        print(f'Wrote {dest} ({dest.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
