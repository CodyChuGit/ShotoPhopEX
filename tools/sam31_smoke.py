"""Load ComfyUI's SAM 3.1 the way the app's reference run does and time its three paths.

Run with ComfyUI's own Python, from anywhere:

    python -s tools/sam31_smoke.py --comfy <ComfyUI dir> --image <png> [--text fox] [--point x,y]

It prints the module tree sizes, runs the SAM3_Detect node for a text prompt and a point prompt,
and reports mask bounds and timings. This is the ground truth the ONNX export must reproduce.
"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="sam3.1_multiplex_fp16.safetensors")
    ap.add_argument("--text", default="fox")
    ap.add_argument("--point", default="481,522")
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]  # ComfyUI parses argv at import
    sys.path.insert(0, args.comfy)
    import torch
    from PIL import Image
    import numpy as np
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_sam3 import SAM3_Detect

    ckpt = folder_paths.get_full_path_or_raise("checkpoints", args.ckpt)
    t0 = time.perf_counter()
    model, clip, _vae, _cv = comfy.sd.load_checkpoint_guess_config(ckpt, output_vae=False, output_clip=True, embedding_directory=None)
    print(f"loaded {ckpt} in {time.perf_counter() - t0:.1f} s; model {type(model.model).__name__}, dtype {model.model.get_dtype()}")
    sam3 = model.model.diffusion_model
    n_det = sum(p.numel() for p in sam3.detector.parameters())
    n_trk = sum(p.numel() for p in sam3.tracker.parameters())
    print(f"detector {n_det / 1e6:.1f} M params ({type(sam3.detector).__name__}), tracker {n_trk / 1e6:.1f} M ({type(sam3.tracker).__name__})")
    bb = sam3.detector.backbone["vision_backbone"]
    print(f"backbone multiplex={bb.multiplex}; detector convs {len(bb.convs)}; scalp {sam3.detector.scalp}")
    for name, mod in [("trunk", bb.trunk), ("transformer", sam3.detector.transformer), ("geometry_encoder", sam3.detector.geometry_encoder), ("segmentation_head", sam3.detector.segmentation_head)]:
        print(f"  detector.{name}: {sum(p.numel() for p in mod.parameters()) / 1e6:.1f} M")
    for name in ["sam_prompt_encoder", "sam_mask_decoder", "memory_encoder", "memory_attention", "obj_ptr_proj"]:
        mod = getattr(sam3.tracker, name, None)
        if mod is not None:
            print(f"  tracker.{name}: {sum(p.numel() for p in mod.parameters()) / 1e6:.1f} M ({type(mod).__name__})")
    print("  tracker attrs:", [k for k in vars(sam3.tracker) if not k.startswith("_")][:30])

    im = Image.open(args.image).convert("RGB")
    arr = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)  # [1,H,W,3]
    H, W = arr.shape[1], arr.shape[2]

    tokens = clip.tokenize(args.text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    print(f"text cond {tuple(cond[0][0].shape)} mask {None if cond[0][1].get('attention_mask') is None else tuple(cond[0][1]['attention_mask'].shape)}")

    def bounds(mask):
        m = mask > 0.5
        ys, xs = torch.where(m)
        if len(xs) == 0:
            return None, 0
        return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], int(m.sum())

    for label, kwargs in [
        ("text", dict(conditioning=cond)),
        ("point", dict(positive_coords=json.dumps([{"x": int(args.point.split(",")[0]), "y": int(args.point.split(",")[1])}]))),
    ]:
        for rep in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = SAM3_Detect.execute(model, arr, threshold=0.5, refine_iterations=2, individual_masks=True, **kwargs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            masks = out.args[0]
            b, n = bounds(masks[0]) if masks.shape[0] else (None, 0)
            print(f"{label} run {rep + 1}: {dt * 1000:.0f} ms, {masks.shape[0]} mask(s), bounds {b} pixels {n}, boxes {out.args[1]}")


if __name__ == "__main__":
    main()
