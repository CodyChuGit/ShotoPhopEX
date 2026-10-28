"""CPU-only ComfyUI/worker cup investigation. Run with embedded Python -I.

run executes sequential isolated processes (ComfyUI, then Rust 0/2 passes).
compare regenerates metrics/PNGs from saved masks. No model/source edits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
COMFY = Path(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI")
IMAGE = Path(r"C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-cup.jpg")
CKPT = COMFY / "models/checkpoints/sam3.1_multiplex_fp16.safetensors"
OUT = ROOT / "target/sam31-parity/cup"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"


def save_mask(path, array):
    import numpy as np
    from PIL import Image
    a = np.asarray(array, dtype=np.uint8).squeeze()
    path.parent.mkdir(parents=True, exist_ok=True)
    a.tofile(path.with_suffix(".bin"))
    Image.fromarray(a * 255).save(path.with_suffix(".png"))


def comfy_run():
    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0], "--cpu", "--use-pytorch-cross-attention"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import numpy as np
    from PIL import Image
    import torch
    import comfy.sd
    import comfy.model_management as mm
    from comfy_extras import nodes_sam3 as nodes
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assert mm.get_torch_device().type == "cpu"
    model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        str(CKPT), output_vae=False, output_clip=True,
        model_options={"dtype": torch.float32},
        te_model_options={"dtype": torch.float32})
    sam = model.model.diffusion_model.eval()
    assert model.model.get_dtype() == torch.float32
    pixels = np.array(Image.open(IMAGE).convert("RGB"))
    if os.environ.get("GN_CUP_NATIVE_PIXELS"):
        pixels = np.fromfile(ROOT / "target/sam31-parity/cup/native.rgba", dtype=np.uint8).reshape(1280, 1920, 4)[..., :3].copy()
    image = torch.from_numpy(pixels.astype(np.float32) / 255).unsqueeze(0)
    meta = {"torch": torch.__version__, "device": str(mm.get_torch_device()),
            "dtype": str(model.model.get_dtype()), "image": str(IMAGE), "prompt": "coffee cup",
            "threshold": 0.5, "max_detections": 1, "refine_iterations": 2,
            "pixels": "Rust image crate" if os.environ.get("GN_CUP_NATIVE_PIXELS") else "Pillow"}
    original_refine = nodes._refine_mask
    original_segment = sam.forward_segment
    steps = []

    def refine(*args, **kwargs):
        coarse = args[2]
        np.save(OUT / "comfy-coarse-logits.npy", coarse.detach().numpy())
        zero = list(args)
        zero[-1] = 0
        coarse_binary = original_refine(*zero)
        save_mask(OUT / "comfy-0", coarse_binary.detach().numpy())
        meta["bbox_xyxy"] = args[3].tolist()
        return original_refine(*args, **kwargs)

    def segment(*args, **kwargs):
        result = original_segment(*args, **kwargs)
        steps.append(result.detach().numpy().copy())
        np.save(OUT / f"comfy-refinement-{len(steps)}-logits.npy", steps[-1])
        print(f"ComfyUI refinement {len(steps)} complete", flush=True)
        return result

    nodes._refine_mask = refine
    sam.forward_segment = segment
    with torch.inference_mode():
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("coffee cup"))
        result = nodes.SAM3_Detect.execute(model, image, conditioning=conditioning,
                                         threshold=0.5, refine_iterations=2)
        save_mask(OUT / "comfy-2", result[0].numpy())
        meta["detections"] = result[1]
    (OUT / "comfy.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def monitored(command, name, env=None):
    import psutil
    log = OUT / f"{name}.log"
    peak_rss = peak_private = 0
    start = time.monotonic()
    with log.open("w") as f:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                processes = [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True)
                infos = [q.memory_info() for q in processes]
                rss = sum(i.rss for i in infos)
                private = sum(getattr(i, "private", i.vms) for i in infos)
                peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
                if max(rss, private) > 14.5 * 10**9:
                    for q in reversed(processes):
                        q.kill()
                    raise RuntimeError(f"{name} exceeded 14.5 GB memory guard")
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.1)
    stats = {"command": command, "returncode": p.returncode,
             "seconds": time.monotonic() - start, "peak_rss_bytes": peak_rss,
             "peak_private_bytes": peak_private}
    (OUT / f"{name}-process.json").write_text(json.dumps(stats, indent=2))
    print(name, stats, flush=True)
    if p.returncode:
        raise RuntimeError(log.read_text(errors="replace")[-6000:])


def compare():
    import numpy as np
    from PIL import Image, ImageDraw
    w, h = Image.open(IMAGE).size
    metrics = {}
    for n in (0, 2):
        paths = {"comfy": OUT / f"comfy-{n}.bin"}
        if (OUT / f"native-input/comfy-{n}.bin").exists():
            paths["comfy_native"] = OUT / f"native-input/comfy-{n}.bin"
        paths.update({b: OUT / f"iter{n}/{b}/st-cup-text-0.bin" for b in ("candle", "onnx")})
        masks = {b: np.fromfile(p, dtype=np.uint8).reshape(h, w).astype(bool) for b, p in paths.items()}
        uy, ux = np.where(np.logical_or.reduce(list(masks.values())))
        view = (max(0, int(ux.min()) - 16), max(0, int(uy.min()) - 16),
                min(w, int(ux.max()) + 17), min(h, int(uy.max()) + 17))
        for b, m in masks.items():
            save_mask(OUT / f"{b}-{n}", m)
        panels = []
        pairs = [("comfy", "candle"), ("comfy", "onnx"), ("candle", "onnx")]
        if "comfy_native" in masks:
            pairs += [("comfy_native", "candle"), ("comfy_native", "onnx"), ("comfy_native", "comfy")]
        for a, b in pairs:
            x, y = masks[a], masks[b]
            diff = x ^ y
            ys, xs = np.where(diff)
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
            metrics[f"{n}:{a}/{b}"] = {"iou": int((x & y).sum()) / int((x | y).sum()),
                "disagreement_pixels": int(diff.sum()), "bounds_xyxy_exclusive": bounds,
                "a_only": int((x & ~y).sum()), "b_only": int((y & ~x).sum())}
            rgb = np.array(Image.open(IMAGE).convert("RGB")) // 3
            rgb[x & ~y] = (255, 60, 60)
            rgb[y & ~x] = (0, 220, 255)
            panel = Image.fromarray(rgb).crop(view)
            panel.thumbnail((480, 320))
            canvas = Image.new("RGB", (480, 345), "white")
            canvas.paste(panel, (0, 25))
            ImageDraw.Draw(canvas).text((5, 5), f"{a} red / {b} cyan; iterations={n}", fill="black")
            panels.append(canvas)
        sheet = Image.new("RGB", (1440, 345 * ((len(panels) + 2) // 3)))
        for i, panel in enumerate(panels):
            sheet.paste(panel, ((i % 3) * 480, (i // 3) * 345))
        sheet.save(OUT / f"disagreement-{n}.png")
    for b in ("comfy", "candle", "onnx"):
        a = np.fromfile(OUT / f"{b}-0.bin", dtype=np.uint8).astype(bool)
        z = np.fromfile(OUT / f"{b}-2.bin", dtype=np.uint8).astype(bool)
        metrics[f"refinement:{b}"] = {"added": int((z & ~a).sum()), "removed": int((a & ~z).sum())}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def audit():
    import collections
    import hashlib
    import onnx
    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    graphs = {}
    for p in Path(r"C:\Users\5090\Documents\_eval\native-models\sam3.1").glob("*.onnx"):
        m = onnx.load(p, load_external_data=False)
        casts = collections.Counter(onnx.TensorProto.DataType.Name(a.i)
                                    for n in m.graph.node if n.op_type == "Cast"
                                    for a in n.attribute if a.name == "to")
        graphs[p.name] = {"sha256": digest(p), "initializer_dtypes": dict(collections.Counter(
            onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer)),
            "cast_targets": dict(casts)}
        if p.with_suffix(p.suffix + ".data").exists():
            graphs[p.name]["external_data_sha256"] = digest(p.with_suffix(p.suffix + ".data"))
    data = {"graphs": graphs, "checkpoint_sha256": digest(CKPT), "image_sha256": digest(IMAGE),
            "genai_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "comfy_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=COMFY, text=True).strip()}
    (OUT / "provenance.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


def native_pixels():
    """Compile a tiny decoder from stdin against the already-built image crate.

    Generated executable/raw bytes stay under target; no Rust source file is added.
    """
    import numpy as np
    from PIL import Image
    deps = ROOT / "target/release/deps"
    image_rlib = next(deps.glob("libimage-*.rlib"))
    exe = OUT / "decode-native.exe"
    source = 'fn main() { let a: Vec<String> = std::env::args().collect(); let im = image::open(&a[1]).unwrap().to_rgba8(); std::fs::write(&a[2], im.as_raw()).unwrap(); }'
    subprocess.run(["rustc", "--edition=2024", "-O", "-L", f"dependency={deps}",
                    "--extern", f"image={image_rlib}", "-o", str(exe), "-"],
                   input=source, text=True, check=True, cwd=ROOT)
    subprocess.run([str(exe), str(IMAGE), str(OUT / "native.rgba")], check=True)
    a = np.fromfile(OUT / "native.rgba", dtype=np.uint8).reshape(1280, 1920, 4)[..., :3]
    b = np.array(Image.open(IMAGE).convert("RGB"))
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    (OUT / "jpeg-difference.json").write_text(json.dumps({"max_abs_u8": int(d.max()),
        "mean_abs_u8": float(d.mean()), "different_channels": int((d != 0).sum()),
        "different_pixels": int((d != 0).any(axis=-1).sum())}, indent=2))
    env = dict(os.environ, GN_CUP_NATIVE_PIXELS="1")
    monitored([sys.executable, "-I", str(Path(__file__).resolve()), "comfy-native"], "comfy-native", env)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("comfy", "run", "compare", "audit", "native", "comfy-native"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "comfy-native":
        OUT = OUT / "native-input"
        OUT.mkdir(parents=True, exist_ok=True)
        comfy_run()
    elif args.mode == "native":
        native_pixels()
    elif args.mode == "comfy":
        comfy_run()
    elif args.mode == "compare":
        compare()
    elif args.mode == "audit":
        audit()
    else:
        audit()
        monitored([sys.executable, "-I", str(Path(__file__).resolve()), "comfy"], "comfy")
        exe = ROOT / "target/release/examples/sam31_parity.exe"
        for n in (0, 2):
            for backend in ("onnx", "candle"):
                env = dict(os.environ, GN_SAM31_PICTURE="st-cup", GN_SAM31_ITERATIONS=str(n),
                           GN_SAM31_OUTPUT_DIR=str(OUT / f"iter{n}"))
                monitored([str(exe), backend, "1"], f"{backend}-{n}", env)
        native_pixels()
        compare()
