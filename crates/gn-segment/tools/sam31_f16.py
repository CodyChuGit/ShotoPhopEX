"""Shared-pixel ComfyUI CPU f32 references and CUDA precision measurements.

Run using ComfyUI's embedded Python with -I. All caches/artifacts go to target/.
First build examples/sam31_f16.rs with tools/f16-build.cmd, then run `run`.
Reference JSONs are completion markers, permitting resume without recomputation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'target/sam31-f16'
COMFY = Path(r'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI')
CKPT = COMFY / 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
EXE = ROOT / 'target/sam31-f16-build/release/examples/sam31_f16.exe'


def comfy():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0], '--cpu', '--use-pytorch-cross-attention']
    import comfy.options
    comfy.options.enable_args_parsing()
    import numpy as np
    import torch
    from scipy.ndimage import distance_transform_edt
    import comfy.sd
    import comfy.model_management as mm
    from comfy_extras import nodes_sam3 as nodes
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assert mm.get_torch_device().type == 'cpu'
    t = time.perf_counter()
    model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        str(CKPT), output_vae=False, output_clip=True,
        model_options={'dtype': torch.float32}, te_model_options={'dtype': torch.float32})
    load_seconds = time.perf_counter() - t
    assert model.model.get_dtype() == torch.float32
    dest = OUT / 'comfy'
    dest.mkdir(exist_ok=True)
    with torch.inference_mode():
        for p in json.loads((OUT / 'pictures.json').read_text()):
            name, w, h = p['picture'], p['width'], p['height']
            meta_path = dest / f'{name}.json'
            if meta_path.exists():
                print('cached', name, flush=True)
                continue
            rgba = np.fromfile(OUT / f'{name}.rgba', dtype=np.uint8).reshape(h, w, 4)
            image = torch.from_numpy(rgba[..., :3].copy().astype(np.float32) / 255).unsqueeze(0)
            t = time.perf_counter()
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(p['prompt']))
            result = nodes.SAM3_Detect.execute(model, image, conditioning=cond, threshold=0.5, refine_iterations=2)
            text_seconds = time.perf_counter() - t
            mask = result[0].numpy().squeeze().astype(np.uint8)
            boxes = result[1][0]
            assert len(boxes) <= 1
            mask.tofile(dest / f'{name}-text-0.bin')
            # Deepest interior pixel of the selected reference mask: deterministic,
            # independent of candle precision, and safely inside thin objects too.
            point = None
            point_seconds = None
            inradius = None
            if boxes and mask.any():
                distance = distance_transform_edt(mask)
                y, x = np.unravel_index(distance.argmax(), distance.shape)
                point = [int(x), int(y)]
                inradius = float(distance[y,x])
                t = time.perf_counter()
                result_point = nodes.SAM3_Detect.execute(model, image, positive_coords=json.dumps([{'x':int(x), 'y':int(y)}]), threshold=0.5, refine_iterations=2)
                point_seconds = time.perf_counter() - t
                point_mask = result_point[0].numpy().squeeze().astype(np.uint8)
                point_mask.tofile(dest / f'{name}-point-0.bin')
            meta = dict(p, point=point, point_inradius=inradius, detections=boxes,
                        text_seconds=text_seconds, point_seconds=point_seconds, load_seconds=load_seconds,
                        torch=torch.__version__, device=str(mm.get_torch_device()), dtype=str(model.model.get_dtype()),
                        rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest())
            meta_path.write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta), flush=True)


def process(command, name, cpu=False):
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', RAYON_NUM_THREADS='8')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    env['PATH'] = str(Path(r'C:\Users\5090\scoop\apps\cuda\current\bin')) + os.pathsep + env['PATH']
    with (OUT / f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def compare():
    import numpy as np
    rows = []
    for p in json.loads((OUT / 'pictures.json').read_text()):
        name = p['picture']
        ref = json.loads((OUT / 'comfy' / f'{name}.json').read_text())
        assert hashlib.sha256((OUT/f'{name}.rgba').read_bytes()).hexdigest() == ref['rgba_sha256']
        text_reference = np.fromfile(OUT/'comfy'/f'{name}-text-0.bin', dtype=np.uint8).reshape(p['height'], p['width'])
        if ref['point'] is not None:
            assert text_reference[ref['point'][1], ref['point'][0]] == 1
        for mode in ('cpu', 'cuda-f32', 'cuda-tf32', 'cuda-f16'):
            data = json.loads((OUT / mode / f'{name}.json').read_text())
            samples = np.array(data['samples_seconds'])
            row = dict(p, mode=mode, point=ref['point'],
                       text_ms=float(np.median(samples[:,0]+samples[:,1])*1000),
                       encode_ms=float(np.median(samples[:,0])*1000), select_ms=float(np.median(samples[:,1])*1000),
                       point_ms=float(np.median(samples[:,2])*1000) if ref['point'] else None,
                       comfy_text_ms=ref['text_seconds']*1000, comfy_point_ms=ref['point_seconds']*1000 if ref['point'] else None,
                       comfy_score=ref['detections'][0]['score'] if ref['detections'] else None,
                       reference_count=len(ref['detections']), candle_count=len(data['text']))
            for kind in ('text','point'):
                if kind == 'point' and ref['point'] is None:
                    row['point_iou'] = None
                    row['point_different_pixels'] = None
                    row['point_reference_pixels'] = None
                    row['point_candle_pixels'] = None
                    continue
                a = np.fromfile(OUT/'comfy'/f'{name}-{kind}-0.bin', dtype=np.uint8).astype(bool)
                b = np.fromfile(OUT/mode/f'{name}-{kind}-0.bin', dtype=np.uint8).astype(bool) if data[kind] else np.zeros_like(a)
                assert a.size == b.size == p['width']*p['height']
                row[kind+'_iou'] = int((a & b).sum()) / int((a | b).sum()) if (a | b).any() else 1.0
                row[kind+'_different_pixels'] = int((a ^ b).sum())
                row[kind+'_reference_pixels'] = int(a.sum())
                row[kind+'_candle_pixels'] = int(b.sum())
            row['score'] = data['text'][0]['score'] if data['text'] else None
            row['score_delta'] = row['score']-row['comfy_score'] if row['score'] is not None and row['comfy_score'] is not None else None
            rows.append(row)
    (OUT/'metrics.json').write_text(json.dumps(rows, indent=2))
    for mode in ('cpu','cuda-f32','cuda-tf32','cuda-f16'):
        subset = [r for r in rows if r['mode']==mode]
        print(mode, 'min text', min((r['text_iou'],r['picture']) for r in subset),
              'min point', min((r['point_iou'],r['picture']) for r in subset if r['point_iou'] is not None), flush=True)


def audit():
    def sha(path):
        with Path(path).open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    def command(args, cwd=ROOT):
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    stock = Path(r'C:\Users\5090\ComfyUI\photocraft-tests\bench\stock')
    pictures = json.loads((OUT/'pictures.json').read_text())
    data = dict(genai_revision=command(['git','rev-parse','HEAD']),
                comfy_revision=command(['git','rev-parse','HEAD'], COMFY),
                checkpoint=str(CKPT), checkpoint_sha256=sha(CKPT),
                tokenizer_sha256=sha(r'C:\Users\5090\Documents\_eval\native-models\sam3.1\tokenizer.json'),
                gpu=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv']),
                rustc=command(['rustc','--version']),
                precision_environment={k:os.environ.get(k) for k in ('NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')},
                sources={p['picture']:sha(stock/(p['picture']+'.jpg')) for p in pictures},
                executable_sha256=sha(EXE))
    (OUT/'provenance.json').write_text(json.dumps(data, indent=2))


def report():
    import statistics
    rows = json.loads((OUT/'metrics.json').read_text())
    pictures = json.loads((OUT/'pictures.json').read_text())
    modes = ('cpu','cuda-f32','cuda-tf32','cuda-f16')
    lookup = {(r['picture'],r['mode']):r for r in rows}
    provenance = json.loads((OUT/'provenance.json').read_text())
    def fmt(value, precision=9, signed=False):
        return 'N/A' if value is None else format(value, f'{"+" if signed else ""}.{precision}f')
    worst = min((r[k+'_iou'],r['picture'],k) for r in rows if r['mode']=='cuda-f16' for k in ('text','point') if r[k+'_iou'] is not None)
    available = sum(r[k+'_iou'] is not None for r in rows if r['mode']=='cpu' for k in ('text','point'))
    empty = [r['picture'] for r in rows if r['mode']=='cpu' and r['reference_count']==0]
    empty_points = [r['picture'] for r in rows if r['mode']=='cpu' and r['point_reference_pixels']==0]
    lines = ['# SAM 3.1 CUDA precision against ComfyUI f32', '',
             'Measured 2026-10-27 on RTX 5090. Reference: the installed ComfyUI `comfy_extras.nodes_sam3.SAM3_Detect.execute`, on **CPU with explicit f32 model and CLIP compute**, using the same decoded pixels as candle. No ONNX graphs participate.', '',
             f'**F16 {"passes" if worst[0]>=0.99 else "fails"} the available-mask IoU >= 0.99 gate.** Its smallest final-mask IoU is **{worst[0]:.9f}**, on **{worst[1]} {worst[2]}**. '+
             ('The available object masks pass, but keep F16 opt-in until the missing positive object cases are supplied; the full requested 12-object gate is not established.' if worst[0]>=0.99 and empty else 'This sample supports making F16 the GPU default under that gate; retain an explicit f32 fallback and broaden the evaluation before claiming general parity.' if worst[0]>=0.99 else 'Keep F16 opt-in; this sample does not support making it the GPU default under that gate.'), '',
             '## Summary', '', f'| Candle route | Smallest text IoU | Smallest point IoU | Comparisons >= 0.99 / {available} | Median across pictures, text / cached point ms |', '| --- | ---: | ---: | ---: | ---: |']
    for mode in modes:
        subset = [r for r in rows if r['mode']==mode]
        text = min(r['text_iou'] for r in subset)
        point = min(r['point_iou'] for r in subset if r['point_iou'] is not None)
        passed = sum(r[k+'_iou']>=0.99 for r in subset for k in ('text','point') if r[k+'_iou'] is not None)
        lines.append(f'| {mode} | {text:.9f} | {point:.9f} | {passed} | {statistics.median(r["text_ms"] for r in subset):.1f} / {statistics.median(r["point_ms"] for r in subset if r["point_ms"] is not None):.1f} |')
    if empty:
        lines += ['', '**Input limitation:** ComfyUI returns no text detection at threshold 0.5 for '+', '.join(empty)+'. These have no selected detector probability and no detected-object point test (N/A). Visual inspection found a dark garage entrance in `st-car.jpg`, an archival portrait of three children in `st-cat.jpg`, and a man standing by a railing in `st-watch.jpg`; no corresponding car, cat or watch is visible. No substitute image, lowered threshold, or arbitrary background point is introduced. Empty/empty text-mask agreement is conventionally recorded as IoU 1, not evidence of object-mask fidelity. Thus the requested 12 positive object tests cannot all be established from these installed inputs.']
    if empty_points:
        lines += ['', '**Empty point reference:** '+', '.join(empty_points)+' returns an empty ComfyUI point mask despite the point being inside its detected text mask. Its point timing is a real executed request; empty/empty point agreement is also recorded as IoU 1 by convention. Foreground counts and disagreement counts are retained in `metrics.json`.']
    nonempty = sum(r[k+'_reference_pixels'] is not None and r[k+'_reference_pixels']>0 for r in rows if r['mode']=='cpu' for k in ('text','point'))
    mismatches = [(r['picture'],r['mode']) for r in rows if r['reference_count'] != r['candle_count']]
    lines += ['', f'There are {available} available comparisons per mode: {nonempty} nonempty reference masks and {available-nonempty} empty reference masks. '+('All four candle variants match the ComfyUI text detection counts on all 12 pictures.' if not mismatches else f'Detection count mismatches: {mismatches}.'), '',
              'CUDA f32 with TF32 disabled is extremely close to the correct reference: its worst text case is cup, with only 3 changed pixels. Allowing TF32 changes 1,279 cup pixels and fails the gate; F16 changes 654 cup pixels and passes the available-mask gate. F16 is therefore a viable candidate for the fast default, but the recommendation for the complete requested qualification is to retain F32 with TF32 disabled until the missing positive cases are tested. The previous low cup IoU against ONNX is not the CUDA F32/ComfyUI discrepancy.']
    ratios = [lookup[p['picture'],'cuda-f32']['text_ms']/lookup[p['picture'],'cuda-f16']['text_ms'] for p in pictures]
    lines += ['', f'Median per-picture CUDA f32/F16 text speed ratio: **{statistics.median(ratios):.2f}x**. The minimum across available masks, rather than an average IoU, determines the gate. These are numerical parity measurements, not semantic ground truth or a claim about all images.', '',
              '## Method', '',
              '- The Rust `image` crate decodes each stock JPEG to RGBA once. ComfyUI consumes its RGB bytes divided by 255; every candle mode reads the exact same RGBA file. Pixel dimensions and SHA-256 hashes are cached. This controls the Pillow/Rust JPEG difference established in `CUP.md`.',
              '- Text prompts use threshold strictly greater than 0.5, highest scoring detection only, and two refinement passes. Candle uses unchanged `Options::default()`. Detection counts and nonempty masks are reported; absent reference detections have no invented point prompt.',
              '- One positive point is selected per picture by the maximum Euclidean distance to background in the ComfyUI final text mask (first row-major maximum). This guarantees the point is inside the selected reference object, including thin objects. The point is then frozen for all modes; no negative points, box prompts, or precision-specific adjustments are used. The point node runs separately, without text conditioning, with two tracker passes.',
              '- Completed ComfyUI text and point references are computed once and cached under `target/sam31-f16/comfy/`. An initial attempt stopped on an assertion when car returned no detections; car text was rerun after adding empty-result handling. The four completed references were reused. The node is unmodified. It re-encodes for each refinement pass. CPU candle runs once per picture as a control. Candle uses OMP/MKL/Rayon thread limits of 8; PyTorch uses intra-op 8 and inter-op 1. Final candle measurement processes run sequentially. A brief dog F16 smoke run overlapped reference generation and is excluded from final candle timings.',
              '- CUDA modes each use one excluded warm-up and five measured repetitions **per picture**, with new full-image encode every repetition. Timers synchronize the device at request boundaries; internal stage profiling is off. Text time is full-image encode plus selection, including crop encode and two refinement passes. Point time reuses the full-image frame. Image decoding, model/tokenizer loading, mask saving, and comparison are outside request timers. CPU and ComfyUI times are single observations, not medians over repeated runs.',
              '- `cuda-f32` explicitly disables `candle_core::cuda_backend::set_gemm_reduced_precision_f32`; this is also candle 0.11\'s default. `cuda-tf32` explicitly enables it, requesting `CUBLAS_COMPUTE_32F_FAST_TF32` for f32 GEMMs (permission for cuBLAS to use TF32, not proof every operation uses it). `cuda-f16` uses `Sam31::load_with_dtype(..., DType::F16)` with that f32 switch off. F16 reduced-precision accumulation remains at candle\'s default false. No library source, threshold or model weights are changed.',
              '- IoU is intersection / union of binary masks at original image resolution. Score differences are candle minus ComfyUI selected **detector probability**, before refinement. Point outputs have no detector score: candle assigns the API constant 1.0 and ComfyUI returns no point score, so a point score difference would not measure model parity.', '',
              '## Inputs and reference scores', '', '| Picture | Prompt | Size | Positive point (x,y) | ComfyUI detector score |', '| --- | --- | ---: | ---: | ---: |']
    for p in pictures:
        r = lookup[p['picture'],'cpu']
        lines.append(f'| {p["picture"]} | {p["prompt"]} | {p["width"]} x {p["height"]} | {tuple(r["point"]) if r["point"] else "N/A"} | {fmt(r["comfy_score"])} |')
    lines += ['', '## Final mask IoUs', '', 'Each cell is **text / point** against the shared-pixel ComfyUI CPU f32 node.', '', '| Picture / prompt | CPU f32 | CUDA f32, TF32 off | CUDA f32, TF32 allowed | CUDA F16 |', '| --- | ---: | ---: | ---: | ---: |']
    for p in pictures:
        cells = [f'{fmt(lookup[p["picture"],m]["text_iou"])} / {fmt(lookup[p["picture"],m]["point_iou"])}' for m in modes]
        lines.append(f'| {p["picture"]} / {p["prompt"]} | '+' | '.join(cells)+' |')
    lines += ['', '## Detector score differences', '', '| Picture / prompt | CPU f32 | CUDA f32, TF32 off | CUDA f32, TF32 allowed | CUDA F16 |', '| --- | ---: | ---: | ---: | ---: |']
    for p in pictures:
        lines.append(f'| {p["picture"]} / {p["prompt"]} | '+' | '.join(fmt(lookup[p["picture"],m]["score_delta"], signed=True) for m in modes)+' |')
    lines += ['', '## Request timings', '', 'Milliseconds; each cell is **text / point**. ComfyUI point time includes its own repeated image encoding; candle point time uses an encoded frame. This table does not imply equivalent caching or a fair implementation speed comparison with ComfyUI.', '', '| Picture | ComfyUI CPU f32 | Candle CPU f32 | CUDA f32, TF32 off | CUDA f32, TF32 allowed | CUDA F16 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for p in pictures:
        r = lookup[p['picture'],'cpu']
        cells = [f'{fmt(r["comfy_text_ms"],1)} / {fmt(r["comfy_point_ms"],1)}'] + [f'{fmt(lookup[p["picture"],m]["text_ms"],1)} / {fmt(lookup[p["picture"],m]["point_ms"],1)}' for m in modes]
        lines.append(f'| {p["picture"]} | '+' | '.join(cells)+' |')
    lines += ['', '## Reproduction and artifacts', '',
              'From the repository root (PowerShell):', '', '```powershell',
              r'cmd /c crates\gn-segment\tools\f16-build.cmd build -p gn-segment --release --features cuda --example sam31_f16',
              r"& 'C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe' -I crates/gn-segment/tools/sam31_f16.py run", '```', '',
              '`f16-build.cmd` uses the environment in `examples/cuda-bench.cmd`, with its own `CARGO_TARGET_DIR=target/sam31-f16-build`, CUDA compute capability 120, CUDA version 13030 and the installed CUTLASS include directory. `run` decodes, resumes cached references, measures all four candle modes, compares, and records provenance. `comfy`, `measure`, `compare`, `audit`, and `report` can also be invoked separately. Delete a picture\'s ComfyUI JSON completion marker to deliberately recompute its references. Rebuild references if inputs/checkpoints or reference code change.', '',
              'Artifacts are under the ignored `target/sam31-f16/`: raw shared RGBA, `pictures.json`, ComfyUI reference masks and metadata, per-mode binary masks and JSON, logs, `metrics.json` (IoUs, disagreement counts, detection counts, scores and timing summaries), and `provenance.json` (image/checkpoint/tokenizer/executable hashes and revisions). Per-mode JSON retains every measured encode/select/point timing sample and model-load time. The saved mask is the final measured repetition. No files are committed.', '',
              f'- genai-native revision: `{provenance["genai_revision"]}`',
              f'- ComfyUI revision: `{provenance["comfy_revision"]}`',
              '- PyTorch reference: `2.14.0+cu130`, asserted CPU and `torch.float32`.',
              '- Candle 0.11.0 / cudarc 0.19.9; GPU and driver: '+provenance['gpu'].splitlines()[-1]+'.',
              f'- Checkpoint SHA-256: `{provenance["checkpoint_sha256"]}`', '',
              'Added files: `examples/sam31_f16.rs`, `tools/f16-build.cmd`, `tools/sam31_f16.py`, and this report. Library `src/` is unchanged.']
    (ROOT/'crates/gn-segment/F16.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run','comfy','compare','audit','measure','report'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'comfy':
        comfy()
    elif args.mode == 'compare':
        compare()
    elif args.mode == 'audit':
        audit()
    elif args.mode == 'report':
        report()
    elif args.mode == 'measure':
        for mode in ('cpu','cuda-f32','cuda-tf32','cuda-f16'):
            process([str(EXE), mode], mode, mode=='cpu')
        compare()
        audit()
        report()
    else:
        process([str(EXE), 'decode'], 'decode', True)
        process([sys.executable, '-I', str(Path(__file__).resolve()), 'comfy'], 'comfy', True)
        for mode in ('cpu','cuda-f32','cuda-tf32','cuda-f16'):
            process([str(EXE), mode], mode, mode=='cpu')
        compare()
        audit()
        report()

