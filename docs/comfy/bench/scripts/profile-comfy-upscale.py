"""Run the installed ComfyUI with node and CUDA phase timing, without editing it."""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
root, work = args.root.resolve(), args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
sys.argv = [str(root / "main.py"), "--windows-standalone-build", "--listen", "127.0.0.1",
            "--port", "8188", "--preview-method", "auto", "--fast", "fp8_matrix_mult"]
for name in ("input", "output", "temp", "user"):
    path = work / name
    path.mkdir(exist_ok=True)
    sys.argv += [f"--{name}-directory", str(path)]

def install_profile():
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import server
    from comfy.ldm.qwen_image21 import model

    log = (work / "comfy-phases.jsonl").open("w", encoding="utf-8", buffering=1)
    pending = []
    last = None


    def record(value):
        log.write(json.dumps(value) + "\n")


    def timed(original, phase):
        def call(*a, **kw):
            # Events preserve asynchronous execution; resolve them at the next node boundary.
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = original(*a, **kw)
            end.record()
            pending.append((phase, begin, end))
            return result
        return call


    original_block = model.QwenImage21TransformerBlock.forward


    def block(self, *a, **kw):
        prefix = kw.get("prefix_len", a[4] if len(a) > 4 else 0)
        if prefix:
            return timed(original_block, "prefix-block")(self, *a, **kw)
        return original_block(self, *a, **kw)


    model.QwenImage21TransformerBlock.forward = block
    model.QwenImage21Transformer2DModel._forward = timed(model.QwenImage21Transformer2DModel._forward, "step-including-prefix")
    original_send = server.PromptServer.send_sync


    def send(self, event, data, sid=None):
        nonlocal last
        if event in ("executing", "execution_success", "execution_error"):
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if last is not None:
                node, started = last
                phases = {}
                for phase, begin, end in pending:
                    phases[phase] = phases.get(phase, 0) + begin.elapsed_time(end)
                record({"node": node, "wallMs": (now - started) * 1000, "cudaMs": phases})
            pending.clear()
            node = data.get("node") if event == "executing" else None
            last = (node, now) if node is not None else None
            record({"event": event, "data": data})
        return original_send(self, event, data, sid)


    server.PromptServer.send_sync = send

source = (root / "main.py").read_text(encoding="utf-8")
marker = "    event_loop, prompt_server, start_all_func = start_comfyui()"
if source.count(marker) != 1:
    raise RuntimeError("installed ComfyUI startup changed; inspect before profiling")
source = source.replace(marker, "    install_profile()\n" + marker)
exec(compile(source, str(root / "main.py"), "exec"), {"__name__": "__main__", "__file__": str(root / "main.py"), "install_profile": install_profile})
