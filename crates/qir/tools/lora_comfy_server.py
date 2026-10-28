"""Run the installed ComfyUI with node and Qwen-2.1 block CUDA timings."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

root = Path("C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
runtime = Path(__file__).resolve().parents[3] / "target/lora-perf/comfy-runtime"
for name in ("input", "output", "temp", "user"):
    (runtime / name).mkdir(parents=True, exist_ok=True)
os.chdir(root)
sys.path.insert(0, str(root))
sys.argv = [str(root / "main.py"), "--windows-standalone-build", "--listen", "127.0.0.1", "--port", "8188", "--preview-method", "none", "--fast", "fp8_matrix_mult"]
for name in ("input", "output", "temp", "user"):
    sys.argv += ["--" + name + "-directory", str(runtime / name)]
def instrument():
    import torch
    import execution
    import comfy.ldm.qwen_image21.model as qwen
    events = []
    original_forward = qwen.QwenImage21TransformerBlock.forward

    def timed_forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = original_forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options)
        end.record()
        events.append(("prefix" if prefix_len == x.shape[1] else "target", start, end))
        return output

    qwen.QwenImage21TransformerBlock.forward = timed_forward
    original_map = execution._async_map_node_over_list

    async def timed_map(*args, **kwargs):
        obj, func = (args[2], args[4]) if len(args) >= 5 else (None, None)
        if obj is None or func != getattr(obj, "FUNCTION", None):
            return await original_map(*args, **kwargs)
        torch.cuda.synchronize()
        events.clear()
        started = time.perf_counter()
        output = await original_map(*args, **kwargs)
        torch.cuda.synchronize()
        record = {"class": type(obj).__name__, "node": args[1], "seconds": time.perf_counter() - started,
                  "allocated_mib": torch.cuda.memory_allocated() / 2**20, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        for name in ("prefix", "target"):
            record[name + "_seconds"] = sum(a.elapsed_time(b) / 1000 for label, a, b in events if label == name)
        print("LORA_PHASE " + json.dumps(record), flush=True)
        return output

    execution._async_map_node_over_list = timed_map

def startup_hook(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "start_comfyui":
        sys.setprofile(None)
        instrument()

# Keep ComfyUI's allocator-hook and CLI initialization order intact.
sys.setprofile(startup_hook)
runpy.run_path(str(root / "main.py"), run_name="__main__")
