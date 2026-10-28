"""Run one native GPU process and sample total board memory once per second."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--vram", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[0] == "--" else args.command
command[0] = str(Path(command[0]).resolve())
samples = []
with args.log.open("wb") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    while child.poll() is None:
        sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if sample.returncode == 0:
            samples.append([time.time()] + [float(v.strip()) for v in sample.stdout.strip().split(",")])
        time.sleep(1)
args.vram.write_text(json.dumps({"exit_code": child.returncode, "peak_board_mib": max((s[1] for s in samples), default=0), "samples": samples}, indent=2))
raise SystemExit(child.returncode)
