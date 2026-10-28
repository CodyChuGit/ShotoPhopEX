"""Compare RGBA tile artifacts and summarize the warm six-tile phase logs."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
args = parser.parse_args()
results = []
for path in sorted(args.before.glob("*.rgba")):
    other = args.after / path.name
    if not other.exists():
        continue
    a = np.frombuffer(path.read_bytes(), np.uint8).reshape(-1, 4)
    b = np.frombuffer(other.read_bytes(), np.uint8).reshape(-1, 4)
    assert a.shape == b.shape, (path, a.shape, b.shape)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    results.append({"tile": path.name, "identical": bool(np.array_equal(a, b)),
                    "mean_abs_rgb": float(diff[:, :3].mean() / 255),
                    "max_byte": int(diff.max()), "passes": bool(diff[:, :3].mean() <= 1)})
for directory in [args.before, args.after]:
    log = directory.with_suffix(".log")
    if log.exists():
        raw = log.read_bytes()
        text = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8")
        lines = re.findall(r"BENCH .*?run=(\d+).*?seconds=([\d.]+) encode=([\d.]+) denoise=([\d.]+) decode=([\d.]+)", text)
        warm = [list(map(float, row[1:])) for row in lines if int(row[0]) >= 6]
        print(json.dumps({"run": str(directory), "warm_tiles": len(warm), "wall_encode_denoise_decode": np.sum(warm, axis=0).tolist()}))
print(json.dumps(results, indent=2))
if not results or not all(r["passes"] for r in results):
    raise SystemExit(1)
