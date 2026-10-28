"""Compare the benchmark's raw RGBA artifacts, failing above 1/255 mean error."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
args = parser.parse_args()
results = []
for path in sorted(args.after.glob("*.rgba")):
    before = (args.before / path.name).read_bytes()
    after = path.read_bytes()
    if not before or len(before) != len(after):
        raise ValueError(f"{path.name}: mismatched or empty images")
    error = sum(abs(a - b) for a, b in zip(before, after)) / len(before)
    results.append({"image": path.name, "identical": before == after, "mae_bytes": error, "mae": error / 255, "pass": error <= 1})
print(json.dumps(results, indent=2))
if not results or not all(row["pass"] for row in results):
    raise SystemExit(1)
