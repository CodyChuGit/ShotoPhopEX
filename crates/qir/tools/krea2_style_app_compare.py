"""Full-size GPU comparison through PhotoshopEX; run only when the GPU is available.
The app must be rebuilt against this worktree. ComfyUI must already be serving the models.
Uses the same request, references, seed and disabled prompt enhancement on both app routes.
"""
import argparse, json, subprocess
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--app", type=Path, required=True)
p.add_argument("--models", type=Path, required=True)
p.add_argument("--runtime", required=True)
p.add_argument("--image1", type=Path, required=True)
p.add_argument("--image2", type=Path, required=True)
p.add_argument("--out", type=Path, default=Path("runs/krea2-style-app"))
p.add_argument("--server", default="http://127.0.0.1:8188")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
request = dict(template="krea2-turbo/image-style", model="krea2_turbo_fp8_scaled.safetensors", prompt="a ceramic bird", seed=123,
    steps=8, guidance=1.0, width=1024, height=1024, target="layer", enhance=False,
    references=[{"path":str(a.image1.resolve())},{"path":str(a.image2.resolve())}])
for engine in ("comfy", "native"):
    args = [str(a.app), "run", "--new", json.dumps(dict(width=1024,height=1024))]
    for path,value in [("integrations.engine",engine),("integrations.modelsPath",str(a.models)),
                       ("integrations.runtimePath",a.runtime),("integrations.comfyServer",a.server)]:
        args += ["--cmd","prefs.set","--params",json.dumps(dict(path=path,value=value))]
    args += ["--cmd","generate.image","--params",json.dumps(request),"--out",str(a.out/f"{engine}.png")]
    run = subprocess.run(args, capture_output=True, text=True)
    (a.out/f"{engine}.stdout.txt").write_text(run.stdout, encoding="utf-8")
    (a.out/f"{engine}.stderr.txt").write_text(run.stderr, encoding="utf-8")
    run.check_returncode()
from PIL import Image
import numpy as np
ref = np.asarray(Image.open(a.out/"comfy.png").convert("RGB"),dtype=np.float64)/255
actual = np.asarray(Image.open(a.out/"native.png").convert("RGB"),dtype=np.float64)/255
assert actual.shape == ref.shape
mse = np.mean((actual-ref)**2)
metrics = dict(psnr_db=float(-10*np.log10(max(mse,1e-30))),max_abs=float(np.max(np.abs(actual-ref))),
               relative_rms=float(np.sqrt(mse/max(np.mean(ref**2),1e-24))))
(a.out/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
