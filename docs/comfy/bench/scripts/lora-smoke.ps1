# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# The LoRA engine's first live check: a Qwen-Image-2.1 edit through the native worker with and
# without AusBoss's Consistency LoRA (the same seed), the worker's warnings (an unmatched module
# would mean the file's names do not map), and the pixel difference between the two results.
# Needs a CLI built with --features native-diffusion; starts its own worker (RUST_LOG=warn).
param([string]$Image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-dog.jpg", [string]$Prompt = "make it a watercolor painting", [string]$Lora = "qwen-image-2.1-consistency.safetensors", [double]$Strength = 1.0)
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:RUST_LOG = "warn"
. "$scripts\native-worker.ps1" -Log "$out\lora-smoke-worker.log"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:CM_IMAGE = $Image
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"auto\",\"integrations.allowResearchModels\":true,\"integrations.enhancePrompts\":false,\"integrations.runtimePath\":\"C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib\",\"integrations.modelsPath\":\"C:/Users/5090/Documents/_eval/native-models\"}}'
function Invoke-Edit {
    & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd generate.edit --params "%CM_EDIT%" --out %CM_OUT%
}
foreach ($variant in @(@("base", ""), @("lora", (',\"loras\":[{\"file\":\"' + $Lora + '\",\"strength\":' + $Strength + '}]')))) {
    $tag, $extra = $variant
    $env:CM_EDIT = '{\"prompt\":\"' + $Prompt + '\",\"template\":\"qwen-2.1/edit\",\"seed\":11,\"enhance\":false' + $extra + '}'
    $env:CM_OUT = Join-Path $out "lora-smoke-$tag.png"
    "== $tag"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-Edit 2>&1 | ForEach-Object { $l = "$_"; if ($l -match '"template":"([^"]+)".*"ms":(\d+)' -or $l -match '"ms":(\d+)') { "   $l".Substring(0, [Math]::Min(160, $l.Length + 3)) } elseif ($l -match 'rror') { "   $l" } }
    "   ($($sw.ElapsedMilliseconds) ms)"
}
"== worker warnings"
Get-Content "$out\lora-smoke-worker.log.err" -ErrorAction SilentlyContinue | Where-Object { $_ -match "WARN|ERROR|LoRA" } | Select-Object -First 10
& "$scripts\native-worker.ps1" -Stop
"== difference"
$py = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"
& $py -I -c "import sys; from PIL import Image, ImageChops; import math; a=Image.open(sys.argv[1]).convert('RGB'); b=Image.open(sys.argv[2]).convert('RGB'); d=ImageChops.difference(a,b); h=d.histogram(); n=a.size[0]*a.size[1]*3; mse=sum(i*i*c for ch in range(3) for i,c in enumerate(h[ch*256:(ch+1)*256]))/n; print('size', a.size, 'psnr %.1f dB' % (99 if mse==0 else 10*math.log10(255*255/mse)))" (Join-Path $out "lora-smoke-base.png") (Join-Path $out "lora-smoke-lora.png")
