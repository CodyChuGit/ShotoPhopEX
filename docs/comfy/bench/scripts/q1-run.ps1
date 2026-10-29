# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Overnight 2026-10-27 Q1: the wearer's candidates (`wearer_candidates`) on the 2.1 worker
# route, two judged, the views the rewriter saw dumped to bench\stock\views-q1 (PHOTOCRAFT_DUMP_VIEWS).
# -Tags picks the scenes (default: the mask scene, `watch`); -Suffix names the outputs.
param([string]$Tags = "watch", [string]$Suffix = "-q1", [string]$Kinds = "half,whole")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:PHOTOCRAFT_THINKER = $null
$env:PHOTOCRAFT_DUMP_VIEWS = Join-Path $st "views$Suffix"
New-Item -ItemType Directory -Force $env:PHOTOCRAFT_DUMP_VIEWS | Out-Null
$keep = { $_ -match "ms grow|FAILED|restarted" }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
"=== $Suffix $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Kinds $Kinds -Suffix $Suffix -Extra "variations=2;prefill=false;maxPixels=786432" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== done $(Get-Date -Format HH:mm:ss)"
