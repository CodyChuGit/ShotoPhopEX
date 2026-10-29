# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The removal sets through the native Qwen-Image-2.1 fill route (the worker resident, the
# research opt-in, `qwen-2.1/fill` named as the template, so no 2511 adapter), one candidate
# a stroke, against the 2511 one-candidate run (-one). Needs a CLI built with
# `--features native-diffusion` and the ComfyUI server up for the rewriter and the detector's
# fallback. Output: <prefix>-<tag>-{half,whole}-n21.png, the sheets st-n21-{half,whole}-{1,2}.png
# in bench\stock and rb-n21-{half,whole}.png in bench (-n21 against -one).
param([string]$Suffix = "-n21", [string]$Extra = "variations=1", [string]$Base = "-one")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
# One worker for the whole run (native-worker.ps1), joined by every stroke's CLI process; the
# ComfyUI server is asked to drop its models first so the card holds the worker's set and the
# rewriter's encoder, not the 2511 too.
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch { "(ComfyUI free: $_)" }
"=== $Suffix ($Extra, native qwen-2.1/fill) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $Suffix -Extra $Extra -Purge heavy -Engine auto -Template qwen-2.1/fill 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix $Suffix -Extra $Extra -Purge heavy -Engine auto -Template qwen-2.1/fill 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
"=== sheets $(Get-Date -Format HH:mm:ss)"
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $Suffix -Base $Base -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st$Suffix-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st$Suffix-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix $Suffix -Base $Base -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb$Suffix-$c.png"
}
& "$scripts\native-worker.ps1" -Stop
"=== done $(Get-Date -Format HH:mm:ss)"
