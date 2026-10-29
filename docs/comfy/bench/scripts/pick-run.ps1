# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Best of two on both routes, on the stock set's hard scenes (2026-10-26):
#  1. Qwen-Image-2.1 as an edit on the native worker, two seeds, the worker's 8B rewriter and
#     judge (`-p21`), against the shipped 2511 run (`-auto3`: the adapter and the smoothed
#     fill, judged by ComfyUI's 4B).
#  2. The shipped 2511 route again with today's rewriter wording (`-auto4`), to see what the
#     wording does to the 4B's names and the results.
param([string]$Tags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:PHOTOCRAFT_THINKER = $null
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
"=== -p21 (2.1 edit, two seeds, native judge) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Suffix "-p21" -Extra "variations=2;prefill=false;maxPixels=786432" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarted" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== -auto4 (2511 shipped, ComfyUI) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Suffix "-auto4" -Purge heavy 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarted" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
"=== sheets"
foreach ($pair in @(@("-p21", "-auto3"), @("-auto4", "-auto3"))) {
    $s, $b = $pair
    foreach ($c in "half", "whole") {
        & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $s -Base $b -Compare $c -PerSheet 13 -Tags $Tags 2>$null | Out-Null
        Copy-Item "$st\st-sheet-1.png" "$st\st$s-vs$b-$c.png" -ErrorAction SilentlyContinue
    }
}
"=== done $(Get-Date -Format HH:mm:ss)"
