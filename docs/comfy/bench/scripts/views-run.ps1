# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# The rewriter's two views per model (2026-10-26, `naming_views`): the worker's 8B names from a
# close-up with a tenth of the stroke's side around it, ComfyUI's 4B keeps a quarter, and the
# close-up keeps its shape beside its surroundings. On the stock set's hard scenes:
#  1. Qwen-Image-2.1 as an edit on the native worker, one candidate (`-e21v`, against `-e21n`)
#     and two judged (`-p21v`, against `-p21`), no ComfyUI.
#  2. The shipped 2511 route (`-auto6`, against `-auto4`): the unsqueezed close-up for the 4B.
param([string]$Tags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:PHOTOCRAFT_THINKER = $null
$keep = { $_ -match "ms grow|FAILED|restarted" }
$short = { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
foreach ($run in @(@("-e21v", 1), @("-p21v", 2))) {
    $suffix, $n = $run
    "=== $suffix (2.1 edit, $n candidate(s), the worker's 8B) $(Get-Date -Format HH:mm:ss)"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Suffix $suffix -Extra "variations=$n;prefill=false;maxPixels=786432" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
}
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== -auto6 (2511 shipped, ComfyUI) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Suffix "-auto6" -Purge heavy 2>&1 | Where-Object $keep | ForEach-Object $short
"=== sheets"
foreach ($pair in @(@("-e21v", "-e21n"), @("-p21v", "-p21"), @("-auto6", "-auto4"))) {
    $s, $b = $pair
    foreach ($c in "half", "whole") {
        & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $s -Base $b -Compare $c -PerSheet 13 -Tags $Tags 2>$null | Out-Null
        Copy-Item "$st\st-sheet-1.png" "$st\st$s-vs$b-$c.png" -ErrorAction SilentlyContinue
    }
}
"=== done $(Get-Date -Format HH:mm:ss)"
