# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Overnight 2026-10-27, after Q8: two measurements for the owner's decisions.
#  1. A wider removal request (`removeMargin` 0.5 against the default 0.25) on the 2.1 worker
#     route, two judged, where the request's size matters: the dog (its shadow ran out of the
#     request), the boat and the glasses (`-m50`, against `-q2`/`-q2b`).
#  2. The shipped 2511 route (Best of two through ComfyUI) on the stock set's ten hard scenes
#     with tonight's code (`-auto7`, against `-auto6`), so 2.1 and 2511 are counted the same night.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:PHOTOCRAFT_THINKER = $null
$keep = { $_ -match "ms grow|FAILED|restarted" }
$short = { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
"=== -m50 (2.1, removeMargin 0.5) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags "dog,boat,glasses" -Suffix "-m50" -Extra "variations=2;prefill=false;maxPixels=786432;removeMargin=0.5" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== -auto7 (2511 shipped, ComfyUI) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike" -Suffix "-auto7" -Purge heavy 2>&1 | Where-Object $keep | ForEach-Object $short
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
"=== done $(Get-Date -Format HH:mm:ss)"
