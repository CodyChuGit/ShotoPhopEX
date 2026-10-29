# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Overnight 2026-10-27: `nameShadow` (the removal prompt asks for the object "and its shadow" on
# the 2.1 route) on the stock set's ten hard scenes, 2.1 two judged on the worker (`-q2s`,
# against `-q2`/`-q2b`). -Sets rb adds the robustness set (`-q2s` against `-q2b`).
param([string]$Cli = "C:\Users\5090\Documents\PhotoshopEX\target-q\release\photocraft-cli.exe", [string]$Sets = "stock", [string]$Tags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:PHOTOCRAFT_THINKER = $null
$env:PHOTOCRAFT_CLI = $Cli
$keep = { $_ -match "ms grow|FAILED|restarted" }
$short = { "$_" -replace ': Remove the (.{0,50}).*', ': $1' }
$extra = "variations=2;prefill=false;maxPixels=786432;nameShadow=true"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
if ($Sets -match "stock") {
    "=== stock -q2s $(Get-Date -Format HH:mm:ss)"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Suffix "-q2s" -Extra $extra -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
}
if ($Sets -match "rb") {
    "=== rb -q2s $(Get-Date -Format HH:mm:ss)"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Suffix "-q2s" -Extra $extra -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
}
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== done $(Get-Date -Format HH:mm:ss)"
