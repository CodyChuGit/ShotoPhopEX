# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The rewriter's wording, checked where it moved (2026-10-26): the shipped 2511 route with
# ComfyUI's 4B on the scenes the wording changed (`-auto5`), then Qwen-Image-2.1 as an edit with
# the worker's 8B, two seeds judged (`-p22`), on the scenes where the 8B named the person.
param([string]$Comfy = "bicycle,boat,glasses", [string]$Native = "glasses,watch,bicycle,cup")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
"=== -auto5 (2511, 4B) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Comfy -Suffix "-auto5" -Purge heavy 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
. "$scripts\native-worker.ps1"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
"=== -p22 (2.1 edit, 8B, two seeds) $(Get-Date -Format HH:mm:ss)"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Native -Suffix "-p22" -Extra "variations=2;prefill=false;maxPixels=786432" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarted" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
& "$scripts\native-worker.ps1" -Stop
foreach ($run in @(@("-auto5", $Comfy), @("-p22", $Native))) {
    $s, $t = $run
    foreach ($c in "half", "whole") {
        & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $s -Base "-auto3" -Compare $c -PerSheet 13 -Tags $t 2>$null | Out-Null
        Copy-Item "$st\st-sheet-1.png" "$st\st$s-$c.png" -ErrorAction SilentlyContinue
    }
}
"=== done $(Get-Date -Format HH:mm:ss)"
