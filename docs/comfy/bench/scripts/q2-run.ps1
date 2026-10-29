# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Overnight 2026-10-27 Q2: the change band (`shadow`, `change_band`) on the 2.1 worker route, two
# judged. -Sets stock,rb picks the runs: the stock set's ten hard scenes with it on (`-q2`,
# against `-p21v`), the robustness set with it on (`-q2`) and off (`-q2off`, its baseline).
# -Suffix renames the "on" run (a second pass: `-q2b`); -StockTags/-RbTags limit the scenes
# (an empty value is not passed on: `powershell -File` drops it and the next flag shifts).
param([string]$Sets = "stock,rb", [string]$Suffix = "-q2", [string]$StockTags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike", [string]$RbTags = "", [switch]$NoOff)
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:PHOTOCRAFT_THINKER = $null
$keep = { $_ -match "ms grow|FAILED|restarted" }
$short = { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
$extra = "variations=2;prefill=false;maxPixels=786432"
$tags = { param($t) if ($t) { @("-Tags", $t) } else { @() } }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
if ($Sets -match "stock") {
    "=== stock $Suffix $(Get-Date -Format HH:mm:ss)"
    $a = & $tags $StockTags
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json @a -Suffix $Suffix -Extra $extra -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
}
if ($Sets -match "rb") {
    $a = & $tags $RbTags
    "=== rb $Suffix $(Get-Date -Format HH:mm:ss)"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" @a -Suffix $Suffix -Extra $extra -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
    if (-not $NoOff) {
        "=== rb -q2off $(Get-Date -Format HH:mm:ss)"
        & powershell -NoProfile -File "$scripts\robust-strokes.ps1" @a -Suffix "-q2off" -Extra "$extra;shadow=false" -Purge heavy -Engine auto -Template qwen-2.1/edit 2>&1 | Where-Object $keep | ForEach-Object $short
    }
}
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== done $(Get-Date -Format HH:mm:ss)"
