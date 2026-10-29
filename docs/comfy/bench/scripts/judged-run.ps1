# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Both sets under the shipped defaults, as -Suffix (e.g. -auto4), compared with the
# smoothed-only run (-Base, default -smooth). Starts ComfyUI when it is not answering and
# stops it afterwards when it started it. Output: <prefix>-<tag>-{half,whole}<Suffix>.png and
# the sheets st<Suffix>-{half,whole}-{1,2}.png in bench\stock, rb<Suffix>-{half,whole}.png in bench.
param([string]$Suffix = "-auto4", [string]$Base = "-smooth")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
function Up { try { $null = Invoke-RestMethod http://127.0.0.1:8188/system_stats -TimeoutSec 3; $true } catch { $false } }
$started = $false
if (-not (Up)) {
    # (its output kept, so a server that dies mid-run says why)
    Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden -RedirectStandardOutput "$rb\comfyui-run.log" -RedirectStandardError "$rb\comfyui-run.err.log"
    $started = $true
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Up) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (-not (Up)) { "ComfyUI did not come up"; exit 1 }
    "ComfyUI started"
}
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $Suffix 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix $Suffix 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $Suffix -Base $Base -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st$Suffix-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st$Suffix-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix $Suffix -Base $Base -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb$Suffix-$c.png"
}
if ($started) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "ComfyUI_windows_portable" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    "ComfyUI stopped"
}
