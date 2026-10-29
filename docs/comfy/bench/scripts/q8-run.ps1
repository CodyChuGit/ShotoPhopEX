# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Overnight 2026-10-27 Q8: the fill's seam tone match A/B (q8_fill_tone.py) on a shared worker.
param([string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\q8-tone", [string]$Cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$env:PYTHONIOENCODING = "utf-8"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
$env:PHOTOCRAFT_CLI = $Cli
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
& "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe" -I "$scripts\q8_fill_tone.py" $Cli $Out
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
