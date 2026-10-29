# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Overnight 2026-10-27 Q6: Generative Upscale live (q6_upscale.py) on a shared worker.
param([string]$Picture = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-hat.jpg", [string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\q6-upscale",
    [string]$Cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$env:PYTHONIOENCODING = "utf-8"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
$env:PHOTOCRAFT_CLI = $Cli
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
& "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe" -I "$scripts\q6_upscale.py" $Cli $Picture $Out
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
