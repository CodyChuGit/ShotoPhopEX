# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Overnight 2026-10-27 Q4: LoRA parity through the app's CLI (q4_lora_parity.py): the native
# route on a shared worker, then the ComfyUI route with the worker stopped, then the PSNR table.
param([string]$Cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe", [string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\q4-lora")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$py = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"
$env:PYTHONIOENCODING = "utf-8"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
$env:PHOTOCRAFT_CLI = $Cli
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
"=== native $(Get-Date -Format HH:mm:ss)"
& $py -I "$scripts\q4_lora_parity.py" native $Cli $Out
& "$scripts\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== comfy $(Get-Date -Format HH:mm:ss)"
& $py -I "$scripts\q4_lora_parity.py" comfy $Cli $Out
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
"=== compare $(Get-Date -Format HH:mm:ss)"
& $py -I "$scripts\q4_lora_parity.py" compare $Out
