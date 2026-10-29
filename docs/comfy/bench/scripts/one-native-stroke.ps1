# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# One removal stroke through the native 2.1 fill with the worker's phase timings in view:
# starts a logging worker, runs robust-strokes.ps1 on one tag and kind, prints the worker's
# log lines (load, encode, sampling, decode) and the stroke's time, stops the worker.
param([string]$Tag = "glasses", [string]$Kind = "whole", [string]$Dir = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock", [string]$Prefix = "st", [string]$Objects = "st-objects.json", [string]$Extra = "variations=1", [string]$Template = "qwen-2.1/fill")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$log = "C:\Users\5090\ComfyUI\photocraft-tests\bench\one-native-worker.log"
$env:RUST_LOG = "info"
. "$scripts\native-worker.ps1" -Log $log
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch { "(ComfyUI free: $_)" }
$sw = [Diagnostics.Stopwatch]::StartNew()
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $Dir -Prefix $Prefix -Objects $Objects -Tags $Tag -Kinds $Kind -Suffix "-one-native" -Extra $Extra -Purge heavy -Engine auto -Template $Template 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
"stroke wall: $($sw.ElapsedMilliseconds) ms"
"--- worker log"
Get-Content "$log.err" -ErrorAction SilentlyContinue | Where-Object { $_ -match "INFO|WARN|ERROR" } | ForEach-Object { ($_ -replace "\x1b\[[0-9;]*m", "") } | ForEach-Object { $_.Substring(0, [Math]::Min(170, $_.Length)) } | Select-Object -Last 40
& "$scripts\native-worker.ps1" -Stop
