# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# The rewriter on the native worker: generate.enhancePrompt under engine native (the worker's
# 2.1 text encoder answering as Qwen3-VL 8B, genai-native Pipeline::chat), an edit with the
# picture in view and an image idea, with the worker's chat timings. -Cli names the CLI build.
param([string]$Cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe", [string]$Image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-glasses.jpg")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:RUST_LOG = "info"
$env:PHOTOCRAFT_CLI = $Cli
. "$scripts\native-worker.ps1" -Log "$out\chat-smoke-worker.log"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
$env:CM_IMAGE = $Image
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"native\",\"integrations.allowResearchModels\":true,\"integrations.runtimePath\":\"C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib\",\"integrations.modelsPath\":\"C:/Users/5090/Documents/_eval/native-models\"}}'
function Invoke-Enhance {
    & $Cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd generate.enhancePrompt --params "%CM_ASK%"
}
foreach ($ask in @('{\"prompt\":\"remove the glasses\",\"task\":\"edit\",\"seed\":3}', '{\"prompt\":\"make it winter\",\"task\":\"edit\",\"seed\":4}', '{\"prompt\":\"a red bicycle by a white wall\",\"task\":\"image\",\"seed\":5}')) {
    $env:CM_ASK = $ask
    $sw = [Diagnostics.Stopwatch]::StartNew()
    "== $ask"
    Invoke-Enhance 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "enhancePrompt|rror" } | ForEach-Object { "   " + $_.Substring(0, [Math]::Min(400, $_.Length)) }
    "   ($($sw.ElapsedMilliseconds) ms)"
}
"== worker"
Get-Content "$out\chat-smoke-worker.log.err" -ErrorAction SilentlyContinue | ForEach-Object { ($_ -replace "\x1b\[[0-9;]*m", "") } | Where-Object { $_ -match "chat|head|loaded|WARN|ERROR" } | ForEach-Object { $_.Substring(0, [Math]::Min(170, $_.Length)) }
& "$scripts\native-worker.ps1" -Stop
