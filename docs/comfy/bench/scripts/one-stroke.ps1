# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# One generative Spot Healing stroke in the foreground, the full output shown: for seeing an
# error the batch scripts swallow. -Image, -Points (JSON), -Size.
param([string]$Image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-poster.jpg", [string]$Points = "[[1423,668],[1453,668]]", [int]$Size = 33)
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:CM_IMAGE = $Image
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_PARAMS = '{\"type\":\"generative\",\"size\":' + $Size + ',\"hardness\":1,\"points\":' + $Points + '}'
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_PARAMS%"
