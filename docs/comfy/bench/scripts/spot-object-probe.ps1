# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# What the detector answers for the spots of the Spot Healing strokes of spot-generative-live.ps1
# (select.byPoint on the same points), and what the generative stroke's layer records
# (generate.info: the request rect and grow). Prints JSON lines.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_PTS = '{\"points\":[[330,664],[370,662],[410,660]]}'
"== select.byPoint on the short stroke's spots"
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byPoint --params "%CM_PTS%" | Where-Object { $_ -match '"instances"' }
$env:CM_PTS = '{\"points\":[[300,664],[350,662],[400,660],[450,662]]}'
"== select.byPoint on the long stroke's spots"
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byPoint --params "%CM_PTS%" | Where-Object { $_ -match '"instances"' }
$env:CM_SPOT = '{\"type\":\"generative\",\"size\":36,\"hardness\":1,\"seed\":7,\"points\":[[330,664],[370,662],[410,660]]}'
"== the short generative stroke, then generate.info"
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_SPOT%" --cmd generate.info --params "{}" | Where-Object { $_ -match '"rect"|"runId"' }
