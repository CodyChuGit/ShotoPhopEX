# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: the soft-brush Spot Healing stroke over the boat (S2 of spot-generative-live.ps1),
# four times at random seeds, with the rewriter's words each time: how often the model puts
# something else (a rock) where the boat was. Output: bench\spot-S2r-*.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_PARAMS = '{\"type\":\"generative\",\"size\":56,\"hardness\":0.3,\"points\":[[300,664],[340,662],[380,660],[420,660],[450,662]]}'
foreach ($i in 1, 2, 3, 4) {
    $env:CM_OUT = Join-Path $out "spot-S2r-$i.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]+)".*"seed":(\d+)') { "S2r-$i ($($sw.ElapsedMilliseconds) ms, seed $($Matches[2])): $($Matches[1])" } }
}
