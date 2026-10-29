# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: the removal's own rewriter note (name the object, then only what is behind it) through
# the tools: the soft-brush boat stroke twice (random seeds) and the half-necklace stroke, with
# the rewriter's words. Output: bench\note-*.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
function Tool($tag, $image, $cmd, $params) {
    $env:CM_IMAGE = Join-Path $out $image; $env:CM_CMD = $cmd; $env:CM_PARAMS = $params; $env:CM_OUT = Join-Path $out "note-$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd %CM_CMD% --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
    $said = $false
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]+)".*"grow":(\d+)') { "$tag ($($sw.ElapsedMilliseconds) ms, grow $($Matches[2])): $($Matches[1])"; $said = $true } }
    if (-not $said) { "$tag FAILED: " + (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 1) -join " ") }
}
$boat = '{\"type\":\"generative\",\"size\":56,\"hardness\":0.3,\"points\":[[300,664],[340,662],[380,660],[420,660],[450,662]]}'
Tool "boat-1" "final-fill-s11.png" "paint.spotHealing" $boat
Tool "boat-2" "final-fill-s11.png" "paint.spotHealing" $boat
Tool "neck-half" "necklace.png" "paint.spotHealing" '{\"type\":\"generative\",\"size\":30,\"hardness\":1,\"points\":[[405,525],[425,590],[450,650],[478,700]]}'
