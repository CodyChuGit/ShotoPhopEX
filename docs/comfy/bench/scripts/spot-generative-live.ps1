# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: the generative Spot Healing Brush on the lighthouse's red boat (final-fill-s11.png; the
# boat spans x 305..445, y 645..682), Lightning 8 (Quality balanced), seed 7, through the tool's
# own command (paint.spotHealing type generative), so what the app sends is what is measured:
#   S1  a hard brush, size 56, one stroke along the boat (the whole boat inside the dab)
#   S2  a soft brush (hardness 0.3), the same stroke: the dab's edge covers the boat's rim below half
#   S3  a hard brush, size 36: the stroke misses the boat's bow and stern (the object reaches out)
# Output: bench\spot-*.png and the timings.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-14} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-14} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
}
function Spot($tag, $params) {
    $env:CM_PARAMS = $params; $env:CM_OUT = Join-Path $out "spot-$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_PARAMS%" --out %CM_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
$long = '[[300,664],[340,662],[380,660],[420,660],[450,662]]'
$short = '[[330,664],[370,662],[410,660]]'
Spot "S1-hard56" ('{\"type\":\"generative\",\"size\":56,\"hardness\":1,\"seed\":7,\"points\":' + $long + '}')
Spot "S2-soft56" ('{\"type\":\"generative\",\"size\":56,\"hardness\":0.3,\"seed\":7,\"points\":' + $long + '}')
Spot "S3-hard36" ('{\"type\":\"generative\",\"size\":36,\"hardness\":1,\"seed\":7,\"points\":' + $short + '}')
