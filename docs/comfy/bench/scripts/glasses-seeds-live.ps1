# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# The glasses scene's half stroke (robust set) seed by seed: variations 1 at seeds 1..4, then
# variations 2 at seeds 1 and 3 (which choose between 1,2 and 3,4), to see what the edge
# choice picks. Output: bench\gl-s<seed>.png, gl-v2-<seed>.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "rb-glasses.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
function Run($tag, $seed, $vars) {
    $env:CM_PARAMS = '{\"type\":\"generative\",\"size\":64,\"hardness\":1,\"variations\":' + $vars + ',\"seed\":' + $seed + ',\"points\":[[245,499],[512,499]]}'
    $env:CM_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
    foreach ($l in $lines) { if ("$l" -match '"seeds":\[([^\]]*)\]') { "{0,-10} {1,6} ms seeds {2}" -f $tag, $sw.ElapsedMilliseconds, $Matches[1] } }
}
foreach ($s in 1, 2, 3, 4) { Run "gl-s$s" $s 1 }
foreach ($s in 1, 3) { Run "gl-v2-$s" $s 2 }
