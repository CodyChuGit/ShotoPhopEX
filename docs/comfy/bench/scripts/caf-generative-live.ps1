# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: Content-Aware Fill through the generative model (PhotoshopEX, 2026-10-21) against the
# classical PatchMatch fill, on the lighthouse picture's red boat (final-fill-s11.png, the rect
# edit-live.ps1 uses): classical, generative with the hole grey (the default), generative with
# the pixels kept. Then the Remove tool over the same spot. Output: bench\caf-*.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CG_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CG_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false}}'
$env:CG_RECT = '{\"x\":300,\"y\":600,\"width\":200,\"height\":120}'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"filled"|"runId"|"damage"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-26} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    elseif ($null -eq $r.runId) { "{0,-26} classical (wall {1,6} ms)" -f $tag, $ms }
    else { "{0,-26} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
}
function Caf($tag, $params) {
    $env:CG_PARAMS = $params; $env:CG_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CG_IMAGE% --cmd prefs.set --params "%CG_PREFS%" --cmd select.rect --params "%CG_RECT%" --cmd edit.contentAwareFill --params "%CG_PARAMS%" --out %CG_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
function Remove($tag, $params) {
    $env:CG_PARAMS = $params; $env:CG_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CG_IMAGE% --cmd prefs.set --params "%CG_PREFS%" --cmd paint.remove --params "%CG_PARAMS%" --out %CG_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
Caf "caf-classical" '{\"output\":\"current\"}'
Caf "caf-generative" '{\"generative\":true,\"seed\":7}'
Caf "caf-generative-keep" '{\"generative\":true,\"erase\":false,\"seed\":7}'
Remove "caf-remove-classical" '{\"points\":[[330,660],[470,660]],\"size\":90,\"generative\":false}'
Remove "caf-remove-generative" '{\"points\":[[330,660],[470,660]],\"size\":90,\"generative\":true,\"seed\":7}'
