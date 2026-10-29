# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Soft vs hard edges, live: the seed-11 boat fill (the case that showed a seam) and the right-side expand.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EL_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:EL_RECT = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'
$fills = @(
    @{ tag = "edge-s11-soft"; params = '{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"seed\":11,\"margin\":0.25}' },
    @{ tag = "edge-s11-hard"; params = '{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"seed\":11,\"margin\":0.25,\"edge\":\"hard\"}' }
)
foreach ($c in $fills) {
    $env:EL_PARAMS = $c.params
    $env:EL_OUT = Join-Path $out "$($c.tag).png"
    $lines = & $cli --% run %EL_IMAGE% --cmd select.rect --params "%EL_RECT%" --cmd generate.fill --params "%EL_PARAMS%" --out %EL_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-14} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-14} {1,6} ms (sampling {2} ms) rect {3}x{4} sent {5}x{6} via {7} -> {8}" -f $c.tag, $r.ms, $r.timings[0].runMs, $r.width, $r.height, $r.requestWidth, $r.requestHeight, $r.template, $env:EL_OUT }
}
$expands = @(
    @{ tag = "expand-right-soft"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":5}' },
    @{ tag = "expand-right-hard"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":5,\"edge\":\"hard\"}' }
)
foreach ($c in $expands) {
    $env:EL_PARAMS = $c.params
    $env:EL_OUT = Join-Path $out "$($c.tag).png"
    $lines = & $cli --% run %EL_IMAGE% --cmd generate.expand --params "%EL_PARAMS%" --out %EL_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-17} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-17} {1,6} ms (sampling {2} ms) canvas {3} rect {4}x{5} sent {6}x{7} via {8} -> {9}" -f $c.tag, $r.ms, $r.timings[0].runMs, ($r.canvas -join "x"), $r.width, $r.height, $r.requestWidth, $r.requestHeight, $r.template, $env:EL_OUT }
}
