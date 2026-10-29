# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$cases = @(
    @{ tag = "expand-right";  params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":5}' },
    @{ tag = "expand-frame";  params = '{\"left\":192,\"right\":192,\"bottom\":192,\"seed\":5}' }
)
foreach ($c in $cases) {
    $env:EX_PARAMS = $c.params
    $env:EX_OUT = Join-Path $out "$($c.tag).png"
    $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-13} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-13} {1,6} ms (sampling {2} ms) canvas {3} rect {4}x{5} sent {6}x{7} via {8} -> {9}" -f $c.tag, $r.ms, $r.timings[0].runMs, ($r.canvas -join "x"), $r.width, $r.height, $r.requestWidth, $r.requestHeight, $r.template, $env:EX_OUT }
}
