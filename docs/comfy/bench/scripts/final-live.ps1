# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Final check of the shipped defaults: fill s11 soft, expand right s6/s8 (should hit the cache of
# the 4 %/edge A/B: identical requests) and the frame at the new feather.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:FL_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:FL_RECT = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'
$env:FL_PARAMS = '{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"seed\":11,\"margin\":0.25}'
$env:FL_OUT = Join-Path $out "final-fill-s11.png"
$lines = & $cli --% run %FL_IMAGE% --cmd select.rect --params "%FL_RECT%" --cmd generate.fill --params "%FL_PARAMS%" --out %FL_OUT%
$r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
if ($null -eq $r) { "fill s11      FAILED: {0}" -f (($lines | Select-Object -Last 1) -join " ") } else { "fill s11      {0,6} ms via {1}" -f $r.ms, $r.template }
$cases = @(
    @{ tag = "final-expand-right-s6"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":6}' },
    @{ tag = "final-expand-right-s8"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":8}' },
    @{ tag = "final-expand-empty-s6"; params = '{\"right\":384,\"seed\":6}' },
    @{ tag = "final-expand-frame";    params = '{\"left\":192,\"right\":192,\"bottom\":192,\"seed\":5}' }
)
foreach ($c in $cases) {
    $env:EX_PARAMS = $c.params
    $env:EX_OUT = Join-Path $out "$($c.tag).png"
    $lines = & $cli --% run %FL_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-22} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-22} {1,6} ms via {2}" -f $c.tag, $r.ms, $r.template }
}
