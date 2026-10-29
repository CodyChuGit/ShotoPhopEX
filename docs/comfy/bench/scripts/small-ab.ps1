# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Small-selection cases: a 120x90 patch of rock (request ~180x135, sent at 512 px) and the
# lighthouse door, with the final Lightning-8 template.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:AB_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:AB_PREFS = '{\"values\":{\"integrations.allowResearchModels\":true}}'
$cases = @(
    @{ tag = "small-rock"; rect = '{\"x\":560,\"y\":700,\"width\":120,\"height\":90}'; seed = 3; prompt = "a seagull standing on the rock" },
    @{ tag = "small-door"; rect = '{\"x\":752,\"y\":520,\"width\":56,\"height\":70}'; seed = 3; prompt = "a round brass porthole window" }
)
foreach ($c in $cases) {
    $tpl = "qwen-edit-2511/fill-lightning-8"
    $env:AB_RECT = $c.rect
    $env:AB_OUT = Join-Path $out "ab-$($c.tag)-lightning-8.png"
    $env:AB_FILL = '{\"prompt\":\"' + $c.prompt + '\",\"template\":\"' + $tpl + '\",\"seed\":' + $c.seed + ',\"margin\":0.25}'
    $lines = & $cli --% run %AB_IMAGE% --cmd prefs.set --params "%AB_PREFS%" --cmd select.rect --params "%AB_RECT%" --cmd generate.fill --params "%AB_FILL%" --out %AB_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-11} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-11} {1,6} ms (sampling {2} ms) rect {3}x{4} sent {5}x{6} -> {7}" -f $c.tag, $r.ms, $r.timings[0].runMs, $r.width, $r.height, $r.requestWidth, $r.requestHeight, $env:AB_OUT }
}
