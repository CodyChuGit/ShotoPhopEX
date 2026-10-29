# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A/B: the mask-guided 2511 template against the plain Lightning-8 one, same seed, plus a second
# seed and a second selection (water only) to see placement behaviour.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:AB_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:AB_PREFS = '{\"values\":{\"integrations.allowResearchModels\":true}}'
$cases = @(
    @{ tag = "edge-s7";   rect = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'; seed = 7 },
    @{ tag = "edge-s11";  rect = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'; seed = 11 },
    @{ tag = "sky-s7";    rect = '{\"x\":80,\"y\":120,\"width\":420,\"height\":300}';  seed = 7 }
)
$prompts = @{ "edge-s7" = "a small red wooden rowing boat floating on the water"; "edge-s11" = "a small red wooden rowing boat floating on the water"; "sky-s7" = "a hot air balloon drifting in the evening sky" }
foreach ($c in $cases) {
    foreach ($tpl in @("qwen-edit-2511/fill-lightning-8", "qwen-edit-2511/fill-guided")) {
        $slug = ($tpl -replace '[/.]', '-')
        $env:AB_RECT = $c.rect
        $env:AB_OUT = Join-Path $out "ab-$($c.tag)-$slug.png"
        $env:AB_FILL = '{\"prompt\":\"' + $prompts[$c.tag] + '\",\"template\":\"' + $tpl + '\",\"seed\":' + $c.seed + ',\"margin\":0.25}'
        $t0 = Get-Date
        $lines = & $cli --% run %AB_IMAGE% --cmd prefs.set --params "%AB_PREFS%" --cmd select.rect --params "%AB_RECT%" --cmd generate.fill --params "%AB_FILL%" --out %AB_OUT%
        $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
        if ($null -eq $r) { "{0,-9} {1,-34} FAILED: {2}" -f $c.tag, $tpl, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-9} {1,-34} {2,6} ms (sampling {3} ms) -> {4}" -f $c.tag, $tpl, $r.ms, $r.timings[0].runMs, $env:AB_OUT }
    }
}
