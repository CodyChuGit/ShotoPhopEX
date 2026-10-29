# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Run the three A/B cases for the given templates (same seeds and selections as guided-ab.ps1).
param([string[]]$Templates = @("qwen-edit-2511/fill-lightning-8"), [string]$Model = "")
# `powershell -File … -Templates a,b` arrives as one string: split it.
$Templates = @($Templates | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne "" })
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:AB_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:AB_PREFS = '{\"values\":{\"integrations.allowResearchModels\":true}}'
$cases = @(
    @{ tag = "edge-s7";  rect = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'; seed = 7;  prompt = "a small red wooden rowing boat floating on the water" },
    @{ tag = "edge-s11"; rect = '{\"x\":320,\"y\":560,\"width\":384,\"height\":320}'; seed = 11; prompt = "a small red wooden rowing boat floating on the water" },
    @{ tag = "sky-s7";   rect = '{\"x\":80,\"y\":120,\"width\":420,\"height\":300}';  seed = 7;  prompt = "a hot air balloon drifting in the evening sky" }
)
foreach ($c in $cases) {
    foreach ($tpl in $Templates) {
        $slug = ($tpl -replace '[/.]', '-')
        $suffix = if ($Model -ne "") { "-" + ($Model -replace '[^a-z0-9]', '') } else { "" }
        $env:AB_RECT = $c.rect
        $env:AB_OUT = Join-Path $out "ab-$($c.tag)-$slug$suffix.png"
        $modelPart = if ($Model -ne "") { ',\"model\":\"' + $Model + '\"' } else { "" }
        $env:AB_FILL = '{\"prompt\":\"' + $c.prompt + '\",\"template\":\"' + $tpl + '\",\"seed\":' + $c.seed + ',\"margin\":0.25' + $modelPart + '}'
        $lines = & $cli --% run %AB_IMAGE% --cmd prefs.set --params "%AB_PREFS%" --cmd select.rect --params "%AB_RECT%" --cmd generate.fill --params "%AB_FILL%" --out %AB_OUT%
        $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
        if ($null -eq $r) { "{0,-9} {1,-36} FAILED: {2}" -f $c.tag, $tpl, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-9} {1,-36} {2,6} ms (sampling {3} ms, encode-cache {4}) -> {5}" -f $c.tag, $tpl, $r.ms, $r.timings[0].runMs, $r.timings[0].queueMs, $env:AB_OUT }
    }
}
