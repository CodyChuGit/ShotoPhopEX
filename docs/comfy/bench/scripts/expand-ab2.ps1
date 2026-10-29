# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A/B 2: consistency of the new side. (a) the 40-step base expand template, (b) Lightning with a
# haze-matching instruction, both at 4 % feather with the edge prefill, seeds 6 and 8.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:PC_EXPAND_FEATHER = "0.04"
Remove-Item Env:PC_PREFILL -ErrorAction SilentlyContinue
$haze = "Add more open sea and evening sky, keeping exactly the same soft haze, muted colours, light and softness as the original photo"
$variants = @(
    @{ tag = "base40";   params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"template\":\"qwen-edit-2511/expand\",\"seed\":SEED}' },
    @{ tag = "l8-haze";  params = '{\"right\":384,\"prompt\":\"' + $haze + '\",\"seed\":SEED}' },
    @{ tag = "l8-empty"; params = '{\"right\":384,\"seed\":SEED}' }
)
foreach ($v in $variants) {
    foreach ($seed in @(6, 8)) {
        $env:EX_PARAMS = $v.params -replace 'SEED', $seed
        $env:EX_OUT = Join-Path $out ("xab2-{0}-s{1}.png" -f $v.tag, $seed)
        $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
        $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
        if ($null -eq $r) { "{0,-9} s{1} FAILED: {2}" -f $v.tag, $seed, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-9} s{1} {2,6} ms via {3}" -f $v.tag, $seed, $r.ms, $r.template }
    }
}
Remove-Item Env:PC_EXPAND_FEATHER -ErrorAction SilentlyContinue
