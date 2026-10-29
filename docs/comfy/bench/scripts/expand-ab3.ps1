# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A/B 3: wrapper wordings for a described expand (sent as imperative prompts so the CLI can test them
# without a rebuild), Lightning 8, 4 % feather, edge prefill, seeds 6 and 8.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:PC_EXPAND_FEATHER = "0.04"
Remove-Item Env:PC_PREFILL -ErrorAction SilentlyContinue
$w1 = "Extend this image beyond its current edges, continuing the scene naturally with more open sea and evening sky. Match the original's perspective, lighting, colours, haze and softness exactly and do not repeat its objects"
$w2 = "Extend this image to show more open sea and evening sky beyond its current edges, continuing the original photo naturally with the same haze, light and colours"
$variants = @(
    @{ tag = "w1"; prompt = $w1 },
    @{ tag = "w2"; prompt = $w2 }
)
foreach ($v in $variants) {
    foreach ($seed in @(6, 8)) {
        $env:EX_PARAMS = '{\"right\":384,\"prompt\":\"' + $v.prompt + '\",\"seed\":' + $seed + '}'
        $env:EX_OUT = Join-Path $out ("xab3-{0}-s{1}.png" -f $v.tag, $seed)
        $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
        $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
        if ($null -eq $r) { "{0,-4} s{1} FAILED: {2}" -f $v.tag, $seed, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-4} s{1} {2,6} ms" -f $v.tag, $seed, $r.ms }
    }
}
Remove-Item Env:PC_EXPAND_FEATHER -ErrorAction SilentlyContinue
