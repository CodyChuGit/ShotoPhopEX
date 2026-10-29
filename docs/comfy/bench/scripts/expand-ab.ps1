# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A/B for the expand's feather width and prefill, seeds 6 and 8 on the right-side widen.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
Add-Type -AssemblyName System.Drawing
function Profile($path) {
    $img = [System.Drawing.Bitmap]::FromFile($path); $line = ""
    for ($x = 980; $x -le 1060; $x += 4) { $sum = 0.0; for ($y = 200; $y -lt 420; $y += 2) { $c = $img.GetPixel($x, $y); $sum += ($c.R + $c.G + $c.B) / 3.0 }; $line += " {0}" -f [int]($sum / 110) }
    $img.Dispose(); $line
}
$variants = @(
    @{ tag = "f08-edge"; feather = "0.08"; prefill = "edge" },
    @{ tag = "f08-grey"; feather = "0.08"; prefill = "grey" },
    @{ tag = "f04-edge"; feather = "0.04"; prefill = "edge" },
    @{ tag = "f02-edge"; feather = "0.02"; prefill = "edge" },
    @{ tag = "f04-grey"; feather = "0.04"; prefill = "grey" }
)
foreach ($v in $variants) {
    $env:PC_EXPAND_FEATHER = $v.feather
    if ($v.prefill -eq "grey") { $env:PC_PREFILL = "grey" } else { Remove-Item Env:PC_PREFILL -ErrorAction SilentlyContinue }
    foreach ($seed in @(6, 8)) {
        $env:EX_PARAMS = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":' + $seed + '}'
        $env:EX_OUT = Join-Path $out ("xab-{0}-s{1}.png" -f $v.tag, $seed)
        $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
        $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
        if ($null -eq $r) { "{0,-9} s{1} FAILED: {2}" -f $v.tag, $seed, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-9} s{1} {2,6} ms  profile x980..1060:{3}" -f $v.tag, $seed, $r.ms, (Profile $env:EX_OUT) }
    }
}
Remove-Item Env:PC_EXPAND_FEATHER, Env:PC_PREFILL -ErrorAction SilentlyContinue
