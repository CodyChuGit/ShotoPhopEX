# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Repeat the empty-prompt seed-6 right expand twice in the same server state.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$env:EX_PARAMS = '{\"right\":384,\"seed\":6}'
foreach ($tag in @("b", "c")) {
    $env:EX_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-expand-empty-s6-$tag.png"
    $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "repeat $tag FAILED: " + (($lines | Select-Object -Last 1) -join " ") } else { "repeat {0}: {1} ms" -f $tag, $r.ms }
}
