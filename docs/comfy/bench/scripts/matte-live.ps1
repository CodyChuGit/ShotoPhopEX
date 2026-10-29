# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Live: generate.removeBackground on the lighthouse (model's choice, then "the lighthouse"), as a
# selection, and a transparent generate.image; the outputs are saved as PSD-free PNGs through
# --out (flattened: the mask is applied, so transparency shows the matte).
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:ML_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:ML_PREFS = '{\"values\":{\"integrations.allowResearchModels\":true}}'
function Run($tag, $cmd, $params) {
    $env:ML_PARAMS = $params
    $env:ML_CMD = $cmd
    $env:ML_OUT = Join-Path $out "$tag.png"
    $lines = & $cli --% run %ML_IMAGE% --cmd prefs.set --params "%ML_PREFS%" --cmd %ML_CMD% --params "%ML_PARAMS%" --out %ML_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-26} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-26} {1,6} ms  bounds {2}  pixels {3}  sent {4}x{5} via {6}" -f $tag, $r.ms, ($r.bounds -join ","), $r.pixels, $r.requestWidth, $r.requestHeight, $r.template }
}
Run "matte-auto" "generate.removeBackground" '{\"seed\":7}'
Run "matte-lighthouse" "generate.removeBackground" '{\"prompt\":\"the lighthouse\",\"seed\":7}'
Run "matte-boat" "generate.removeBackground" '{\"prompt\":\"the red boat\",\"seed\":7}'
Run "matte-selection" "generate.removeBackground" '{\"prompt\":\"the red boat\",\"seed\":7,\"asSelection\":true}'
# A transparent generation into a new document.
$env:ML_PARAMS = '{\"prompt\":\"a vintage brass compass, product photo\",\"transparent\":true,\"template\":\"qwen-2.1/image\",\"target\":\"document\",\"seed\":5}'
$env:ML_OUT = Join-Path $out "transparent-compass.png"
$lines = & $cli --% run %ML_IMAGE% --cmd prefs.set --params "%ML_PREFS%" --cmd generate.image --params "%ML_PARAMS%" --out %ML_OUT%
$lines | Where-Object { $_ -match '"runId"|rror' } | ForEach-Object { if ($_.Length -gt 300) { $_.Substring(0, 300) } else { $_ } }
