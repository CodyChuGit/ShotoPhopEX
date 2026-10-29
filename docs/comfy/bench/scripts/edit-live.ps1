# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Live: generate.edit on the lighthouse (whole picture, then confined to a selection), and the
# automatic purge across model switches (Lightning edit -> 40-step base -> 2.1 matte -> Lightning).
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EL_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:EL_PREFS = '{\"values\":{\"integrations.allowResearchModels\":true}}'
$env:EL_RECT = '{\"x\":300,\"y\":600,\"width\":200,\"height\":120}'
function Report($tag, $lines, $ms) {
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-24} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-24} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
}
function RunPlain($tag, $cmd, $params) {
    $env:EL_PARAMS = $params; $env:EL_CMD = $cmd; $env:EL_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %EL_IMAGE% --cmd prefs.set --params "%EL_PREFS%" --cmd %EL_CMD% --params "%EL_PARAMS%" --out %EL_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
function RunSelected($tag, $cmd, $params) {
    $env:EL_PARAMS = $params; $env:EL_CMD = $cmd; $env:EL_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %EL_IMAGE% --cmd prefs.set --params "%EL_PREFS%" --cmd select.rect --params "%EL_RECT%" --cmd %EL_CMD% --params "%EL_PARAMS%" --out %EL_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
# The 2.1 matte was the last run on the server: the first edit switches model sets (purge).
RunPlain "edit-stormy" "generate.edit" '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":3}'
RunSelected "edit-boat-blue-sel" "generate.edit" '{\"prompt\":\"turn the red boat blue\",\"seed\":3}'
RunPlain "edit-base40" "generate.edit" '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":3,\"template\":\"qwen-edit-2511/edit\"}'
RunPlain "edit-matte-after" "generate.removeBackground" '{\"prompt\":\"the red boat\",\"seed\":8}'
RunPlain "edit-lightning-after" "generate.edit" '{\"prompt\":\"make it a sunny day with a blue sky\",\"seed\":5}'
