# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The user's line drawing through Generative Edit with the two prompts they tried in Fill.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
Copy-Item "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\input\photocraft-b5a0623b9109e4b4-image.png" (Join-Path $bench "user-drawing.png") -Force
$env:UD_IMAGE = Join-Path $bench "user-drawing.png"
function Report($tag, $lines, $ms) {
    foreach ($l in $lines) {
        if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-10} {1,6} ms (wall {2,6} ms) via {3}, sent {4}x{5}" -f $tag, $r.ms, $ms, $r.template, $r.requestWidth, $r.requestHeight }
        elseif ("$l" -match "rror") { "{0,-10} {1}" -f $tag, $l }
    }
}
$env:UD_PARAMS = '{\"prompt\":\"turn this drawing into a hyper realistic portrait that looks similar\",\"seed\":1}'
$env:UD_OUT = Join-Path $bench "user-drawing-edit-portrait.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %UD_IMAGE% --cmd generate.edit --params "%UD_PARAMS%" --out %UD_OUT%
Report "portrait" $lines $sw.ElapsedMilliseconds
$env:UD_PARAMS = '{\"prompt\":\"make this hyper realistic\",\"seed\":2}'
$env:UD_OUT = Join-Path $bench "user-drawing-edit-realistic.png"
$sw.Restart()
$lines = & $cli --% run %UD_IMAGE% --cmd generate.edit --params "%UD_PARAMS%" --out %UD_OUT%
Report "realistic" $lines $sw.ElapsedMilliseconds
