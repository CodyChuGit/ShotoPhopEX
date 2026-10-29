# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The user's two doodle prompts through Generative Edit with the final rewriter rules.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EF_DOODLE = Join-Path $bench "user-drawing.png"
function Show($tag, $lines, $ms) {
    foreach ($l in $lines) {
        if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-30} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
        elseif ("$l" -match '"generative"') { $r = ("$l" | ConvertFrom-Json).result.generative; "    enhanced: $($r.enhanced)" }
        elseif ("$l" -match "rror") { "{0,-30} {1}" -f $tag, $l }
    }
}
$env:EF_PARAMS = '{\"prompt\":\"make this hyper realistic\",\"seed\":1,\"enhance\":true}'
$env:EF_OUT = Join-Path $bench "enhance-doodle-realistic-final.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %EF_DOODLE% --cmd generate.edit --params "%EF_PARAMS%" --cmd generate.info --params {} --out %EF_OUT%
Show "doodle: make this hyper realistic" $lines $sw.ElapsedMilliseconds
$env:EF_PARAMS = '{\"prompt\":\"turn this drawing into a hyper realistic portrait that looks similar\",\"seed\":1,\"enhance\":true}'
$env:EF_OUT = Join-Path $bench "enhance-doodle-portrait-final.png"
$sw.Restart()
$lines = & $cli --% run %EF_DOODLE% --cmd generate.edit --params "%EF_PARAMS%" --cmd generate.info --params {} --out %EF_OUT%
Show "doodle: portrait" $lines $sw.ElapsedMilliseconds
