# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The automatic purge needs one process (the engine remembers the last model set per server):
# three edits in one CLI run, Lightning -> 40-step base -> Lightning, each a different model set.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:PL_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:PL_A = '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":11}'
$env:PL_B = '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":11,\"template\":\"qwen-edit-2511/edit\"}'
$env:PL_C = '{\"prompt\":\"make it a sunny day with a blue sky\",\"seed\":12}'
$env:PL_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\purge-chain.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %PL_IMAGE% --cmd generate.edit --params "%PL_A%" --cmd generate.edit --params "%PL_B%" --cmd generate.edit --params "%PL_C%" --out %PL_OUT%
$sw.Stop()
$i = 0
foreach ($l in $lines) {
    if ("$l" -match '"runId"') {
        $r = ("$l" | ConvertFrom-Json).result; $i++
        "run {0}: {1,7} ms (sampling {2} ms, queue {3} ms) via {4}" -f $i, $r.ms, $r.timings[0].runMs, $r.timings[0].queueMs, $r.template
    }
}
"wall: {0} ms" -f $sw.ElapsedMilliseconds
