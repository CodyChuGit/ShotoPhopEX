# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The prompt enhancer end to end through the CLI against the live server: what the rewriter
# says for the user's doodle prompts, a fill with a selection, an image idea; then a Generative
# Edit of the doodle with `enhance: true` (does the face appear now?) and a Krea 2 image.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EL_DOODLE = Join-Path $bench "user-drawing.png"
$env:EL_LIGHT = Join-Path $bench "final-fill-s11.png"
function Show($tag, $lines, $ms) {
    foreach ($l in $lines) {
        if ("$l" -match '"enhanced"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-28} {1,6} ms  enhanced={2}`n    {3}" -f $tag, $ms, $r.enhanced, $r.prompt }
        elseif ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-28} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
        elseif ("$l" -match "rror") { "{0,-28} {1}" -f $tag, $l }
    }
}
function Enhance($tag, $image, $params) {
    $env:EL_PARAMS = $params
    $sw = [Diagnostics.Stopwatch]::StartNew()
    if ($image) {
        $env:EL_IMAGE = $image
        $lines = & $cli --% run %EL_IMAGE% --cmd generate.enhancePrompt --params "%EL_PARAMS%"
    }
    else {
        $lines = & $cli --% run --new "{\"width\":512,\"height\":512}" --cmd generate.enhancePrompt --params "%EL_PARAMS%"
    }
    Show $tag $lines $sw.ElapsedMilliseconds
}
Enhance "doodle edit: realistic" $env:EL_DOODLE '{\"prompt\":\"make this hyper realistic\",\"task\":\"edit\",\"seed\":1}'
Enhance "doodle edit: portrait" $env:EL_DOODLE '{\"prompt\":\"turn this drawing into a hyper realistic portrait that looks similar\",\"task\":\"edit\",\"seed\":1}'
Enhance "doodle edit: seed 2" $env:EL_DOODLE '{\"prompt\":\"make this hyper realistic\",\"task\":\"edit\",\"seed\":2}'
Enhance "lighthouse edit: stormy" $env:EL_LIGHT '{\"prompt\":\"make the sky stormy\",\"task\":\"edit\",\"seed\":1}'
Enhance "lighthouse edit: turn it blue" $env:EL_LIGHT '{\"prompt\":\"turn it blue\",\"task\":\"edit\",\"seed\":1}'
# A fill: the sky region selected (the lighthouse is 1024x1024; the sky is the upper part).
$env:EL_SEL = '{\"x\":80,\"y\":40,\"width\":600,\"height\":300}'
$env:EL_PARAMS = '{\"prompt\":\"a hot air balloon\",\"task\":\"fill\",\"seed\":1}'
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %EL_LIGHT% --cmd select.rect --params "%EL_SEL%" --cmd generate.enhancePrompt --params "%EL_PARAMS%"
Show "lighthouse fill: balloon" $lines $sw.ElapsedMilliseconds
$env:EL_PARAMS = '{\"prompt\":\"remove this\",\"task\":\"fill\",\"seed\":1}'
$sw.Restart()
$lines = & $cli --% run %EL_LIGHT% --cmd select.rect --params "%EL_SEL%" --cmd generate.enhancePrompt --params "%EL_PARAMS%"
Show "lighthouse fill: remove this" $lines $sw.ElapsedMilliseconds
Enhance "image idea: fox" $null '{\"prompt\":\"a fox in snow\",\"task\":\"image\",\"seed\":1}'
Enhance "image idea: detailed" $null '{\"prompt\":\"A close-up photo of a weathered fisherman mending a net on a wooden pier at golden hour, shallow depth of field, 85mm lens\",\"task\":\"image\",\"seed\":1}'
# The doodle through Generative Edit with the user's own prompt, enhanced on the way.
$env:EL_PARAMS = '{\"prompt\":\"make this hyper realistic\",\"seed\":1,\"enhance\":true}'
$env:EL_OUT = Join-Path $bench "enhance-doodle-realistic.png"
$sw.Restart()
$lines = & $cli --% run %EL_DOODLE% --cmd generate.edit --params "%EL_PARAMS%" --cmd generate.info --params {} --out %EL_OUT%
Show "edit doodle (enhanced)" $lines $sw.ElapsedMilliseconds
foreach ($l in $lines) { if ("$l" -match '"generative"') { $r = ("$l" | ConvertFrom-Json).result.generative; "    layer remembers: prompt=`"$($r.prompt)`"`n    enhanced=`"$($r.enhanced)`"" } }
$env:EL_PARAMS = '{\"prompt\":\"turn this drawing into a hyper realistic portrait that looks similar\",\"seed\":1,\"enhance\":true}'
$env:EL_OUT = Join-Path $bench "enhance-doodle-portrait.png"
$sw.Restart()
$lines = & $cli --% run %EL_DOODLE% --cmd generate.edit --params "%EL_PARAMS%" --out %EL_OUT%
Show "edit doodle portrait (enh.)" $lines $sw.ElapsedMilliseconds
# A Krea 2 image from a short idea, enhanced.
$env:EL_PARAMS = '{\"prompt\":\"a fox in snow\",\"seed\":1,\"enhance\":true,\"target\":\"document\"}'
$env:EL_OUT = Join-Path $bench "enhance-image-fox.png"
$sw.Restart()
$lines = & $cli --% run --new "{\"width\":512,\"height\":512}" --cmd generate.image --params "%EL_PARAMS%" --out %EL_OUT%
Show "image fox (enhanced)" $lines $sw.ElapsedMilliseconds
$env:EL_PARAMS = '{\"prompt\":\"a fox in snow\",\"seed\":1,\"target\":\"document\"}'
$env:EL_OUT = Join-Path $bench "enhance-image-fox-plain.png"
$sw.Restart()
$lines = & $cli --% run --new "{\"width\":512,\"height\":512}" --cmd generate.image --params "%EL_PARAMS%" --out %EL_OUT%
Show "image fox (as typed)" $lines $sw.ElapsedMilliseconds
(Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8188/system_stats").Content | ConvertFrom-Json | ForEach-Object { "vram free after: {0:N1} GB of {1:N1}" -f ($_.devices[0].vram_free / 1GB), ($_.devices[0].vram_total / 1GB) }
