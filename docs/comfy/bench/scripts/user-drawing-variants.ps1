# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Prompt and tier variants for the doodle: does the model stop seeing a plate?
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:UV_IMAGE = Join-Path $bench "user-drawing.png"
function Report($tag, $lines, $ms) {
    foreach ($l in $lines) {
        if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-12} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
        elseif ("$l" -match "rror") { "{0,-12} {1}" -f $tag, $l }
    }
}
$cases = @(
    @{ tag = "descriptive"; params = '{\"prompt\":\"Redraw this simple line drawing of a smiling face as a photorealistic portrait photo of a person with the same round face, two eyes, a nose and a smile, in the same position\",\"seed\":1}' },
    @{ tag = "noun";        params = '{\"prompt\":\"a photorealistic portrait photo of a smiling man, framed like the sketch\",\"seed\":1}' },
    @{ tag = "sketch-word"; params = '{\"prompt\":\"turn this sketch of a face into a realistic photo of a human face\",\"seed\":1}' },
    @{ tag = "base40";      params = '{\"prompt\":\"turn this drawing into a hyper realistic portrait that looks similar\",\"seed\":1,\"template\":\"qwen-edit-2511/edit\"}' }
)
foreach ($c in $cases) {
    $env:UV_PARAMS = $c.params
    $env:UV_OUT = Join-Path $bench ("user-drawing-" + $c.tag + ".png")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %UV_IMAGE% --cmd generate.edit --params "%UV_PARAMS%" --out %UV_OUT%
    Report $c.tag $lines $sw.ElapsedMilliseconds
}
