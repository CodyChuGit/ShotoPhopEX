# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Candidate rewritten sentences for "make this hyper realistic" on the doodle, straight through
# generate.edit (no rewriter), to see which wording makes 2511 draw a human face.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:DS_IMAGE = Join-Path $bench "user-drawing.png"
$cands = @(
    @{ tag = "A-real-face-replace-lines"; p = "Turn the hand-drawn smiley face into a photograph of a real human face with the same expression and pose, replacing the black outline strokes with real skin, hair and facial features, keeping the white background unchanged." },
    @{ tag = "B-photo-of-real-face"; p = "Make the hand-drawn doodle of a smiling face look like a hyper-realistic photograph of a real human face with the same expression and pose, keeping the white background unchanged." },
    @{ tag = "C-portrait-no-lines"; p = "Turn the hand-drawn smiley face into a hyper realistic portrait photo of a real person with the same smiling expression and head pose, with no drawn lines left, keeping the white background unchanged." }
)
foreach ($c in $cands) {
    # Every quote escaped: the --% line passes the value inside "…" to the exe verbatim.
    $env:DS_PARAMS = '{\"prompt\":\"' + $c.p.Replace('"', '\"') + '\",\"seed\":1,\"enhance\":false}'
    $env:DS_OUT = Join-Path $bench ("doodle-sentence-" + $c.tag + ".png")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %DS_IMAGE% --cmd generate.edit --params "%DS_PARAMS%" --out %DS_OUT%
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "{0,-28} {1,6} ms via {2}" -f $c.tag, $r.ms, $r.template } elseif ("$l" -match "rror") { "{0,-28} {1}" -f $c.tag, $l } }
}
