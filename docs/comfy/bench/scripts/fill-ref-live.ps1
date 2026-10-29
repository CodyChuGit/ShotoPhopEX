# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: Generative Fill from a reference image (PhotoshopEX, 2026-10-21), the technique of
# AusBoss's "Krea 2 Clothing From Reference" workflow on the app's 2511 route:
#   1. two pictures from Krea 2 (a person in a plain cardigan; a red sweater as the reference),
#   2. the cardigan selected by text (SAM 3.1 through ComfyUI),
#   3. generate.fill with the sweater as a reference: the selection goes out flattened to grey,
#      the template names the hole, `grow` gives the bulkier piece room; once as typed and once
#      with the enhancer (which sees the pair side by side),
#   4. the plain fill with the same words, for comparison.
# Output: C:\Users\5090\ComfyUI\photocraft-tests\bench\fill-ref-*.png and the JSON lines.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:FR_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false}}'
$env:FR_NEW = '{\"width\":1024,\"height\":1024}'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-22} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-22} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
    foreach ($l in $lines) { if ("$l" -match '"enhanced"') { "  info.enhanced: [" + ("$l" | ConvertFrom-Json).result.generative.enhanced + "]" } }
}
function Make($tag, $params) {
    $env:FR_PARAMS = $params; $env:FR_OUT = Join-Path $out "$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run --new "%FR_NEW%" --cmd prefs.set --params "%FR_PREFS%" --cmd generate.image --params "%FR_PARAMS%" --out %FR_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
function Fill($tag, $params) {
    $env:FR_PARAMS = $params; $env:FR_OUT = Join-Path $out "$tag.png"; $env:FR_IMAGE = Join-Path $out "fill-ref-person.png"
    $env:FR_SEL = '{\"prompt\":\"the cardigan\"}'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %FR_IMAGE% --cmd prefs.set --params "%FR_PREFS%" --cmd select.byText --params "%FR_SEL%" --cmd generate.fill --params "%FR_PARAMS%" --cmd generate.info --params "{}" --out %FR_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
Make "fill-ref-person" '{\"prompt\":\"a young woman standing in a park on a bright day, wearing a plain light grey knitted cardigan over a white top, looking at the camera, photo\",\"seed\":11,\"target\":\"layer\"}'
Make "fill-ref-sweater" '{\"prompt\":\"product photo of a red cable-knit wool sweater with a round neckline and chunky ribbed cuffs, worn by a person, plain studio background\",\"seed\":12,\"target\":\"layer\"}'
$ref = ($out -replace '\\', '/') + "/fill-ref-sweater.png"
Fill "fill-ref-typed" ('{\"prompt\":\"the red cable-knit sweater\",\"seed\":7,\"grow\":8,\"references\":[{\"path\":\"' + $ref + '\"}]}')
Fill "fill-ref-enhanced" ('{\"prompt\":\"the red sweater\",\"seed\":7,\"grow\":8,\"enhance\":true,\"references\":[{\"path\":\"' + $ref + '\"}]}')
Fill "fill-ref-plain" '{\"prompt\":\"a red cable-knit wool sweater\",\"seed\":7}'
Fill "fill-ref-lightning" ('{\"prompt\":\"the red cable-knit sweater\",\"seed\":7,\"grow\":8,\"template\":\"qwen-edit-2511/fill-ref-lightning-8\",\"references\":[{\"path\":\"' + $ref + '\"}]}')
Fill "fill-ref-lightning-enh" ('{\"prompt\":\"the red sweater\",\"seed\":7,\"grow\":8,\"enhance\":true,\"template\":\"qwen-edit-2511/fill-ref-lightning-8\",\"references\":[{\"path\":\"' + $ref + '\"}]}')
