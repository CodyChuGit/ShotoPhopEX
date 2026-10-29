# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: which removal instruction works on the 2511 fill (PhotoshopEX, 2026-10-21), on the
# lighthouse's red boat (final-fill-s11.png, rect 300,600 200x120), Lightning 8 (Quality balanced):
#   A  pixels kept, a generic "remove the object in the selected area" instruction
#   B  the mask-guided template (image 2 = the mask), pixels kept
#   C  pixels kept, the object named ("the red boat"): the ceiling a VLM naming step could reach
#   D  the hole flat grey (erase), the REMOVE_PROMPT wording
#   E  the guided template with the hole grey
#   F  the hole filled with the median colour of the surroundings is not here (needs code)
# Output: bench\caf-m-*.png and the timings.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_RECT = '{\"x\":300,\"y\":600,\"width\":200,\"height\":120}'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-14} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-14} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
}
function Fill($tag, $params) {
    $env:CM_PARAMS = $params; $env:CM_OUT = Join-Path $out "caf-m-$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.rect --params "%CM_RECT%" --cmd generate.fill --params "%CM_PARAMS%" --out %CM_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
Fill "A-generic" '{\"prompt\":\"Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there\",\"erase\":false,\"seed\":7}'
Fill "B-guided" '{\"prompt\":\"remove the object there and continue the surrounding scene across it seamlessly, as if nothing had ever been there\",\"erase\":false,\"template\":\"qwen-edit-2511/fill-guided\",\"seed\":7}'
Fill "C-named" '{\"prompt\":\"Remove the red boat and fill its place with the surrounding sea continued seamlessly, as if it had never been there\",\"erase\":false,\"seed\":7}'
Fill "D-grey" '{\"prompt\":\"Fill the flat grey area with exactly what the surrounding scene would show there, continuing its background, surfaces, textures, patterns and lines seamlessly across it, as if nothing had ever been there\",\"erase\":true,\"seed\":7}'
Fill "E-guided-grey" '{\"prompt\":\"replace the flat grey with the surrounding scene continued seamlessly across it, as if nothing had ever been there\",\"erase\":true,\"template\":\"qwen-edit-2511/fill-guided\",\"seed\":7}'
