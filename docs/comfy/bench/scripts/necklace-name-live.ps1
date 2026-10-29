# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: does naming the object remove the necklace where "the object in the selected area"
# does not (necklace.png, the necklace's silhouette from Select by Text grown 64 and 123,
# seeds 7 and 8, Lightning 8, Quality balanced, pixels kept)?
#   E  REMOVE_PROMPT rewritten by the vision-language model with the picture in view (enhance)
#   N  the object named by hand: the ceiling a naming step could reach
# Output: bench\neck-n-*.png, the timings and E's rewritten prompts.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "necklace.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$remove = 'Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there'
$named = 'Remove the gold chain necklace and its pendant completely, leaving bare skin on the neck and collarbones and the top''s neckline exactly as they are, as if she had never worn it'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-10} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-10} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]+)"') { "    rewritten: $($Matches[1])" } }
}
foreach ($seed in 7, 8) {
    foreach ($grow in 64, 123) {
        $env:CM_SEL = '{\"prompt\":\"necklace\",\"grow\":' + $grow + '}'
        $env:CM_OUT = Join-Path $out "neck-n-E$grow-$seed.png"
        $env:CM_PARAMS = '{\"prompt\":\"' + $remove + '\",\"erase\":false,\"enhance\":true,\"seed\":' + $seed + '}'
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_SEL%" --cmd generate.fill --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
        Report "E$grow-$seed" $lines $sw.ElapsedMilliseconds
        $env:CM_OUT = Join-Path $out "neck-n-N$grow-$seed.png"
        $env:CM_PARAMS = '{\"prompt\":\"' + $named + '\",\"erase\":false,\"seed\":' + $seed + '}'
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_SEL%" --cmd generate.fill --params "%CM_PARAMS%" --out %CM_OUT%
        Report "N$grow-$seed" $lines $sw.ElapsedMilliseconds
    }
}
