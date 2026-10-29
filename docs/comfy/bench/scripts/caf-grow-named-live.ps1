# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: with the object named by the vision-language rewriter (enhance), does the hole still
# need to reach far past the object? The boat's silhouette (select.byPoint on the stroke's
# spots) grown 15 and 30 (both failed unnamed at least once), seeds 7 and 8, Lightning 8,
# Quality balanced, pixels kept. Output: bench\caf-gn-*.png and the rewritten prompts.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$remove = 'Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-10} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-10} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]+)"') { "    rewritten: $($Matches[1])" } }
}
foreach ($seed in 7, 8) {
    foreach ($grow in 15, 30) {
        $env:CM_SEL = '{\"points\":[[330,664],[370,662],[410,660]],\"grow\":' + $grow + '}'
        $env:CM_OUT = Join-Path $out "caf-gn-$grow-$seed.png"
        $env:CM_PARAMS = '{\"prompt\":\"' + $remove + '\",\"erase\":false,\"enhance\":true,\"seed\":' + $seed + '}'
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byPoint --params "%CM_SEL%" --cmd generate.fill --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
        Report "gn$grow-$seed" $lines $sw.ElapsedMilliseconds
    }
}
