# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: how far past a thin object (the necklace of necklace.png: a 6-px chain, 5296 px in a
# 409x297 box) the hole must reach for the 2511 fill to remove it. The necklace's silhouette
# from Select by Text grown by 32, 64, 96 and 123 (30 % of its larger side), seeds 7 and 8,
# Lightning 8, Quality balanced, REMOVE_PROMPT, pixels kept. Output: bench\neck-g-*.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "necklace.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$remove = 'Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-10} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-10} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
}
foreach ($seed in 7, 8) {
    foreach ($grow in 32, 64, 96, 123) {
        $env:CM_SEL = '{\"prompt\":\"necklace\",\"grow\":' + $grow + '}'
        $env:CM_OUT = Join-Path $out "neck-g-$grow-$seed.png"
        $env:CM_PARAMS = '{\"prompt\":\"' + $remove + '\",\"erase\":false,\"seed\":' + $seed + '}'
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_SEL%" --cmd generate.fill --params "%CM_PARAMS%" --out %CM_OUT%
        Report "g$grow-$seed" $lines $sw.ElapsedMilliseconds
    }
}
