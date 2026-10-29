# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Live: the generative retouch tools on the necklace (necklace.png: a 6-px gold chain from
# (405,520) down to a pendant at (481,722) and up to (800,445)), through the tools' own commands,
# Lightning 8 (Quality balanced):
#   K1  Spot Healing, a stroke along the left half of the chain: the detector should take the
#       necklace whole, the rewriter name it
#   K2  Spot Healing, one dab on the pendant: under a quarter of the necklace, so the dab alone
#   K3  the Remove tool, a stroke along the whole chain
# Output: bench\neck-k*.png and the timings, with the rewritten prompts.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "necklace.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
function Report($tag, $lines, $ms) {
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-10} FAILED: {1}" -f $tag, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 2) -join " ") }
    else { "{0,-10} {1,6} ms (wall {2,6} ms) via {3}" -f $tag, $r.ms, $ms, $r.template }
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]+)".*"grow":(\d+)') { "    grow $($Matches[2]); rewritten: $($Matches[1])" } }
}
function Tool($tag, $cmd, $params) {
    $env:CM_CMD = $cmd; $env:CM_PARAMS = $params; $env:CM_OUT = Join-Path $out "neck-$tag.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd %CM_CMD% --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
    Report $tag $lines $sw.ElapsedMilliseconds
}
Tool "k1-half" "paint.spotHealing" '{\"type\":\"generative\",\"size\":30,\"hardness\":1,\"points\":[[405,525],[425,590],[450,650],[478,700]]}'
Tool "k2-pendant" "paint.spotHealing" '{\"type\":\"generative\",\"size\":40,\"hardness\":1,\"points\":[[481,722]]}'
Tool "k3-remove" "paint.remove" '{\"generative\":true,\"size\":24,\"points\":[[405,520],[420,590],[450,650],[480,700],[520,680],[600,630],[700,560],[800,450]]}'
