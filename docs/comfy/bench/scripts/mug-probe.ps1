# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# What Select by Text finds on the mug scene for the names the rewriter and its fallbacks give.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "rb-mug.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
foreach ($name in "white ceramic coffee cup", "coffee cup", "cup", "mug") {
    $env:CM_TXT = '{\"prompt\":\"' + $name + '\"}'
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_TXT%"
    $r = $null
    foreach ($l in $lines) { if ("$l" -match '"instances"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-26} nothing: {1}" -f $name, (($lines | Select-Object -Last 1) -join " ") }
    else { "{0,-26} {1} instance(s): {2}" -f $name, $r.count, (($r.instances | ForEach-Object { "[" + ($_.bounds -join ",") + "] " + $_.pixels + " px" }) -join "; ") }
}
