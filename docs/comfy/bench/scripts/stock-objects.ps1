# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Locate the object to remove in each stock photograph (stock-fetch.ps1's bench\stock\st-<tag>.jpg)
# by Select by Text, writing bench\stock\st-objects.json in robust-strokes.ps1's form. The names
# below are what the photographs actually show (the search hits were not all on subject).
# -Only a,b,c limits the subjects.
param([string]$Only = "")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$names = [ordered]@{
    glasses = "glasses"; watch = "mask"; necklace = "tiara"; earring = "earring"; ring = "ring"; tattoo = "tattoo";
    car = "car"; sign = "sign"; bird = "bird"; cup = "cup"; poster = "poster"; dog = "dog"; lamppost = "lamp post";
    beach = "person"; cat = "girl"; bicycle = "bicycle"; boat = "boat"; cone = "traffic cone"; umbrella = "umbrella";
    hat = "hat"; bench = "bench"; bottle = "bottle"; cow = "cow"; hydrant = "car"; kite = "kite"; motorbike = "motorcycle"
}
$want = @($Only -split ',' | Where-Object { $_ -ne "" })
$objects = [ordered]@{}
$existing = Join-Path $out "st-objects.json"
if ((Test-Path $existing) -and $want.Count -gt 0) {
    $prev = Get-Content $existing -Raw | ConvertFrom-Json
    foreach ($p in $prev.PSObject.Properties) { $objects[$p.Name] = $p.Value }
}
foreach ($tag in $names.Keys) {
    if ($want.Count -gt 0 -and $want -notcontains $tag) { continue }
    $name = $names[$tag]
    $env:CM_IMAGE = Join-Path $out "st-$tag.jpg"
    if (-not (Test-Path $env:CM_IMAGE)) { "{0,-10} no file" -f $tag; continue }
    $env:CM_TXT = '{\"prompt\":\"' + $name + '\"}'
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_TXT%"
    $found = $null
    foreach ($l in $lines) { if ("$l" -match '"instances"') { $found = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $found -or $found.count -lt 1) {
        "{0,-10} {1,-13} not found" -f $tag, $name
        $objects[$tag] = $null
    } else {
        $i = $found.instances[0]
        "{0,-10} {1,-13} bounds {2} px {3} ({4} instance(s))" -f $tag, $name, ($i.bounds -join ","), $i.pixels, $found.count
        $objects[$tag] = @{ name = $name; bounds = @($i.bounds); pixels = $i.pixels }
    }
}
$objects | ConvertTo-Json -Depth 4 | Out-File $existing -Encoding utf8
