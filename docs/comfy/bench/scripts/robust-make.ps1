# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# The robustness set for the generative removal: one scene per line below, each with one
# object to remove, made by the app's own text-to-image (generate.image, 1024², seed 31,
# Quality balanced), then the object located by Select by Text. Output: bench\rb-<tag>.png
# and bench\rb-objects.json ({tag: {name, bounds:[x,y,w,h], pixels}}), the input of
# robust-strokes.ps1. A scene whose object Select by Text cannot find is listed with null.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_NEW = '{\"width\":1024,\"height\":1024}'
$scenes = @(
    @("watch",    "wristwatch",  "close-up photograph of a man's forearm and hand resting on a wooden table, wearing a steel wristwatch with a black dial, natural window light, photorealistic"),
    @("glasses",  "glasses",     "head and shoulders portrait photograph of a middle-aged man with short grey hair wearing black-framed glasses, plain beige wall, soft light, photorealistic"),
    @("earring",  "earring",     "close-up profile portrait photograph of a woman with hair tied back wearing a large silver hoop earring, neutral grey background, photorealistic"),
    @("tattoo",   "tattoo",      "photograph of a woman's upper arm and shoulder with a small black rose tattoo, white tank top, white background, photorealistic"),
    @("logo",     "red logo",    "photograph of a young man wearing a plain white t-shirt with a large red circular logo printed on the chest, standing against a concrete wall, photorealistic"),
    @("car",      "blue car",    "photograph of a quiet suburban street with a single blue hatchback car parked by the curb, houses and trees behind, sunny day, photorealistic"),
    @("person",   "person",      "photograph of a wide empty beach at low tide with one distant person walking along the waterline, overcast sky, photorealistic"),
    @("lamppost", "lamp post",   "photograph of a park lawn with a single black iron lamp post, trees in the background, autumn, photorealistic"),
    @("sign",     "sign",        "photograph of a brick wall with a small rectangular blue street sign with white text mounted on it, photorealistic"),
    @("bird",     "bird",        "photograph of a single crow perched on a telephone wire against a pale sky, photorealistic"),
    @("mug",      "mug",         "top-down photograph of a white ceramic mug on a red and white checkered tablecloth, photorealistic"),
    @("ring",     "ring",        "close-up photograph of a woman's hand resting on linen fabric wearing a gold ring with a small diamond, photorealistic"),
    @("poster",   "poster",      "photograph of a weathered brick wall with a single white paper poster taped on it, photorealistic"),
    @("dog",      "dog",         "photograph of a golden retriever sitting on a green lawn in a garden, hedge behind, photorealistic")
)
$objects = [ordered]@{}
foreach ($s in $scenes) {
    $tag = $s[0]; $name = $s[1]; $prompt = $s[2]
    $env:CM_OUT = Join-Path $out "rb-$tag.png"
    if (-not (Test-Path $env:CM_OUT)) {
        $env:CM_GEN = '{\"prompt\":\"' + $prompt + '\",\"target\":\"document\",\"width\":1024,\"height\":1024,\"seed\":31}'
        $lines = & $cli --% run --new "%CM_NEW%" --cmd prefs.set --params "%CM_PREFS%" --cmd generate.image --params "%CM_GEN%" --out %CM_OUT%
        $made = $lines | Where-Object { $_ -match '"runId"' }
        if (-not $made) { "${tag}: image FAILED: " + ($lines | Select-Object -Last 1); continue }
    }
    $env:CM_IMAGE = $env:CM_OUT
    $env:CM_TXT = '{\"prompt\":\"' + $name + '\"}'
    $lines = & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_TXT%"
    $found = $null
    foreach ($l in $lines) { if ("$l" -match '"instances"') { $found = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $found -or $found.count -lt 1) {
        "{0,-9} {1,-11} not found" -f $tag, $name
        $objects[$tag] = $null
    } else {
        $i = $found.instances[0]
        "{0,-9} {1,-11} bounds {2} px {3} ({4} instance(s))" -f $tag, $name, ($i.bounds -join ","), $i.pixels, $found.count
        $objects[$tag] = @{ name = $name; bounds = @($i.bounds); pixels = $i.pixels }
    }
}
$objects | ConvertTo-Json -Depth 4 | Out-File (Join-Path $out "rb-objects.json") -Encoding utf8
