# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The fill's own settings against realism: eight whole-object strokes (six stock scenes, two
# generated) under each variant, compared with the shipped default (the hole smoothed, Lightning
# 8, euler/simple, 0.75 MP: the -smooth run). Variants: the request at 1.0 and 1.5 MP; the
# res_multistep sampler; the beta scheduler; the 40-step base tier at CFG 4; the base tier at
# 1.0 MP. Output: <prefix>-<tag>-whole-<variant>.png and the sheets tune-<variant>-{st,rb}.png.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$stTags = "dog,watch,cow,bicycle,hat,kite"
$rbTags = "watch,mug"
$variants = @(
    @("1mp",     "balanced", "maxPixels=1048576"),
    @("15mp",    "balanced", "maxPixels=1572864"),
    @("resms",   "balanced", "sampler=res_multistep"),
    @("beta",    "balanced", "scheduler=beta"),
    @("best",    "best",     ""),
    @("best1mp", "best",     "maxPixels=1048576")
)
foreach ($v in $variants) {
    $tag = $v[0]; $q = $v[1]; $extra = $v[2]
    # (an empty argument is dropped by the native call and -Extra would swallow -Tags: pass a no-op)
    if ($extra -eq "") { $extra = "variations=2" }
    "== $tag"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-$tag" -Kinds whole -Quality $q -Extra $extra -Tags $stTags 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-$tag" -Kinds whole -Quality $q -Extra $extra -Tags $rbTags 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-$tag" -Base "-smooth" -Compare whole -PerSheet 6 -Tags $stTags 2>&1 | Out-Null
    Copy-Item "$st\st-sheet-1.png" "$st\tune-$tag-st.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-$tag" -Base "-smooth" -Compare whole -PerSheet 2 -Tags $rbTags 2>&1 | Out-Null
    Copy-Item "$rb\rb-sheet-1.png" "$rb\tune-$tag-rb.png"
}
