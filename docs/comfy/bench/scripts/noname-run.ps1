# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# One candidate a stroke, with and without the rewriter's name: does a dab that skips the
# vision-language model (no "Naming the object", the detector on the point prompt only, the
# 2511 left resident between dabs) lose removals? Both sets, the adapter candidate
# (variations 1): -Suffix "-one" named (the default), -Suffix "-noname" with enhance=false.
# Output: <prefix>-<tag>-{half,whole}-{one,noname}.png, the per-stroke lines (time, growth,
# words) on stdout and in <prefix>-log.txt, then the sheets st-noname-{half,whole}-{1,2}.png
# in bench\stock and rb-noname-{half,whole}.png in bench (-noname against -one).
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
foreach ($variant in @(@("-one", "variations=1"), @("-noname", "enhance=false;variations=1"))) {
    $suffix, $extra = $variant
    "=== $suffix ($extra) $(Get-Date -Format HH:mm:ss)"
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $suffix -Extra $extra -Purge heavy 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix $suffix -Extra $extra -Purge heavy 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
}
"=== sheets $(Get-Date -Format HH:mm:ss)"
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-noname" -Base "-one" -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st-noname-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st-noname-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-noname" -Base "-one" -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb-noname-$c.png"
}
"=== done $(Get-Date -Format HH:mm:ss)"
