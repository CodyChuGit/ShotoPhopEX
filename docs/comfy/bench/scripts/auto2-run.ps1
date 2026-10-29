# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Both sets under the shipped defaults after the adapter landed (auto: the adapter's tight red
# hole as the first variation, the smoothed plain fill as the second, the judge picking),
# compared with the smoothed-only run. Output: <prefix>-<tag>-{half,whole}-auto2.png and the
# sheets st-auto2-{half,whole}-{1,2}.png in bench\stock, rb-auto2-{half,whole}.png in bench.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-auto2" 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-auto2" 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-auto2" -Base "-smooth" -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st-auto2-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st-auto2-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-auto2" -Base "-smooth" -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb-auto2-$c.png"
}
