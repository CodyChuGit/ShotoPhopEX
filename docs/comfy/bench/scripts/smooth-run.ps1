# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Both sets with the hole smoothed over in the model's picture (prefill=true), then the
# comparison sheets: st-smooth-{half,whole}-{1,2}.png in bench\stock and rb-smooth-{half,whole}.png
# in bench. Output: the strokes' lines, then the sheet paths.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-smooth" -Extra "prefill=true" 2>&1 | Where-Object { "$_" -notmatch "flattened" }
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-smooth" -Extra "prefill=true" 2>&1 | Where-Object { "$_" -notmatch "flattened" }
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-smooth" -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st-smooth-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st-smooth-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-smooth" -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb-smooth-$c.png"
}
