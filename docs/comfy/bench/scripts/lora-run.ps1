# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Both sets through the Object Remover v2 adapter (the qwen-edit-2511/remove-lightning-8 template:
# the hole painted red, the trigger sentence), compared with the shipped default (the hole
# smoothed, the -smooth run). Output: <prefix>-<tag>-{half,whole}-lora.png and the sheets
# st-lora-{half,whole}-{1,2}.png in bench\stock, rb-lora-{half,whole}.png in bench.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$tpl = "qwen-edit-2511/remove-lightning-8"
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-lora" -Template $tpl 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
& powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-lora" -Template $tpl 2>&1 | Where-Object { "$_" -match "ms grow|FAILED" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
foreach ($c in "half", "whole") {
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-lora" -Base "-smooth" -Compare $c -PerSheet 13 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$st\st-sheet-1.png" "$st\st-lora-$c-1.png"; Copy-Item "$st\st-sheet-2.png" "$st\st-lora-$c-2.png"
    & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-lora" -Base "-smooth" -Compare $c -PerSheet 14 2>&1 | Where-Object { "$_" -match "sheet" }
    Copy-Item "$rb\rb-sheet-1.png" "$rb\rb-lora-$c.png"
}
