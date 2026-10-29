# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Removal through Qwen-Image-2.1 as an edit by instruction (AusBoss's Cut Out + Remove recipe,
# research/notes/ausboss-catalogue-2026-10.md 2.2): `qwen-2.1/edit` as the removal template
# ("Remove the <name> …" on the crop, no mask in the model's view, the layer mask confining
# the change, the result realigned), on the native worker, one candidate a stroke; without and
# with AusBoss's Consistency LoRA. Against the 2511 one-candidate run (-one).
# -Prefill false shows the model the real picture (AusBoss's way; the smoothed hole came back
# as blur, 2026-10-26); -Thinker comfy names the object with ComfyUI's 4B (the worker's 8B
# named the person, not the thing); -MaxPixels caps the crop (the worker paged at 1 MP).
param(
    [string]$Tags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike",
    [string]$Kinds = "half,whole",
    [string]$Prefill = "false",
    [string]$Thinker = "comfy",
    [int]$MaxPixels = 786432,
    [string]$Tag = "e21",
    # Only the pass without the LoRA.
    [switch]$Plain
)
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
if ($Thinker -eq "comfy") { $env:PHOTOCRAFT_THINKER = "comfy" } else { $env:PHOTOCRAFT_THINKER = $null }
. "$scripts\native-worker.ps1"
if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; exit 1 }
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch { "(ComfyUI free: $_)" }
$extra = "variations=1;prefill=$Prefill;maxPixels=$MaxPixels"
$variants = @(@("-$Tag", ""), @("-${Tag}c", "qwen-image-2.1-consistency.safetensors:1.0"))
if ($Plain) { $variants = @(, $variants[0]) }
foreach ($variant in $variants) {
    $suffix, $loras = $variant
    "=== $suffix (loras: '$loras', $extra, thinker $Thinker) $(Get-Date -Format HH:mm:ss)"
    # (an empty `-Loras ''` is dropped by the shell and leaves the parameter without its value:
    # pass it only when there is one)
    $more = if ($loras) { @("-Loras", $loras) } else { @() }
    & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Tags $Tags -Kinds $Kinds -Suffix $suffix -Extra $extra -Purge heavy -Engine auto -Template qwen-2.1/edit @more 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarted" } | ForEach-Object { "$_" -replace ': Remove the (.{0,40}).*', ': $1' }
}
& "$scripts\native-worker.ps1" -Stop
"=== sheets"
foreach ($s in ($variants | ForEach-Object { $_[0] })) {
    foreach ($c in "half", "whole") {
        & powershell -NoProfile -File "$scripts\robust-sheets.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix $s -Base "-one" -Compare $c -PerSheet 13 -Tags $Tags 2>$null | Where-Object { "$_" -match "sheet" }
        Copy-Item "$st\st-sheet-1.png" "$st\st$s-$c.png" -ErrorAction SilentlyContinue
    }
}
"=== done $(Get-Date -Format HH:mm:ss)"
