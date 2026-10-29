# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Overnight 2026-10-27 Q3: the worker's chat with and without the key/value cache. For each CLI
# build (-Old: before genai-native's KvCache, -New: with it): the removal rewriter's question on
# the twenty naming-probe pictures (`naming-base`, seed 7: the answers must match word for word)
# and chat-smoke.ps1's three rewrites (a 120-token image prompt among them), with the worker's
# own chat timings (`chat` lines: tokens, prefill_s, chat_s).
param([string]$Old, [string]$New = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$pics = Get-ChildItem "$out\stock\naming-base" -Filter *.png | ForEach-Object { $_.FullName }
$probe = "C:\Users\5090\Documents\PhotoshopEX\target\release\examples\chat_probe.exe"
$prompt = "Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there"
try { $null = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/free -Body '{"unload_models":true,"free_memory":true}' -ContentType application/json -TimeoutSec 10 } catch {}
foreach ($run in @(@("old", $Old), @("new", $New))) {
    $tag, $cli = $run
    if (-not $cli) { continue }
    "=== $tag $(Get-Date -Format HH:mm:ss) $cli"
    $env:RUST_LOG = "info"
    $env:PHOTOCRAFT_CLI = $cli
    . "$scripts\native-worker.ps1" -Log "$out\q3-$tag-worker.log"
    if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker"; continue }
    # The first question loads the head; ask one to warm it before timing.
    & $probe "$scripts\removal-system.txt" $prompt 7 $pics[0] | Out-Null
    & $probe "$scripts\removal-system.txt" $prompt 7 @pics | Tee-Object -FilePath "$out\q3-$tag-probe.tsv"
    & "$scripts\native-worker.ps1" -Stop
    & powershell -NoProfile -File "$scripts\chat-smoke.ps1" -Cli $cli 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "==|ms\)|chat " }
}
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
"=== done $(Get-Date -Format HH:mm:ss)"
