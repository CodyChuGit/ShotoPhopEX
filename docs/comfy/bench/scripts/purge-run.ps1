# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The purge policy's cost: six whole-object strokes (four stock, two generated) under each of
# switch (today's default), heavy and never, timed; the first stroke of each policy warms the
# server, so the five after it are the number. Starts ComfyUI when it is not answering and
# stops it afterwards when it started it; one restart between policies so each starts cold.
# Output: the strokes' lines with wall times, then a per-policy summary; the images as
# <prefix>-<tag>-whole-p<policy>.png (compare with robust-sheets.ps1 -Suffix -pnever ...).
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$scripts = "docs\comfy\bench\scripts"
$st = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
$rb = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
function Up { try { $null = Invoke-RestMethod http://127.0.0.1:8188/system_stats -TimeoutSec 3; $true } catch { $false } }
function Stop-Server { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "ComfyUI_windows_portable" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 3 }
function Start-Server($tag) {
    Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden -RedirectStandardOutput "$rb\comfyui-purge-$tag.log" -RedirectStandardError "$rb\comfyui-purge-$tag.err.log"
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Up) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (-not (Up)) { "ComfyUI did not come up"; exit 1 }
}
$wasUp = Up
$summary = @()
foreach ($policy in "switch", "heavy", "never") {
    if (Up) { Stop-Server }
    Start-Server $policy
    "== $policy"
    $lines = @()
    $lines += & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $st -Prefix st -Objects st-objects.json -Suffix "-p$policy" -Kinds whole -Purge $policy -Tags "dog,watch,cow,hat" 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarts" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
    $lines += & powershell -NoProfile -File "$scripts\robust-strokes.ps1" -Dir $rb -Prefix rb -Objects rb-objects.json -Suffix "-p$policy" -Kinds whole -Purge $policy -Tags "watch,mug" 2>&1 | Where-Object { "$_" -match "ms grow|FAILED|restarts" } | ForEach-Object { "$_" -replace ': Remove the (.{0,24}).*', ': $1' }
    $lines
    $ms = @($lines | ForEach-Object { if ("$_" -match '\s(\d+) ms grow') { [int]$Matches[1] } })
    if ($ms.Count -gt 1) {
        $warm = $ms[1..($ms.Count - 1)]
        $summary += "{0,-7} first {1,6} ms, then {2} strokes: mean {3,6} ms, min {4,6}, max {5,6}" -f $policy, $ms[0], $warm.Count, [int]($warm | Measure-Object -Average).Average, ($warm | Measure-Object -Minimum).Minimum, ($warm | Measure-Object -Maximum).Maximum
    }
}
"== summary"
$summary
if (-not $wasUp) { Stop-Server; "ComfyUI stopped" }
