# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Why does a fresh load sometimes thrash? Purge, then run the 1088x704 edit in a fresh process,
# three times, reading ComfyUI's own log (model loads, partial loads) after each.
param([string]$Flags = "--fast fp8_matrix_mult", [int]$Runs = 3)
$sp = $PSScriptRoot
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $sp "restart-comfyui.ps1") -Flags $Flags
$log = (Get-Content "C:\Users\5090\ComfyUI\current-log.txt" -Raw).Trim()
$err = "$log.err"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:TP_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\user-drawing.png"
$env:TP_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\thrash-probe.png"
function LogSince([ref]$pos) {
    $lines = @()
    foreach ($f in @($log, $err)) {
        if (Test-Path $f) { $all = Get-Content $f; if ($all.Count -gt $pos.Value) { $lines += $all[$pos.Value..($all.Count - 1)] }; $pos.Value = $all.Count }
    }
    $lines | Where-Object { $_ -match "Requested to load|loaded completely|loaded partially|lowvram|Unloading|unload|memory|Prompt executed|gc|OOM|Error" } | ForEach-Object { "    log: $_" }
}
$pos = 0
for ($i = 1; $i -le $Runs; $i++) {
    Invoke-WebRequest -Uri "http://127.0.0.1:8188/free" -Method Post -ContentType "application/json" -Body '{"unload_models": true, "free_memory": true}' -UseBasicParsing -TimeoutSec 60 | Out-Null
    Start-Sleep -Seconds 2
    $s = (Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    "run {0}: before, vram free {1:N0} MB" -f $i, ($s.devices[0].vram_free / 1MB)
    $env:TP_PARAMS = '{\"prompt\":\"turn this sketch of a face into a realistic photo of a human face\",\"seed\":' + $i + '}'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lines = & $cli --% run %TP_IMAGE% --cmd generate.edit --params "%TP_PARAMS%" --out %TP_OUT%
    $sw.Stop()
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "run {0}: {1} ms (sampling {2} ms, queue {3} ms, wall {4} ms) sent {5}x{6}" -f $i, $r.ms, $r.timings[0].runMs, $r.timings[0].queueMs, $sw.ElapsedMilliseconds, $r.requestWidth, $r.requestHeight } }
    $s = (Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    "run {0}: after, vram free {1:N0} MB, torch free {2:N0} MB" -f $i, ($s.devices[0].vram_free / 1MB), ($s.devices[0].torch_vram_free / 1MB)
    LogSince ([ref]$pos)
}
