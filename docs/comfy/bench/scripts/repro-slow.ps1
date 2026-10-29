# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# 1. Does our Krea 2 template run on this server? 2. Reproduce the slow fresh load: the edit at
# 1088x704 in three separate processes (the CLI purges itself when the card is nearly full),
# with the server log read after each.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$log = (Get-Content "C:\Users\5090\ComfyUI\current-log.txt" -Raw).Trim() + ".err"
$pos = (Get-Content $log).Count
function LogSince([ref]$pos) {
    $all = Get-Content $log
    if ($all.Count -gt $pos.Value) { $all[$pos.Value..($all.Count - 1)] | Where-Object { $_ -match "Requested to load|loaded completely|loaded partially|lowvram|unloaded|Prompt executed|Error|OOM" } | ForEach-Object { "    log: " + ($_ -replace "\x1b\[[0-9;]*m", "") } }
    $pos.Value = $all.Count
}
$env:RS_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\user-drawing.png"
$env:RS_IMG_PARAMS = '{\"prompt\":\"a red fox sitting in autumn leaves, photo\",\"width\":1024,\"height\":1024,\"seed\":3,\"target\":\"document\"}'
$env:RS_IMG_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\krea2-fox.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %RS_IMAGE% --cmd generate.image --params "%RS_IMG_PARAMS%" --out %RS_IMG_OUT%
foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "krea2 image: {0} ms (wall {1} ms) via {2}" -f $r.ms, $sw.ElapsedMilliseconds, $r.template } elseif ("$l" -match "rror") { "krea2 image: $l" } }
LogSince ([ref]$pos)
for ($i = 1; $i -le 3; $i++) {
    $s = (Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    "edit {0}: before, vram free {1:N0} MB" -f $i, ($s.devices[0].vram_free / 1MB)
    $env:RS_PARAMS = '{\"prompt\":\"turn this sketch of a face into a realistic photo of a human face\",\"seed\":' + (10 + $i) + '}'
    $env:RS_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\repro-slow.png"
    $sw.Restart()
    $lines = & $cli --% run %RS_IMAGE% --cmd generate.edit --params "%RS_PARAMS%" --out %RS_OUT%
    foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "edit {0}: {1} ms (sampling {2} ms, queue {3} ms, wall {4} ms)" -f $i, $r.ms, $r.timings[0].runMs, $r.timings[0].queueMs, $sw.ElapsedMilliseconds } elseif ("$l" -match "rror") { "edit {0}: {1}" -f $i, $l } }
    LogSince ([ref]$pos)
}
