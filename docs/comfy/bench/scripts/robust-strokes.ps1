# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
# The robustness set's strokes (robust-make.ps1's rb-objects.json), through the tools' own
# commands, Lightning (Quality balanced), random seeds:
#   half   Spot Healing (generative), one stroke along the object's long axis over its first
#          half: the hard case (the rest of the object stays in view; the detector and the
#          rewriter's name have to take it whole)
#   whole  the Remove tool (generative), a serpentine stroke over the object's whole box
# -Tags a,b,c runs those scenes (default: all). Output: bench\rb-<tag>-half.png,
# rb-<tag>-whole.png, and a line per run with the rewriter's words, the growth and the time
# (appended to bench\rb-log.txt).
# -Dir, -Prefix and -Objects point it at another set (the stock photographs: -Dir bench\stock
# -Prefix st -Objects st-objects.json); the image is <Prefix>-<tag>.<any extension>.
# -Quality best runs the 40-step tier; -Suffix names the outputs apart (<Prefix>-<tag>-half<Suffix>.png);
# -Kinds half,whole picks the strokes.
# -Extra adds fields to every stroke's params (e.g. ',\"prefill\":true'); -Template names the fill
# template (research-only ones allowed for the measurement). -Purge switch|heavy|never sets
# integrations.purgeModels for the run.
# -Engine comfy|auto|native sets integrations.engine (auto and native need the app's runtime
# and model folders, given here as the owner's; the detector then runs natively too).
# -Loras "file.safetensors:1.0,other.safetensors:0.5" adds the LoRAs to every stroke (the
# native engine applies them). With a shared native worker (PHOTOCRAFT_NATIVE_ENGINE, from
# native-worker.ps1) a worker that stopped mid-run is started again and the stroke retried.
param([string]$Tags = "", [string]$Dir = "C:\Users\5090\ComfyUI\photocraft-tests\bench", [string]$Prefix = "rb", [string]$Objects = "rb-objects.json", [string]$Quality = "balanced", [string]$Suffix = "", [string]$Kinds = "half,whole", [string]$Extra = "", [string]$Template = "", [string]$Purge = "switch", [string]$Engine = "comfy", [string]$Loras = "")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = if ($env:PHOTOCRAFT_CLI) { $env:PHOTOCRAFT_CLI } else { (Resolve-Path "target\release\photocraft-cli.exe").Path }
$out = $Dir
$tpl = if ($Template -eq "") { "" } else { ',\"integrations.defaultFillTemplate\":\"' + $Template + '\",\"integrations.allowResearchModels\":true' }
# -Extra is key=value pairs separated by semicolons ("prefill=true;grow=20"): quotes do not
# survive the outer shell's argument passing, so they are added here.
$Extra = (($Extra -split ';' | Where-Object { $_ -match '^\s*([A-Za-z]\w*)\s*=\s*(.+?)\s*$' } | ForEach-Object { $k = $Matches[1]; $v = $Matches[2]; if ($v -match '^(true|false|-?\d+(\.\d+)?)$') { ',\"' + $k + '\":' + $v } else { ',\"' + $k + '\":\"' + $v + '\"' } }) -join '')
if ($Loras) {
    $items = $Loras -split ',' | Where-Object { $_ -match '^\s*([^:]+?)\s*(?::\s*(-?\d+(\.\d+)?))?\s*$' } | ForEach-Object {
        $s = if ($Matches[2]) { $Matches[2] } else { "1" }
        '{\"file\":\"' + $Matches[1] + '\",\"strength\":' + $s + '}'
    }
    $Extra += ',\"loras\":[' + ($items -join ',') + ']'
}
$folders = if ($Engine -eq "comfy") { "" } else { ',\"integrations.runtimePath\":\"C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib\",\"integrations.modelsPath\":\"C:/Users/5090/Documents/_eval/native-models\"' }
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"' + $Engine + '\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"' + $Quality + '\",\"integrations.purgeModels\":\"' + $Purge + '\"' + $tpl + $folders + '}}'
# (not `$kinds`: that is the [string] parameter, case ignored, and would turn the list into "half whole")
$kindList = @($Kinds -split ',' | Where-Object { $_ -ne "" })
# (not `$objects`: PowerShell variable names ignore case, and the [string] parameter would
# turn the parsed set back into a string)
$set = Get-Content (Join-Path $out $Objects) -Raw | ConvertFrom-Json
$want = @($Tags -split ',' | Where-Object { $_ -ne "" })
$log = Join-Path $out "$Prefix-log.txt"
$script:restarts = 0
function Pts($list) { "[" + (($list | ForEach-Object { "[$($_[0]),$($_[1])]" }) -join ",") + "]" }
function Invoke-Cli {
    & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd %CM_CMD% --params "%CM_PARAMS%" --cmd generate.info --params "{}" --out %CM_OUT%
}
function Tool($tag, $kind, $cmd, $params) {
    $image = Get-ChildItem (Join-Path $out "$Prefix-$tag.*") | Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' } | Select-Object -First 1
    $env:CM_IMAGE = $image.FullName; $env:CM_CMD = $cmd; $env:CM_PARAMS = $params; $env:CM_OUT = Join-Path $out "$Prefix-$tag-$kind$Suffix.png"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # (stderr merged and stringified at the call of Invoke-Cli, so a failed stroke's error
    # reaches the log; after `--%` the rest of the line goes to the CLI verbatim, so the native
    # call sits on its own line in its own function)
    $lines = Invoke-Cli 2>&1 | ForEach-Object { "$_" }
    # A server that died mid-run (the GPU driver resetting under model churn, 2026-10-23) is
    # started again, its output kept, and the stroke run once more.
    if (($lines -join " ") -match "cannot reach the ComfyUI server|No connection could be made|forcibly closed by the remote host|answered unexpectedly") {
        $script:restarts++
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden -RedirectStandardOutput "C:\Users\5090\ComfyUI\photocraft-tests\bench\comfyui-restart-$stamp.log" -RedirectStandardError "C:\Users\5090\ComfyUI\photocraft-tests\bench\comfyui-restart-$stamp.err.log"
        $deadline = (Get-Date).AddMinutes(4)
        while ((Get-Date) -lt $deadline) { try { $null = Invoke-RestMethod http://127.0.0.1:8188/system_stats -TimeoutSec 3; break } catch { Start-Sleep -Seconds 5 } }
        "(server restarted at $stamp, stroke retried)" | Out-File $log -Append -Encoding utf8
        $sw.Restart()
        $lines = Invoke-Cli 2>&1 | ForEach-Object { "$_" }
    }
    # A shared native worker that stopped (it exits on a poisoned CUDA context): a fresh one.
    if ($env:PHOTOCRAFT_NATIVE_ENGINE -and (($lines -join " ") -match "the engine stopped|the engine is gone|connect to port|CUDA_ERROR")) {
        $script:restarts++
        & "$PSScriptRoot\native-worker.ps1" -Stop | Out-Null
        . "$PSScriptRoot\native-worker.ps1" | Out-Null
        "(native worker restarted, stroke retried)" | Out-File $log -Append -Encoding utf8
        $sw.Restart()
        $lines = Invoke-Cli 2>&1 | ForEach-Object { "$_" }
    }
    $said = $null
    foreach ($l in $lines) { if ("$l" -match '"enhanced":"([^"]*)".*"grow":(\d+)') { $said = "{0,-9} {1,-5} {2,6} ms grow {3,3}: {4}" -f $tag, $kind, $sw.ElapsedMilliseconds, $Matches[2], $Matches[1] } }
    if ($null -eq $said) { $said = "{0,-9} {1,-5} FAILED: {2}" -f $tag, $kind, (($lines | Where-Object { $_ -notmatch "flattened" } | Select-Object -Last 1) -join " ") }
    $said; $said | Out-File $log -Append -Encoding utf8
}
foreach ($p in $set.PSObject.Properties) {
    $tag = $p.Name; $o = $p.Value
    if ($null -eq $o) { continue }
    if ($want.Count -gt 0 -and $want -notcontains $tag) { continue }
    $x = $o.bounds[0]; $y = $o.bounds[1]; $w = $o.bounds[2]; $h = $o.bounds[3]
    $cx = $x + [int]($w / 2); $cy = $y + [int]($h / 2)
    $s = [Math]::Round([Math]::Min([Math]::Max([Math]::Min($w, $h) / 2, 12), 64))
    # half: along the long axis, from 5 % to 50 % of it
    # (parenthesised sums: the comma binds tighter than + in PowerShell)
    if ($w -ge $h) { $half = @(@(($x + [int](0.05 * $w)), $cy), @(($x + [int](0.5 * $w)), $cy)) }
    else { $half = @(@($cx, ($y + [int](0.05 * $h))), @($cx, ($y + [int](0.5 * $h)))) }
    if ($kindList -contains "half") { Tool $tag "half" "paint.spotHealing" ('{\"type\":\"generative\",\"size\":' + $s + ',\"hardness\":1' + $Extra + ',\"points\":' + (Pts $half) + '}') }
    if ($kindList -notcontains "whole") { continue }
    # whole: a serpentine over the box, rows a brush apart
    $sw2 = [Math]::Round([Math]::Min([Math]::Max([Math]::Min($w, $h) / 2, 16), 80))
    $rows = New-Object System.Collections.ArrayList; $yy = $y + [int]($sw2 / 2); $i = 0
    $left = $x + [int]($sw2 / 2); $right = $x + $w - [int]($sw2 / 2)
    while ($yy -lt $y + $h -and $i -lt 14) {
        if ($i % 2 -eq 0) { [void]$rows.Add("[$left,$yy]"); [void]$rows.Add("[$right,$yy]") }
        else { [void]$rows.Add("[$right,$yy]"); [void]$rows.Add("[$left,$yy]") }
        $yy += [int](0.8 * $sw2); $i++
    }
    if ($rows.Count -lt 2) { [void]$rows.Add("[$x,$cy]"); [void]$rows.Add("[$($x + $w),$cy]") }
    Tool $tag "whole" "paint.remove" ('{\"generative\":true,\"size\":' + $sw2 + $Extra + ',\"points\":[' + ($rows -join ",") + ']}')
}
if ($script:restarts -gt 0) { "server restarts: $($script:restarts)" }
