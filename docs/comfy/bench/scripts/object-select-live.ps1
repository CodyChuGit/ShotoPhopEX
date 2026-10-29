# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [int]$Port = 7884,
    [string]$Work = "$env:TEMP\pex-object-select-live",
    # -Native: SAM 3.1 through the native worker (plan-native-engine Phase 1): the session gets
    # the runtime and model folders (`integrations.runtimePath` / `modelsPath`, `engine` stays
    # `auto`), so ComfyUI may be off.
    [switch]$Native,
    [string]$RuntimePath = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib",
    [string]$ModelsPath = "C:/Users/5090/Documents/_eval/native-models"
)
# The Object Selection tool's SAM 3.1 mode, live: the release GUI (its own config folder under
# $Work) with the fox picture; the object finder's probe appears when the pointer rests over the
# fox (ui.move, then ui.inspect.objectSelect.probe); a click applies it without a new detection;
# a box around the fox's head selects through the box prompt; Refine mode excludes a spot;
# Select › Subject runs the detector. Screenshots land in $Work. Needs ComfyUI on 127.0.0.1:8188
# with the SAM 3.1 checkpoint and krea2-fox.png in $Images, or -Native with the ONNX graphs.
$ErrorActionPreference = "Stop"
$config = Join-Path $Work "config"
$token = Join-Path $Work "control.token"
New-Item -ItemType Directory -Force $config | Out-Null
$Images = (Resolve-Path $Images).Path

function Start-App {
    $env:PHOTOCRAFT_CONFIG_DIR = $config
    $p = Start-Process -FilePath $Exe -ArgumentList @("--control", "$Port", "--control-token-file", "`"$token`"", "--automation-read-root", "`"$Images`"", "--automation-write-root", "`"$Work`"") -PassThru
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $Port)
            $s = $c.GetStream()
            $script:reader = New-Object System.IO.StreamReader($s)
            $script:writer = New-Object System.IO.StreamWriter($s)
            $script:writer.AutoFlush = $true
            $script:client = $c
            $null = Call "auth" @{ token = (Get-Content $token -Raw).Trim() }
            return $p
        } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "the app did not open its control port"
}

$script:n = 0
function Call([string]$method, $params = @{}) {
    $script:n++
    $script:writer.WriteLine((@{ id = $script:n; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 20))
    while ($true) {
        $reply = $script:reader.ReadLine() | ConvertFrom-Json
        if ($reply.id -eq $script:n) { break }
    }
    if (-not $reply.ok) { throw "$method failed: $($reply.error)" }
    return $reply.result
}
function Exec([string]$id, $params = @{}) { Call "engine.execute" @{ id = $id; params = $params; wait = $true } }
function Inspect { Call "ui.inspect" @{} }
function Shot([string]$name) { $null = Call "ui.screenshot" @{ path = "$name.png" }; Write-Host "  screenshot: $(Join-Path $Work "$name.png")" }
# Document → screen points: the view as ui.inspect reports it (zoom, center, the canvas rect).
function Doc-To-Screen([double]$x, [double]$y) {
    $i = Inspect
    $v = $i.views[$i.activeDocument]
    $r = $i.canvasRect
    $cx = $r[0] + $r[2] / 2; $cy = $r[1] + $r[3] / 2
    return @(($cx + ($x - $v.center[0]) * $v.zoom), ($cy + ($y - $v.center[1]) * $v.zoom))
}
function Wait-Probe([int]$seconds = 15) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $o = (Inspect).objectSelect
        if ($o.probe) { return $o }
        Start-Sleep -Milliseconds 150
    }
    return (Inspect).objectSelect
}
function Selection { (Inspect).document.selectionBounds }
function Prompts { (Exec "generate.jobs" @{ limit = 50; status = "done" }).jobs | Where-Object { $_.command -like "select.*" } }

$app = Start-App
try {
    if ($Native) {
        $null = Exec "prefs.set" @{ path = "integrations.runtimePath"; value = $RuntimePath }
        $null = Exec "prefs.set" @{ path = "integrations.modelsPath"; value = $ModelsPath }
        $h = Exec "generate.health" @{ template = "sam3.1/segment-point" }
        Write-Host ("0. SAM 3.1 backend: {0} ({1})" -f $h.backend, $h.device)
    }
    $null = Call "app.open" @{ path = "krea2-fox.png" }
    $null = Call "ui.set" @{ tool = "ObjectSelection" }
    Start-Sleep -Milliseconds 800

    Write-Host "1. The object finder: rest the pointer on the fox"
    $t0 = Get-Date
    $at = Doc-To-Screen 480 520
    $null = Call "ui.move" @{ x = $at[0]; y = $at[1] }
    $o = Wait-Probe
    Write-Host ("  probe after {0:N1} s: {1}" -f ((Get-Date) - $t0).TotalSeconds, ($o.probe | ConvertTo-Json -Compress))
    Shot "sam-1-finder"

    Write-Host "2. Click: the probe applies, no new detection"
    $before = @(Prompts).Count
    $null = Call "ui.click" @{ x = $at[0]; y = $at[1] }
    Start-Sleep -Milliseconds 800
    $sel = Selection
    $o = (Inspect).objectSelect
    Write-Host ("  selection bounds {0}; last query {1}; detections {2} -> {3}" -f ($sel -join ","), ($o.last | ConvertTo-Json -Compress), $before, @(Prompts).Count)
    Shot "sam-2-clicked"

    Write-Host "3. Refine: exclude the tail with Alt-click"
    $null = Call "ui.set" @{ toolOptions = @{ object_refine = $true } }
    $tail = Doc-To-Screen 300 820
    $null = Call "ui.click" @{ x = $tail[0]; y = $tail[1]; modifiers = @{ alt = $true } }
    Start-Sleep -Milliseconds 2500
    $o = (Inspect).objectSelect
    Write-Host ("  refined: {0}; bounds {1}" -f ($o.last | ConvertTo-Json -Compress), ((Selection) -join ","))
    Shot "sam-3-refined"
    $null = Call "ui.set" @{ toolOptions = @{ object_refine = $false } }

    Write-Host "4. A box around the head"
    $a = Doc-To-Screen 330 250; $b = Doc-To-Screen 760 640
    $null = Call "ui.pointer" @{ tool = "ObjectSelection"; events = @(@{ kind = "down"; x = 330; y = 250 }, @{ kind = "move"; x = 600; y = 500 }, @{ kind = "up"; x = 760; y = 640 }) }
    Start-Sleep -Milliseconds 2500
    Write-Host ("  box selection bounds {0}; last query {1}" -f ((Selection) -join ","), ((Inspect).objectSelect.last | ConvertTo-Json -Compress))
    Shot "sam-4-box"

    Write-Host "5. Select > Subject through the detector"
    $null = Exec "select.deselect"
    $null = Call "ui.menu.invoke" @{ id = "select.subject" }
    Start-Sleep -Milliseconds 2500
    $row = (Exec "generate.jobs" @{ limit = 1 }).jobs | Select-Object -First 1
    Write-Host ("  {0}: {1}; bounds {2}" -f $row.command, $row.status, ((Selection) -join ","))
    Shot "sam-5-subject"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
