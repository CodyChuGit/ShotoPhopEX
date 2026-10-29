# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [int]$Port = 7885,
    [string]$Work = "$env:TEMP\pex-quick-select-live"
)
# The Quick Selection tool's SAM 3.1 mode, live: the release GUI (its own config folder under
# $Work) with the fox picture; a stroke across the fox becomes one select.byPoint with the spots
# along the stroke; a second stroke adds; an Alt stroke subtracts the object under it. Screenshots
# land in $Work. Needs ComfyUI on 127.0.0.1:8188 with the SAM 3.1 checkpoint and krea2-fox.png
# in $Images.
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
function Selection { (Inspect).document.selectionBounds }
function LastJob { (Exec "generate.jobs" @{ limit = 1 }).jobs | Select-Object -First 1 }
# A stroke from (x0,y0) to (x1,y1) in document pixels, 20 samples, with the modifiers given.
function Stroke([double]$x0, [double]$y0, [double]$x1, [double]$y1, $mods = $null) {
    $events = @(@{ kind = "down"; x = $x0; y = $y0 })
    for ($i = 1; $i -lt 20; $i++) {
        $t = $i / 20.0
        $events += @{ kind = "move"; x = ($x0 + ($x1 - $x0) * $t); y = ($y0 + ($y1 - $y0) * $t) }
    }
    $events += @{ kind = "up"; x = $x1; y = $y1 }
    $p = @{ tool = "QuickSelection"; events = $events }
    if ($mods) { $p.modifiers = $mods }
    $null = Call "ui.pointer" $p
}
function Wait-Job([int]$seconds = 30) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $j = LastJob
        if ($j -and $j.status -ne "running" -and $j.status -ne "queued") { return $j }
        Start-Sleep -Milliseconds 200
    }
    return (LastJob)
}

$app = Start-App
try {
    $null = Call "app.open" @{ path = "krea2-fox.png" }
    $null = Call "ui.set" @{ tool = "QuickSelection" }
    Start-Sleep -Milliseconds 800

    Write-Host "1. A stroke across the fox's body"
    $t0 = Get-Date
    Stroke 420 450 620 650
    $j = Wait-Job
    Write-Host ("  {0}: {1} after {2:N1} s; bounds {3}" -f $j.command, $j.status, ((Get-Date) - $t0).TotalSeconds, ((Selection) -join ","))
    Shot "quick-1-stroke"

    Write-Host "2. A second stroke on the head adds"
    Stroke 480 300 560 380
    $j = Wait-Job
    Write-Host ("  {0}: {1}; bounds {2}" -f $j.command, $j.status, ((Selection) -join ","))
    Shot "quick-2-add"

    Write-Host "3. An Alt stroke over the tail subtracts the object under it"
    Stroke 260 780 340 860 @{ alt = $true }
    $j = Wait-Job
    Write-Host ("  {0}: {1}; bounds {2}" -f $j.command, $j.status, ((Selection) -join ","))
    Shot "quick-3-subtract"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
