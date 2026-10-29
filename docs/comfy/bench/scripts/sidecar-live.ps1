# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [string]$ComfyPath = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable",
    [int]$Port = 7886,
    [string]$Work = "$env:TEMP\pex-sidecar-live"
)
# The app-managed ComfyUI (the sidecar, plan-native-engine §5.4), live: with no ComfyUI running,
# the release GUI (its own config folder under $Work) gets the ComfyUI folder preference, reports
# the engine stopped, starts it hidden (generate.engine.start), reaches ready, runs a Select
# Subject through it (which would have failed before), stops it, and a generative command with
# the engine stopped starts it by itself. Screenshots land in $Work. Stop any ComfyUI on 8188
# before running (the script refuses otherwise).
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
function Engine { Exec "generate.engine.status" }
function Wait-Engine([string]$state, [int]$seconds = 120) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $e = Engine
        if ($e.state -eq $state) { return $e }
        Start-Sleep -Milliseconds 500
    }
    return (Engine)
}
function Selection { (Inspect).document.selectionBounds }

try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 2; throw "a ComfyUI already answers on 8188; stop it first" } catch [System.Net.WebException] {}

$app = Start-App
try {
    Write-Host "1. No folder set: the engine is unconfigured; a folder makes it stopped"
    Write-Host ("  {0}" -f (Engine).state)
    $null = Exec "prefs.set" @{ path = "integrations.comfyPath"; value = $ComfyPath }
    $e = Engine
    Write-Host ("  {0} (url {1}, log {2})" -f $e.state, $e.url, $e.log)
    Shot "sidecar-1-stopped"

    Write-Host "2. Start: hidden child, ready within the timeout"
    $t0 = Get-Date
    $e = Exec "generate.engine.start"
    Write-Host ("  after start: {0} pid {1}" -f $e.state, $e.pid)
    $e = Wait-Engine "ready"
    Write-Host ("  {0} after {1:N1} s ({2})" -f $e.state, ((Get-Date) - $t0).TotalSeconds, $e.detail)
    $hidden = (Get-Process -Id $e.pid).MainWindowHandle -eq 0
    Write-Host ("  console window hidden: {0}" -f $hidden)
    Shot "sidecar-2-ready"

    Write-Host "3. A detector command through it"
    $null = Call "app.open" @{ path = "krea2-fox.png" }
    $r = Exec "select.subjectML" @{}
    Write-Host ("  select.subjectML: {0} instance(s), bounds {1}" -f $r.count, ((Selection) -join ","))

    Write-Host "4. Stop, then a generative command starts it by itself"
    $e = Exec "generate.engine.stop"
    Write-Host ("  after stop: {0}" -f $e.state)
    $null = Exec "select.deselect"
    $t0 = Get-Date
    $r = Exec "select.subjectML" @{}
    Write-Host ("  select.subjectML with the engine stopped: {0} instance(s) after {1:N1} s; engine {2}" -f $r.count, ((Get-Date) - $t0).TotalSeconds, (Engine).state)
    Shot "sidecar-4-autostart"

    Write-Host "5. Quit stops it (engineStopOnQuit)"
    $pid2 = (Engine).pid
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
if ($pid2) {
    $alive = $null -ne (Get-Process -Id $pid2 -ErrorAction SilentlyContinue)
    Write-Host ("  engine process {0} alive after the app was killed: {1} (a killed app cannot run on_exit; a normal quit stops it)" -f $pid2, $alive)
    if ($alive) { Stop-Process -Id $pid2 -ErrorAction SilentlyContinue }
}
