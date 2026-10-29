# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [int]$Port = 7879,
    [string]$Work = "$env:TEMP\pex-generations-live",
    [switch]$KeepOpen
)
# Plan Phase 1 acceptance, live: the release GUI (its own config folder under $Work) runs a fill,
# an edit and an image against the local ComfyUI as background jobs; the Generations panel shows
# them running then completed with thumbnails; Place brings a result back after its document
# closed; a restart keeps the list; a dead server URL gives a "serverUnreachable" row.
# Screenshots land in $Work. Needs ComfyUI on 127.0.0.1:8188.
$ErrorActionPreference = "Stop"
$config = Join-Path $Work "config"
$token = Join-Path $Work "control.token"
New-Item -ItemType Directory -Force $config | Out-Null

function Start-App {
    $env:PHOTOCRAFT_CONFIG_DIR = $config
    $p = Start-Process -FilePath $Exe -ArgumentList @("--control", "$Port", "--control-token-file", "`"$token`"", "--automation-write-root", "`"$Work`"") -PassThru
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $Port)
            $s = $c.GetStream()
            $script:reader = New-Object System.IO.StreamReader($s)
            $script:writer = New-Object System.IO.StreamWriter($s)
            $script:writer.AutoFlush = $true
            $script:client = $c
            $t = (Get-Content $token -Raw).Trim()
            $null = Call "auth" @{ token = $t }
            return $p
        } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "the app did not open its control port"
}

$script:n = 0
function Call([string]$method, $params = @{}) {
    $script:n++
    $line = (@{ id = $script:n; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 20)
    $script:writer.WriteLine($line)
    while ($true) {
        $reply = $script:reader.ReadLine() | ConvertFrom-Json
        if ($reply.id -eq $script:n) { break }
    }
    if (-not $reply.ok) { throw "$method failed: $($reply.error)" }
    return $reply.result
}

function Exec([string]$id, $params = @{}, [bool]$wait = $true) {
    return Call "engine.execute" @{ id = $id; params = $params; wait = $wait }
}

function Gen { (Call "ui.inspect" @{}).generations }

function Wait-Idle([int]$seconds = 300) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $g = Gen
        if ($g.running -eq 0) { return $g }
        Start-Sleep -Milliseconds 500
    }
    throw "jobs still running after $seconds s"
}

function Shot([string]$name) {
    # Relative to the automation write root ($Work).
    $null = Call "ui.screenshot" @{ path = "$name.png" }
    Write-Host "  screenshot: $(Join-Path $Work "$name.png")"
}

$app = Start-App
try {
    Write-Host "1. fill, edit and image as background jobs"
    $null = Exec "file.new" @{ width = 1024; height = 768 }
    $null = Exec "edit.fill" @{ color = "#6c8fb3" }
    $null = Exec "select.rect" @{ x = 360; y = 240; width = 300; height = 260 }
    $t0 = Get-Date
    $null = Exec "generate.fill" @{ prompt = "a red kite"; seed = 11 } $false
    Start-Sleep -Milliseconds 400
    $g = Gen
    Write-Host ("  while running: running={0} completed={1}" -f $g.running, $g.completed)
    Shot "1-running"
    $g = Wait-Idle
    Write-Host ("  fill done in {0:N1} s: completed={1} thumbnails={2}" -f ((Get-Date) - $t0).TotalSeconds, $g.completed, $g.thumbnails)
    $t0 = Get-Date
    $null = Exec "generate.edit" @{ prompt = "make it sunset"; seed = 12 } $false
    $g = Wait-Idle
    Write-Host ("  edit done in {0:N1} s: completed={1}" -f ((Get-Date) - $t0).TotalSeconds, $g.completed)
    $t0 = Get-Date
    $null = Exec "generate.image" @{ prompt = "a lighthouse on a cliff at dawn"; width = 768; height = 512; seed = 13; target = "document" } $false
    $g = Wait-Idle
    Write-Host ("  image done in {0:N1} s: completed={1} thumbnails={2}" -f ((Get-Date) - $t0).TotalSeconds, $g.completed, $g.thumbnails)
    Start-Sleep -Seconds 2
    Shot "2-completed"
    $jobs = (Exec "generate.jobs" @{}).jobs
    $jobs | ForEach-Object { Write-Host ("  row {0}: {1} {2} '{3}' files={4}" -f $_.id, $_.status, $_.command, $_.prompt, $_.result.files.Count) }

    Write-Host "2. Place after the documents closed"
    $null = Exec "file.closeAll" @{}
    $fill = $jobs | Where-Object { $_.command -eq "generate.fill" } | Select-Object -First 1
    $r = Exec "generate.place" @{ id = $fill.id; target = "document" }
    Write-Host ("  placed as document {0}: {1}x{2}" -f $r.document, $r.width, $r.height)
    Shot "3-placed"

    Write-Host "3. a dead server URL"
    $null = Exec "prefs.set" @{ path = "integrations.comfyServer"; value = "http://127.0.0.1:9" }
    $null = Exec "select.rect" @{ x = 10; y = 10; width = 50; height = 50 }
    try { $null = Exec "generate.fill" @{ prompt = "a boat" } $true } catch { Write-Host "  failed as expected" }
    $failed = (Exec "generate.jobs" @{ status = "failed" }).jobs | Select-Object -First 1
    Write-Host ("  failed row: kind={0}" -f $failed.failure.kind)
    $null = Exec "prefs.set" @{ path = "integrations.comfyServer"; value = "http://127.0.0.1:8188" }
    Start-Sleep -Seconds 1
    Shot "4-failed"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "4. a restart keeps the list"
$app = Start-App
try {
    Start-Sleep -Seconds 2
    $all = Exec "generate.jobs" @{}
    Write-Host ("  after restart: total={0}, rows: {1}" -f $all.total, (($all.jobs | ForEach-Object { "$($_.status):$($_.command)" }) -join ", "))
    Shot "5-restarted"
} finally {
    if ($script:client) { $script:client.Close() }
    if (-not $KeepOpen) { Stop-Process -Id $app.Id -ErrorAction SilentlyContinue }
}
