# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [int]$Port = 7882,
    [string]$Work = "$env:TEMP\pex-variations-live",
    [int]$Variations = 4
)
# Plan Phase 4 acceptance, live: the release GUI (its own config folder under $Work) runs a
# Generative Fill with four variations in the background; the Variations strip shows four
# placeholders while it runs (and no modal progress dialog), then four tiles; clicking the third tile (ui.click on the rect
# `ui.inspect.variations` reports) shows it in one undo step, and Keep leaves one layer. Needs
# ComfyUI on 127.0.0.1:8188 and edit-stormy.png in $Images.
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
function Shot([string]$name) { $null = Call "ui.screenshot" @{ path = "$name.png" }; Write-Host "  screenshot: $(Join-Path $Work "$name.png")" }
function Run0 { ((Call "ui.inspect" @{}).variations.runs | Select-Object -First 1) }
function Layers { @((Call "ui.inspect" @{}).document.layers).Count }
function Click-Rect($r) {
    $null = Call "ui.click" @{ x = $r[0] + $r[2] / 2; y = $r[1] + $r[3] / 2 }
    Start-Sleep -Milliseconds 600
}

$app = Start-App
try {
    $null = Call "app.open" @{ path = "edit-stormy.png" }
    # Open water left of the rocks, below the horizon.
    $null = Exec "select.rect" @{ x = 40; y = 600; width = 260; height = 150 }

    Write-Host "1. A fill with $Variations variations, in the background"
    $t0 = Get-Date
    $r = Call "engine.execute" @{ id = "generate.fill"; params = @{ prompt = "a small red sailboat"; variations = $Variations }; wait = $false }
    Start-Sleep -Milliseconds 800
    $run = Run0
    Start-Sleep -Milliseconds 1500  # past the modal dialog's delay
    $run = Run0
    Write-Host ("  strip: {0}, {1} placeholders (the screenshot shows no modal progress dialog)" -f $run.status, @($run.pending).Count)
    Shot "phase4-1-pending"
    $deadline = (Get-Date).AddSeconds(600)
    while ((Get-Date) -lt $deadline) {
        $run = Run0
        if ($run.status -ne "running") { break }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Milliseconds 800
    $run = Run0
    Write-Host ("  done in {0:N1} s: {1}, {2} tiles, showing {3}" -f ((Get-Date) - $t0).TotalSeconds, $run.status, @($run.tiles).Count, $run.shown)
    Shot "phase4-2-tiles"

    Write-Host "2. Click the third tile"
    $before = Layers
    Click-Rect ($run.tiles | Where-Object { $_.index -eq 2 }).rect
    $run = Run0
    Write-Host ("  showing {0} of {1}" -f $run.shown, @($run.layers).Count)
    Shot "phase4-3-third"

    Write-Host "3. Keep it"
    $keep = $run.keep | Select-Object -First 1
    Click-Rect $keep.rect
    $run = Run0
    $after = Layers
    Write-Host ("  layers {0} -> {1}; the run has {2} left, showing {3}" -f $before, $after, @($run.layers).Count, $run.shown)
    Shot "phase4-4-kept"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
