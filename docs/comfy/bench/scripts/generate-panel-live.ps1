# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [int]$Port = 7880,
    [string]$Work = "$env:TEMP\pex-generate-live"
)
# Plan Phase 2 acceptance, live: the release GUI (its own config folder under $Work) and the
# Generate panel driven through its own Generate path (ui.set generatePanel + Ctrl+Enter in its
# prompt): a 2511 edit with two layer references, an image from a layer reference into a new
# document, and a fill whose time is compared with the estimate the button showed. Needs
# ComfyUI on 127.0.0.1:8188 and the bench images (krea2-fox.png, edit-stormy.png,
# transparent-compass.png) in $Images.
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
function Panel { (Call "ui.inspect" @{}).generatePanel }
function Shot([string]$name) { $null = Call "ui.screenshot" @{ path = "$name.png" }; Write-Host "  screenshot: $(Join-Path $Work "$name.png")" }

# Generate through the panel: set its state, focus the prompt, press Ctrl+Enter, wait for the job.
function Run-Panel($state) {
    $state.focus = $true
    $null = Call "ui.set" @{ generatePanel = $state }
    Start-Sleep -Milliseconds 600
    $p = Panel
    if ($p.blocked) { throw "the panel says: $($p.blocked)" }
    $eta = $p.eta
    Write-Host ("  {0} via {1}, button: '{2}'" -f $p.command, $p.effectiveTemplate, $p.etaLabel)
    $t0 = Get-Date
    $null = Call "ui.key" @{ key = "Enter"; command = $true }
    Start-Sleep -Milliseconds 400
    $deadline = (Get-Date).AddSeconds(600)
    while ((Get-Date) -lt $deadline) {
        $i = Call "ui.inspect" @{}
        if (-not $i.generatePanel.job -and $i.generations.running -eq 0) { break }
        Start-Sleep -Milliseconds 300
    }
    $secs = ((Get-Date) - $t0).TotalSeconds
    $row = (Exec "generate.jobs" @{ limit = 1 }).jobs | Select-Object -First 1
    Write-Host ("  done in {0:N1} s (estimate {1:N1} s, ratio {2:N2}); row: {3} {4} {5}" -f $secs, $eta, ($secs / [Math]::Max($eta, 0.1)), $row.status, $row.command, $row.template)
    if ($row.status -ne "done") { Write-Host "  failure: $($row.failure.kind): $($row.failure.message)" }
    return @{ secs = $secs; eta = $eta; row = $row }
}

$app = Start-App
try {
    Write-Host "Setup: a lighthouse picture with two hidden reference layers (fox, compass)"
    $null = Call "app.open" @{ path = "krea2-fox.png" }
    $null = Exec "select.all"; $null = Exec "edit.copy"
    $null = Call "app.open" @{ path = "edit-stormy.png" }
    $fox = (Exec "edit.paste").layer
    $null = Exec "layer.setProps" @{ layer = $fox; visible = $false; name = "Fox" }
    $null = Call "app.open" @{ path = "transparent-compass.png" }
    $null = Exec "select.all"; $null = Exec "edit.copy"
    $null = Exec "document.activate" @{ document = 1 }
    $compass = (Exec "edit.paste").layer
    $null = Exec "layer.setProps" @{ layer = $compass; visible = $false; name = "Compass" }
    try { $null = Exec "select.deselect" } catch {}

    Write-Host "1. Edit with two references"
    $edit = Run-Panel @{ mode = "edit"; prompt = "Put the fox from image 2 on the rocks in front of the lighthouse, holding the brass compass from image 3 in its mouth"; references = @(@{ layer = $fox }, @{ layer = $compass }); count = 1 }
    Shot "phase2-1-edit-two-refs"

    Write-Host "2. Image from a layer reference, into a new document"
    $docs = (Call "ui.inspect" @{}).views.Count
    $image = Run-Panel @{ mode = "image"; into = "document"; prompt = "A watercolor painting of the fox from image 1 sitting in a spring meadow"; references = @(@{ layer = $fox }); aspect = "3:2"; size = "1K"; count = 1 }
    Write-Host ("  documents: {0} -> {1}" -f $docs, (Call "ui.inspect" @{}).views.Count)
    Shot "phase2-2-image-ref"

    Write-Host "3. Fill from a selection, estimate against the clock"
    $null = Exec "document.activate" @{ document = 1 }
    $null = Exec "select.rect" @{ x = 120; y = 380; width = 260; height = 180 }
    $fill = Run-Panel @{ mode = "fill"; prompt = "a small red sailboat"; references = @(); count = 1 }
    $fill2 = Run-Panel @{ mode = "fill"; prompt = "a small green sailboat"; references = @(); count = 1 }
    Shot "phase2-3-fill"
    Write-Host ("Fill estimate vs warm run: {0:N1} s vs {1:N1} s ({2:P0} off)" -f $fill2.eta, $fill2.secs, ([Math]::Abs($fill2.secs - $fill2.eta) / [Math]::Max($fill2.eta, 0.1)))
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
