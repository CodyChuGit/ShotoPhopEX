# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Cli = "$PSScriptRoot\..\..\..\..\target\release\photocraft-cli.exe",
    [int]$Port = 7883,
    [string]$Work = "$env:TEMP\pex-walkthrough"
)
# The plan's acceptance walkthrough (plan-artcraft-experience.md §9), live, through the control
# channel and the panel's own Generate path: the release GUI with its own config folder under
# $Work, ComfyUI on 127.0.0.1:8188. Screenshots land in $Work. Steps that need a human (a native
# file dialog, hearing the blip) are replaced by the state they leave: the Save is the CLI's
# `generate.export` into $Work\saved (automation may not choose folders), the alert is
# `ui.inspect.generations.lastAlert`.
$ErrorActionPreference = "Stop"
$config = Join-Path $Work "config"
$token = Join-Path $Work "control.token"
$saved = Join-Path $Work "saved"
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Force $config, $saved | Out-Null

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
function Click-Rect($r) { $null = Call "ui.click" @{ x = $r[0] + $r[2] / 2; y = $r[1] + $r[3] / 2 }; Start-Sleep -Milliseconds 700 }
function Wait-Idle([int]$seconds = 600) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $i = Inspect
        if ($i.generations.running -eq 0 -and -not $i.generatePanel.job) { return $i }
        Start-Sleep -Milliseconds 400
    }
    throw "still running after $seconds s"
}
# Generate through the panel: its state, the keyboard in its prompt, Ctrl+Enter.
function Panel-Generate($state) {
    $state.focus = $true
    $null = Call "ui.set" @{ generatePanel = $state }
    Start-Sleep -Milliseconds 500
    $p = (Inspect).generatePanel
    if ($p.blocked) { throw "the panel says: $($p.blocked)" }
    Write-Host ("  {0} via {1}, button '{2}'" -f $p.command, $p.effectiveTemplate, $p.etaLabel)
    $null = Call "ui.key" @{ key = "Enter"; command = $true }
    Start-Sleep -Milliseconds 500
}
function Last-Row { (Exec "generate.jobs" @{ limit = 1 }).jobs | Select-Object -First 1 }
function Run-Cli([string[]]$a) { & $Cli @a | Out-String }

$app = Start-App
try {
    Write-Host "1. Launch: the Essentials dock"
    $i = Inspect
    Write-Host ("  panels: generate={0} generations={1}" -f $i.panels.generate, $i.panels.generations)
    Shot "w1-launch"

    Write-Host "2. Image, Krea 2 Turbo, four variations, into a new document"
    $t0 = Get-Date
    Panel-Generate @{ mode = "image"; into = "document"; prompt = "a lighthouse at dawn, photo"; template = "krea2-turbo/image"; count = 4; aspect = "3:2"; size = "1K" }
    $i = Inspect
    Write-Host ("  while running: generations running={0}" -f $i.generations.running)
    Shot "w2a-running"
    $i = Wait-Idle
    Start-Sleep -Seconds 1
    $i = Inspect
    $run = $i.variations.runs | Select-Object -First 1
    Write-Host ("  done in {0:N1} s; document '{1}' with {2} layers; strip: {3} tiles" -f ((Get-Date) - $t0).TotalSeconds, $i.document.name, @($i.document.layers).Count, @($run.tiles).Count)
    $lighthouse = Last-Row
    Shot "w2b-four-tiles"
    Click-Rect (($run.tiles | Where-Object { $_.index -eq 1 }).rect)
    $run = (Inspect).variations.runs | Select-Object -First 1
    Write-Host ("  clicked the second tile: showing {0}" -f $run.shown)
    Click-Rect (($run.keep | Select-Object -First 1).rect)
    $i = Inspect
    Write-Host ("  kept it: {0} layers left in the document" -f @($i.document.layers).Count)
    Shot "w2c-kept"

    Write-Host "3. Fill a selection; the task bar stays away while the panel is open"
    $null = Exec "select.rect" @{ x = 620; y = 120; width = 220; height = 180 }
    $t0 = Get-Date
    Panel-Generate @{ mode = "fill"; prompt = "a red kite"; template = ""; count = 1 }
    $bar = @((Inspect).modelPickers | Where-Object { $_.id -eq "generative-bar" }).Count
    Write-Host ("  task bar on screen: {0}" -f ($bar -gt 0))
    $null = Wait-Idle
    $kite = Last-Row
    Write-Host ("  fill {0} in {1:N1} s via {2}" -f $kite.status, ((Get-Date) - $t0).TotalSeconds, $kite.template)
    Shot "w3-kite"

    Write-Host "4. Edit with the kite layer as a reference"
    try { $null = Exec "select.deselect" } catch {}
    $t0 = Get-Date
    Panel-Generate @{ mode = "edit"; prompt = "make the sky stormy, keep the kite"; references = @(@{ layer = $kite.result.layers[0] }); count = 1 }
    $null = Wait-Idle
    $edit = Last-Row
    Write-Host ("  edit {0} in {1:N1} s via {2}, references: {3}" -f $edit.status, ((Get-Date) - $t0).TotalSeconds, $edit.template, @($edit.params.references).Count)
    Shot "w4-stormy"

    Write-Host "5. Server unreachable, then Recreate"
    $null = Exec "prefs.set" @{ path = "integrations.comfyServer"; value = "http://127.0.0.1:9" }
    $null = Exec "select.rect" @{ x = 100; y = 400; width = 200; height = 150 }
    Panel-Generate @{ mode = "fill"; prompt = "a small boat"; template = ""; count = 1 }
    $null = Wait-Idle 60
    $failed = Last-Row
    Write-Host ("  row: {0} {1}" -f $failed.status, $failed.failure.kind)
    Shot "w5a-failed"
    $null = Exec "prefs.set" @{ path = "integrations.comfyServer"; value = "http://127.0.0.1:8188" }
    $null = Call "engine.execute" @{ id = "generate.recreate"; params = @{ id = $failed.id }; wait = $false }
    $null = Wait-Idle
    $re = Last-Row
    Write-Host ("  recreated: {0}" -f $re.status)
    Shot "w5b-recreated"

    Write-Host "6. Close the document; Place the lighthouse; Save (CLI) with the naming pattern"
    $null = Exec "file.closeAll" @{}
    $r = Exec "generate.place" @{ id = $lighthouse.id; target = "document" }
    Write-Host ("  placed as a document {0}x{1}" -f $r.width, $r.height)
    Shot "w6-placed"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}
$env:PHOTOCRAFT_CONFIG_DIR = $config
$out = Run-Cli @("run", "--new", '{\"width\":64,\"height\":64}', "--cmd", "generate.export", "--params", "{\`"id\`":$($lighthouse.id),\`"index\`":0,\`"dir\`":\`"$($saved.Replace('\','\\'))\`"}")
Write-Host ("  saved: {0}" -f ((Get-ChildItem $saved -Name) -join ", "))

Write-Host "7. Quit and relaunch: the list is still there"
$app = Start-App
try {
    Start-Sleep -Seconds 3
    $g = (Inspect).generations
    Write-Host ("  after relaunch: completed={0} failed={1} thumbnails={2}" -f $g.completed, $g.failed, $g.thumbnails)
    Shot "w7-relaunched"

    Write-Host "8. Home screen: the Image card opens the panel in Image mode; Help's Expand card"
    $null = Call "ui.set" @{ generatePanel = @{ mode = "fill" } }
    Start-Sleep -Milliseconds 600
    $card = (Inspect).hub.cards | Where-Object { $_.id -eq "image" }
    Shot "w8a-home"
    Click-Rect $card.rects[0]
    Write-Host ("  after the Image card: mode={0}" -f (Inspect).generatePanel.mode)
    $null = Exec "file.new" @{ width = 800; height = 600 }
    $null = Call "ui.menu.invoke" @{ id = "help.generativeFeatures" }
    Start-Sleep -Milliseconds 700
    Shot "w8b-help"
    $card = (Inspect).hub.cards | Where-Object { $_.id -eq "expand" }
    Click-Rect ($card.rects | Select-Object -Last 1)
    $i = Inspect
    Write-Host ("  after Help's Expand card: mode={0}, window open={1}" -f $i.generatePanel.mode, $i.hub.open)

    Write-Host "9. Preferences: Enter to generate, Alerts, the strip at the bottom"
    $null = Exec "prefs.set" @{ path = "integrations.enterToGenerate"; value = $true }
    $null = Exec "prefs.set" @{ path = "integrations.alerts"; value = "doneAndFailed" }
    $null = Exec "prefs.set" @{ path = "integrations.variationsStrip"; value = "bottom" }
    $null = Exec "edit.fill" @{ color = "#6c8fb3" }
    $null = Exec "select.rect" @{ x = 300; y = 200; width = 200; height = 160 }
    $null = Call "ui.set" @{ generatePanel = @{ mode = "fill"; prompt = "a hot air balloon"; count = 2; focus = $true } }
    Start-Sleep -Milliseconds 500
    $null = Call "ui.key" @{ key = "Enter" }
    Start-Sleep -Milliseconds 500
    $null = Wait-Idle
    Start-Sleep -Seconds 1
    $i = Inspect
    Write-Host ("  plain Enter generated: {0}; lastAlert={1}; strip place={2}, tiles={3}" -f (Last-Row).prompt, $i.generations.lastAlert, $i.variations.place, @(($i.variations.runs | Select-Object -First 1).tiles).Count)
    Shot "w9-bottom-strip"
    $guiTotal = (Exec "generate.jobs" @{}).total
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "10. The CLI lists the same records"
$env:PHOTOCRAFT_CONFIG_DIR = $config
$out = Run-Cli @("run", "--new", '{\"width\":64,\"height\":64}', "--cmd", "generate.jobs", "--params", "{}")
$total = if ($out -match '"total"\s*:\s*(\d+)') { [int]$Matches[1] } else { -1 }
Write-Host ("  CLI total={0}, GUI total={1}" -f $total, $guiTotal)
