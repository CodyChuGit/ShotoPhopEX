# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [string]$Images = "$PSScriptRoot\..\images",
    [int]$Port = 7881,
    [string]$Work = "$env:TEMP\pex-model-picker-live"
)
# Plan Phase 3 acceptance, live: the release GUI (its own config folder under $Work) with the model
# picker opened on each surface through `ui.inspect.modelPickers` + `ui.click`, a screenshot of
# each popover: the Generate panel in Fill and Image mode, the task bar, and the Generate Image
# dialog. Needs ComfyUI on 127.0.0.1:8188 (for the `not installed` badges) and krea2-fox.png in
# $Images. Nothing is generated.
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

# Show or hide a dock group through its Window menu toggle (the layout persists between runs).
function Set-Panel([string]$name, [bool]$on) {
    if ([bool](Call "ui.inspect" @{}).panels.$name -ne $on) { $null = Call "ui.menu.invoke" @{ id = "window.toggle.$name" } }
}

# Open the picker `id` (as `ui.inspect.modelPickers` lists it), screenshot its popover, close it.
function Open-Picker([string]$id, [string]$shot) {
    $deadline = (Get-Date).AddSeconds(10)
    $picker = $null
    while (-not $picker -and (Get-Date) -lt $deadline) {
        $picker = (Call "ui.inspect" @{}).modelPickers | Where-Object { $_.id -eq $id } | Select-Object -First 1
        if (-not $picker) { Start-Sleep -Milliseconds 300 }
    }
    if (-not $picker) { throw "no model picker '$id' on screen" }
    # A surface that just appeared (the task bar) or a dock that just changed settles over a few
    # frames: click where the picker is now, and once more if the popover did not open.
    $open = $false
    for ($try = 0; $try -lt 2 -and -not $open; $try++) {
        Start-Sleep -Milliseconds 500
        $picker = (Call "ui.inspect" @{}).modelPickers | Where-Object { $_.id -eq $id } | Select-Object -First 1
        $x = $picker.rect[0] + $picker.rect[2] / 2
        $y = $picker.rect[1] + $picker.rect[3] / 2
        $null = Call "ui.click" @{ x = $x; y = $y }
        Start-Sleep -Milliseconds 500
        $open = [bool]((Call "ui.inspect" @{}).modelPickers | Where-Object { $_.id -eq $id }).open
    }
    Start-Sleep -Milliseconds 1000  # the background probe answers; badges settle
    Write-Host ("  {0}: '{1}', popover open: {2}" -f $id, $picker.label, $open)
    Shot $shot
    $null = Call "ui.key" @{ key = "Escape" }
    Start-Sleep -Milliseconds 300
}

$app = Start-App
try {
    $null = Call "app.open" @{ path = "krea2-fox.png" }
    $null = Exec "select.rect" @{ x = 200; y = 200; width = 300; height = 220 }

    # A 900-pt window gives the panel its one-row form (mode, prompt, Generate) while Generations
    # is open too; hide Generations so the full form, with the picker, has room.
    Set-Panel "generations" $false
    Set-Panel "generate" $true
    Write-Host "1. The Generate panel, Fill"
    $null = Call "ui.set" @{ generatePanel = @{ mode = "fill"; prompt = "a red kite in the sky" } }
    Start-Sleep -Milliseconds 500
    $i = Call "ui.inspect" @{}
    if ($i.generatePanel.compact) { Write-Host ("  still compact; window {0}x{1}, panels: {2}" -f $i.window.width, $i.window.height, ($i.panels | ConvertTo-Json -Compress)) }
    Open-Picker "generate-panel" "phase3-1-panel-fill"

    Write-Host "2. The Generate panel, Image"
    $null = Call "ui.set" @{ generatePanel = @{ mode = "image"; prompt = "a lighthouse at dusk" } }
    Open-Picker "generate-panel" "phase3-2-panel-image"

    Write-Host "3. The task bar (the panel hidden, so the bar shows for the selection)"
    Set-Panel "generate" $false
    $null = Exec "select.rect" @{ x = 200; y = 200; width = 300; height = 220 }
    $null = Call "ui.set" @{ generativeBar = $true; generativeMode = "fill" }
    Open-Picker "generative-bar" "phase3-3-task-bar"
    Set-Panel "generate" $true
    Set-Panel "generations" $true

    Write-Host "4. The Generate Image dialog"
    $null = Call "ui.menu.invoke" @{ id = "generate.image" }
    Open-Picker "flt-generate.image" "phase3-4-dialog"
} finally {
    if ($script:client) { $script:client.Close() }
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
