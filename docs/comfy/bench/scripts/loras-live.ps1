# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
param(
    [string]$Exe = "$PSScriptRoot\..\..\..\..\target\release\photocraft.exe",
    [int]$Port = 7881,
    [string]$Work = "$env:TEMP\pex-loras-live"
)
# Overnight 2026-10-27 (Q5): the Generate panel's LoRAs rows live in the release GUI (its own
# config folder under $Work, so the owner's preferences are not touched): the native folders
# set, the panel on a Qwen-Image-2.1 edit, one LoRA switched on through ui.set; ui.inspect shows
# the template's LoRA files and the request carries the one switched on; a screenshot with the
# disclosure open. Starts no generation.
$ErrorActionPreference = "Stop"
$config = Join-Path $Work "config"
$token = Join-Path $Work "control.token"
New-Item -ItemType Directory -Force $config | Out-Null
$env:PHOTOCRAFT_CONFIG_DIR = $config
$p = Start-Process -FilePath $Exe -ArgumentList @("--control", "$Port", "--control-token-file", "`"$token`"", "--automation-write-root", "`"$Work`"") -PassThru
$deadline = (Get-Date).AddSeconds(60)
while ($true) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $Port)
        $s = $c.GetStream()
        $reader = New-Object System.IO.StreamReader($s)
        $writer = New-Object System.IO.StreamWriter($s)
        $writer.AutoFlush = $true
        break
    } catch {
        if ((Get-Date) -gt $deadline) { Stop-Process -Id $p.Id -Force; throw "no control port" }
        Start-Sleep -Milliseconds 500
    }
}
$script:n = 0
function Call([string]$method, $params = @{}) {
    $script:n++
    $writer.WriteLine((@{ id = $script:n; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 20))
    while ($true) {
        $reply = $reader.ReadLine() | ConvertFrom-Json
        if ($reply.id -eq $script:n) { break }
    }
    if (-not $reply.ok) { throw "$method failed: $($reply.error)" }
    return $reply.result
}
try {
    $null = Call "auth" @{ token = (Get-Content $token -Raw).Trim() }
    $runtime = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib"
    $null = Call "engine.execute" @{ id = "prefs.set"; params = @{ values = @{ "integrations.engine" = "native"; "integrations.allowResearchModels" = $true; "integrations.runtimePath" = $runtime; "integrations.modelsPath" = "C:/Users/5090/Documents/_eval/native-models" } }; wait = $true }
    $null = Call "engine.execute" @{ id = "file.new"; params = @{ width = 512; height = 512 }; wait = $true }
    $null = Call "ui.set" @{ generatePanel = @{ mode = "edit"; template = "qwen-2.1/edit"; prompt = "restore the photo"; loras = @{ "qwen-image-2.1-restore.safetensors" = 0.5 } } }
    Start-Sleep -Seconds 2
    $panel = (Call "ui.inspect" @{}).generatePanel
    "effective template: $($panel.effectiveTemplate)"
    "LoRA files: $(($panel.loras.files) -join ', ')"
    "native: $($panel.loras.native)"
    "request loras: $($panel.params.loras | ConvertTo-Json -Compress)"
    $null = Call "ui.screenshot" @{ path = "loras-panel.png" }
    "screenshot: $(Join-Path $Work 'loras-panel.png')"
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
