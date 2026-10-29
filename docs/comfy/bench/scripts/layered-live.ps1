# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
# Rust rebuild R3 live check: Split into Layers (Qwen-Image-Layered, 3 layers, seed 7) through
# the app on the native worker (ComfyUI's models folder, engine = native) and on ComfyUI (the
# server this script starts, then stops: never both on the card), each layer exported as PNG,
# then compared per layer (mean |RGB| difference and alpha IoU at 0.5) and timed. -MaxPixels sets the
# working size by pixel count (1000000 on a 3:4 picture = 864x1152) instead of 640 px.
param([string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\layered", [string]$Picture = "C:\Users\5090\ComfyUI\photocraft-tests\bench\swap\gen\gen-tshirt.png", [int]$Layers = 3, [int]$Seed = 7, [int]$MaxPixels = 0)
$sc = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = if ($env:PHOTOCRAFT_CLI) { $env:PHOTOCRAFT_CLI } else { "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe" }
$models = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"
New-Item -ItemType Directory -Force $Out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
$prefs = @{
    native = @{ "integrations.engine" = "native"; "integrations.runtimePath" = "C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = $models; "integrations.enhancePrompts" = $false }
    comfy = @{ "integrations.engine" = "comfy"; "integrations.comfyServer" = "http://127.0.0.1:8188"; "integrations.enhancePrompts" = $false }
}
function Split-Run([string]$route) {
    # The split adds its layers as ids 3.. (1 is the picture; 2 is taken), background first; the result line names them.
    $args2 = @($Picture, "--cmd", "prefs.set", "--params", (Q @{ values = $prefs[($route -replace "-warm", "")] }), "--cmd", "generate.splitLayers", "--params", (Q $(if ($MaxPixels) { @{ layers = $Layers; seed = $Seed; maxPixels = $MaxPixels } } else { @{ layers = $Layers; seed = $Seed } })))
    foreach ($i in 1..$Layers) { $args2 += "--cmd", "layer.exportAs", "--params", (Q @{ layer = $i + 2; path = (("$Out\$route-$i.png") -replace '\\', '/') }) }
    $t0 = Get-Date
    $lines = & $cli run @args2 2>&1
    $res = $lines | Where-Object { "$_" -match '"command":"generate\.splitLayers"' } | Select-Object -First 1
    "{0}: {1:N1} s {2}" -f $route, ((Get-Date) - $t0).TotalSeconds, $(if ($res) { "$res".Substring(0, [Math]::Min(240, "$res".Length)) } else { "FAILED " + (($lines | Select-Object -Last 3) -join " | ") })
}
# Native first: the worker over ComfyUI's folder.
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
. "$sc\native-worker.ps1" -Models $models -Log "$Out\worker.log"
if ($env:PHOTOCRAFT_NATIVE_ENGINE) { Split-Run "native"; Split-Run "native-warm" }
. "$sc\native-worker.ps1" -Stop
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
# Then ComfyUI, started here and stopped after.
$before = @(Get-Process python -ErrorAction SilentlyContinue | ForEach-Object Id)
Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden | Out-Null
$up = $false
foreach ($k in 1..90) { Start-Sleep -Seconds 2; try { Invoke-WebRequest http://127.0.0.1:8188/system_stats -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true; break } catch {} }
if ($up) { Split-Run "comfy" } else { "ComfyUI did not come up" }
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } | ForEach-Object { Stop-Process -Id $_.Id -Force -Confirm:$false; "stopped ComfyUI python $($_.Id)" }
Add-Type -AssemblyName System.Drawing
foreach ($i in 1..$Layers) {
    $a = "$Out\native-$i.png"; $b = "$Out\comfy-$i.png"
    if (-not ((Test-Path $a) -and (Test-Path $b))) { "layer ${i}: missing"; continue }
    $ia = New-Object System.Drawing.Bitmap $a; $ib = New-Object System.Drawing.Bitmap $b
    # RGB over the pixels visible in either (the colour under zero alpha is invisible and differs
    # freely), on the area both exports share (layer.exportAs trims to the layer's bounds).
    $W = [Math]::Min($ia.Width, $ib.Width); $H = [Math]::Min($ia.Height, $ib.Height)
    $sum = 0.0; $n = 0; $inter = 0; $union = 0
    for ($y = 0; $y -lt $H; $y += 3) { for ($x = 0; $x -lt $W; $x += 3) {
        $pa = $ia.GetPixel($x, $y); $pb = $ib.GetPixel($x, $y)
        $oa = $pa.A -ge 128; $ob = $pb.A -ge 128; if ($oa -and $ob) { $inter++ }
        if ($oa -or $ob) { $union++; $sum += [Math]::Abs($pa.R - $pb.R) + [Math]::Abs($pa.G - $pb.G) + [Math]::Abs($pa.B - $pb.B); $n += 3 } } }
    "layer {0}: visible |RGB| {1:N2}/255, alpha IoU {2:N3}" -f $i, $(if ($n) { $sum / $n } else { 0 }), $(if ($union) { $inter / $union } else { 1 })
    $ia.Dispose(); $ib.Dispose()
}
