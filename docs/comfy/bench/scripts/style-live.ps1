# The Krea 2 style reference (krea2-turbo/image-style: Krea 2 Turbo + Ostris's style LoRA, one or
# two reference pictures) natively and on ComfyUI, one route on the GPU at a time: the native
# worker first, then ComfyUI started fresh and stopped. Same prompt, references, size, seed and
# steps; prints each route's time and the native picture's mean |RGB| difference and PSNR
# against ComfyUI's.
param(
    [string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\style",
    [string]$Ref1 = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-beach.jpg",
    [string]$Ref2 = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-boat.jpg",
    [string]$Prompt = "a lighthouse on a cliff above the sea",
    [int]$Seed = 123
)
$ErrorActionPreference = "Continue"
$cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe"
$models = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"
New-Item -ItemType Directory -Force $Out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 8).Replace(' ', '\u0020').Replace('"', '\"') }
$params = @{ prompt = $Prompt; template = "krea2-turbo/image-style"; target = "document"; width = 1024; height = 1024; seed = $Seed; steps = 8; enhance = $false
    references = @(@{ path = ($Ref1 -replace '\\', '/') }, @{ path = ($Ref2 -replace '\\', '/') }) }
$prefs = @{
    native = @{ "integrations.engine" = "native"; "integrations.runtimePath" = "C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = $models; "integrations.enhancePrompts" = $false }
    comfy  = @{ "integrations.engine" = "comfy"; "integrations.comfyServer" = "http://127.0.0.1:8188"; "integrations.enhancePrompts" = $false }
}
function Run([string]$route) {
    $t0 = Get-Date
    $lines = & $cli run --new (Q @{ width = 1024; height = 1024 }) --cmd prefs.set --params (Q @{ values = $prefs[$route] }) --cmd generate.image --params (Q $params) --out "$Out\$route.png" 2>&1 | ForEach-Object { "$_" }
    $res = $lines | Where-Object { $_ -match '"command":"generate.image"' } | Select-Object -First 1
    "{0}: {1:N1} s {2}" -f $route, ((Get-Date) - $t0).TotalSeconds, $(if ($res) { $res.Substring(0, [Math]::Min(200, $res.Length)) } else { "FAILED " + (($lines | Select-Object -Last 2) -join " | ") })
}
Run "native"
# The ComfyUI this script starts, and only it, is stopped at the end: the launcher's process
# and its descendants, found from the PID Start-Process returns.
$launcher = Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden -PassThru
function Descendants([int]$id) { Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" | ForEach-Object { [int]$_.ProcessId; Descendants ([int]$_.ProcessId) } }
$up = $false
foreach ($k in 1..120) { Start-Sleep -Seconds 2; try { Invoke-WebRequest http://127.0.0.1:8188/system_stats -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true; break } catch {} }
if ($up) { Run "comfy" } else { "ComfyUI did not come up" }
$ours = @(Descendants $launcher.Id) + @($launcher.Id)
foreach ($id in $ours) { Stop-Process -Id $id -Force -Confirm:$false -ErrorAction SilentlyContinue }
"stopped the ComfyUI this script started: $($ours.Count) processes"
if ((Test-Path "$Out\native.png") -and (Test-Path "$Out\comfy.png")) {
    Add-Type -AssemblyName System.Drawing
    $a = New-Object System.Drawing.Bitmap "$Out\native.png"; $b = New-Object System.Drawing.Bitmap "$Out\comfy.png"
    $sum = 0.0; $sq = 0.0; $n = 0
    for ($y = 0; $y -lt $a.Height; $y += 2) { for ($x = 0; $x -lt $a.Width; $x += 2) {
        $pa = $a.GetPixel($x, $y); $pb = $b.GetPixel($x, $y)
        foreach ($d in @(($pa.R - $pb.R), ($pa.G - $pb.G), ($pa.B - $pb.B))) { $sum += [Math]::Abs($d); $sq += $d * $d; $n++ } } }
    $mse = $sq / $n
    "native vs ComfyUI: mean |RGB| {0:N2}/255, PSNR {1:N1} dB" -f ($sum / $n), $(if ($mse -gt 0) { 10 * [Math]::Log10(255 * 255 / $mse) } else { 99 })
    $a.Dispose(); $b.Dispose()
}
