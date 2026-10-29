# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Rust rebuild R4 live check: the same requests on a worker whose models root is the snapshot
# folder (native-models) and on one whose root is ComfyUI's own `models` folder (ComfyUI's single
# files, the tokenizer assembled from ComfyUI's qwen25_tokenizer), then the pictures compared.
# Krea 2 text to image, 2.1 text to image, a 2.1 edit. Prints times and the mean absolute
# difference per pair (0-255).
param([string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\r4")
$sc = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
New-Item -ItemType Directory -Force $Out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
$native = @{ "integrations.engine" = "auto"; "integrations.runtimePath" = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = "C:/Users/5090/Documents/_eval/native-models"; "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false }
$roots = [ordered]@{ snap = "C:/Users/5090/Documents/_eval/native-models"; comfy = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models" }
$jobs = @(
    @{ id = "krea2"; new = @{ width = 1024; height = 1024 }; cmd = "generate.image"; p = @{ prompt = "a red fox sitting in fresh snow at dawn, 35mm photo"; template = "krea2-turbo/image"; width = 1024; height = 1024; seed = 11; target = "layer"; enhance = $false } },
    @{ id = "q21img"; new = @{ width = 1024; height = 1024 }; cmd = "generate.image"; p = @{ prompt = "a lighthouse on a rocky coast under a stormy sky, oil painting"; template = "qwen-2.1/image"; width = 1024; height = 1024; seed = 12; target = "layer"; enhance = $false } },
    @{ id = "q21edit"; image = "$bench\swap\gen\gen-tshirt.png"; cmd = "generate.edit"; p = @{ prompt = "Make it winter: snow falling outside the window and a wool scarf around his neck."; template = "qwen-2.1/edit"; wholePicture = $true; seed = 13; enhance = $false } }
)
foreach ($r in $roots.Keys) {
    $env:PHOTOCRAFT_NATIVE_ENGINE = $null
    . "$sc\native-worker.ps1" -Models $roots[$r] -Log "$Out\worker-$r.log"
    if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker for $r"; continue }
    foreach ($j in $jobs) {
        $png = "$Out\$($j.id)-$r.png"
        # (An `if` returning a one-item array hands back the bare string, which splats badly.)
        $start = [System.Collections.Generic.List[string]]::new()
        if ($j.image) { $start.Add($j.image) } else { $start.Add("--new"); $start.Add((Q $j.new)) }
        $t0 = Get-Date
        $lines = & $cli run @start --cmd prefs.set --params (Q @{ values = $native }) --cmd $j.cmd --params (Q $j.p) --out $png 2>&1
        $res = $lines | Where-Object { "$_" -match ('"command":"' + [regex]::Escape($j.cmd) + '"') } | Select-Object -First 1
        "{0}-{1}: {2:N1} s {3}" -f $j.id, $r, ((Get-Date) - $t0).TotalSeconds, $(if ($res) { ("$res" | ConvertFrom-Json).result.template } else { "FAILED " + (($lines | Select-Object -Last 3) -join " | ") })
    }
    . "$sc\native-worker.ps1" -Stop
}
Add-Type -AssemblyName System.Drawing
foreach ($j in $jobs) {
    $a = "$Out\$($j.id)-snap.png"; $b = "$Out\$($j.id)-comfy.png"
    if (-not ((Test-Path $a) -and (Test-Path $b))) { continue }
    $ia = New-Object System.Drawing.Bitmap $a; $ib = New-Object System.Drawing.Bitmap $b
    $sum = 0.0; $n = 0
    for ($y = 0; $y -lt $ia.Height; $y += 4) { for ($x = 0; $x -lt $ia.Width; $x += 4) {
        $pa = $ia.GetPixel($x, $y); $pb = $ib.GetPixel($x, $y)
        $sum += [Math]::Abs($pa.R - $pb.R) + [Math]::Abs($pa.G - $pb.G) + [Math]::Abs($pa.B - $pb.B); $n += 3 } }
    "{0}: mean |snap - comfy| = {1:N2} / 255" -f $j.id, ($sum / $n)
    $ia.Dispose(); $ib.Dispose()
}
