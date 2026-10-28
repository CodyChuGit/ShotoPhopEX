# The native-vs-ComfyUI speed table (the owner's goal of 2026-10-28: every native feature at
# least as fast as its ComfyUI route at the same settings and size). Every generative feature at
# the owner's usual 1 MP size on a 3:4 portrait (864x1152), the same template, seed and
# settings on both routes; each case run twice in a row: "cold" (the first run, which loads the
# case's model after whatever ran before) and "warm" (the second). One GPU job at a time: the
# native route runs on one shared worker (native-worker.ps1, with the flags the app passes: no
# --text-quant, SAM 3.1 in f16), which is then stopped; ComfyUI is
# started fresh afterwards (start-comfyui-fast.ps1) and stopped at the end.
# Prints one line per case and route, and a markdown table (seconds) for benchmarks.md.
param(
    [string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\speed",
    [string]$Source = "C:\Users\5090\ComfyUI\photocraft-tests\bench\swap\gen\gen-tshirt.png",
    [string[]]$Only = @(),
    [string[]]$Routes = @("native", "comfy"),
    [switch]$SaveImages,
    [switch]$ProfileComfy,
    [ValidateSet("default", "none", "restore")][string]$UpscaleLora = "default"
)
$ErrorActionPreference = "Continue"
$sc = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = [IO.Path]::GetFullPath((Join-Path $sc "..\..\..\.."))
$cli = Join-Path $repo "target\release\photocraft-cli.exe"
if ($env:PHOTOCRAFT_CLI) { $cli = $env:PHOTOCRAFT_CLI }
# Keep the CLI and the shared worker on the same executable, including in a worktree.
$env:PHOTOCRAFT_CLI = $cli
$models = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"
New-Item -ItemType Directory -Force $Out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 8).Replace(' ', '\u0020').Replace('"', '\"') }

# The 864x1152 portrait: the source's centre at 3:4, resampled.
$pic = Join-Path $Out "portrait-864x1152.png"
if (-not (Test-Path $pic)) {
    Add-Type -AssemblyName System.Drawing
    $src = [System.Drawing.Image]::FromFile($Source)
    $cw = [int][Math]::Min($src.Width, $src.Height * 3 / 4); $ch = [int]($cw * 4 / 3)
    $bmp = New-Object System.Drawing.Bitmap 864, 1152
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawImage($src, (New-Object System.Drawing.Rectangle 0, 0, 864, 1152), (New-Object System.Drawing.Rectangle ([int](($src.Width - $cw) / 2)), ([int](($src.Height - $ch) / 2)), $cw, $ch), [System.Drawing.GraphicsUnit]::Pixel)
    $g.Dispose(); $src.Dispose(); $bmp.Save($pic); $bmp.Dispose()
}
$ref = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock\st-hat.jpg"

# Each case: a start (the picture, or a new 864x1152 document) and its commands. Templates are
# named so both routes run the same model.
$cases = [ordered]@{
    "image (Krea 2)"     = @{ new = $true; cmds = @(, @("generate.image", @{ prompt = "a red fox sitting in fresh snow at dawn, 35mm photo"; template = "krea2-turbo/image"; target = "layer"; width = 864; height = 1152; seed = 11; enhance = $false })) }
    "image (2.1)"        = @{ new = $true; cmds = @(, @("generate.image", @{ prompt = "a lighthouse on a rocky coast under a stormy sky, oil painting"; template = "qwen-2.1/image"; target = "layer"; width = 864; height = 1152; seed = 12; enhance = $false })) }
    "image from a ref"   = @{ new = $true; cmds = @(, @("generate.image", @{ prompt = "a portrait of a woman wearing this hat"; template = "qwen-2.1/image-ref"; target = "layer"; width = 864; height = 1152; seed = 13; enhance = $false; references = @(@{ path = ($ref -replace '\\', '/') }) })) }
    "fill"               = @{ cmds = @(@("select.rect", @{ x = 180; y = 520; width = 500; height = 420 }), @("generate.fill", @{ prompt = "a denim jacket"; template = "qwen-2.1/fill"; seed = 14; enhance = $false })) }
    "fill from a ref"    = @{ cmds = @(@("select.rect", @{ x = 180; y = 520; width = 500; height = 420 }), @("generate.fill", @{ prompt = "this hat"; fillAsEdit = $true; seed = 15; enhance = $false; references = @(@{ path = ($ref -replace '\\', '/') }) })) }
    "edit"               = @{ cmds = @(, @("generate.edit", @{ prompt = "Make it winter: snow falling and a wool scarf around his neck."; template = "qwen-2.1/edit"; wholePicture = $true; seed = 16; enhance = $false })) }
    "expand"             = @{ cmds = @(, @("generate.expand", @{ right = 288; prompt = ""; template = "qwen-2.1/expand"; seed = 17 })) }
    "remove (stroke)"    = @{ cmds = @(, @("paint.remove", @{ points = @(@(100, 100), @(240, 100), @(240, 240), @(100, 240), @(100, 100)); size = 40; generative = $true; variations = 1; enhance = $false })) }
    "remove background"  = @{ cmds = @(, @("generate.removeBackground", @{ prompt = "the man"; template = "qwen-2.1/matte"; seed = 18 })) }
    "angles"             = @{ cmds = @(, @("generate.angles", @{ rotation = 45; seed = 19 })) }
    "select by text"     = @{ cmds = @(, @("select.byText", @{ prompt = "t-shirt" })) }
    "upscale 2x"         = @{ cmds = @(, @("generate.upscale", @{ scale = 2; seed = 20; template = "qwen-2.1/edit" })) }
    "split (3 layers)"   = @{ cmds = @(, @("generate.splitLayers", @{ layers = 3; seed = 7; maxPixels = 1000000 })) }
}
if ($UpscaleLora -eq "restore") {
    $cases["upscale 2x"].cmds[0][1]["loras"] = @(@{ file = "qwen-image-2.1-restore.safetensors"; strength = 0.5 })
} elseif ($UpscaleLora -eq "none") {
    $cases["upscale 2x"].cmds[0][1]["loras"] = @()
}
$prefs = @{
    native = @{ "integrations.engine" = "native"; "integrations.runtimePath" = "C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = $models; "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false }
    comfy  = @{ "integrations.engine" = "comfy"; "integrations.comfyServer" = "http://127.0.0.1:8188"; "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false }
}

# The warm run asks for something new (the seed + 1, another SAM prompt): ComfyUI returns a
# repeated request from its cache without running anything (0.2-0.6 s in the first table).
function Fresh($params, [int]$bump) {
    $h = @{}; foreach ($k in $params.Keys) { $h[$k] = $params[$k] }
    if ($bump -and $h.ContainsKey("seed")) { $h["seed"] = [int]$h["seed"] + $bump }
    if ($bump -and $h.ContainsKey("prompt") -and -not $h.ContainsKey("seed")) { $h["prompt"] = "the " + $h["prompt"] }
    $h
}
function Run-Case([string]$route, [string]$name, $c, [int]$bump = 0) {
    $a = [System.Collections.Generic.List[string]]::new()
    if ($c.new) { $a.Add("--new"); $a.Add((Q @{ width = 864; height = 1152 })) } else { $a.Add($pic) }
    $a.Add("--cmd"); $a.Add("prefs.set"); $a.Add("--params"); $a.Add((Q @{ values = $prefs[$route] }))
    foreach ($cmd in $c.cmds) { $a.Add("--cmd"); $a.Add($cmd[0]); $a.Add("--params"); $a.Add((Q (Fresh $cmd[1] $bump))) }
    $tag = ($name -replace '[^a-zA-Z0-9]', '-') + "-$route-$bump"
    if ($SaveImages) { $a.Add("--out"); $a.Add((Join-Path $Out "$tag.png")) }
    $t0 = Get-Date
    $lines = & $cli run @a 2>&1 | ForEach-Object { "$_" }
    $secs = ((Get-Date) - $t0).TotalSeconds
    $lines | Set-Content (Join-Path $Out "$tag.log")
    $last = $c.cmds[-1][0]
    $res = $lines | Where-Object { $_ -match ('"command":"' + [regex]::Escape($last) + '"') } | Select-Object -First 1
    $tpl = if ($res -and $res -match '"template":"([^"]+)"') { $Matches[1] } else { "" }
    [pscustomobject]@{ ok = [bool]$res; secs = $secs; template = $tpl; err = $(if ($res) { "" } else { ($lines | Select-Object -Last 2) -join " | " }) }
}

$table = [ordered]@{}
foreach ($route in $Routes) {
    if ($route -eq "native") {
        $env:PHOTOCRAFT_NATIVE_ENGINE = $null
        . "$sc\native-worker.ps1" -Models $models -Log "$Out\worker.log" -SamPrecision f16 -TextQuant $false
        if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no native worker"; continue }
    } else {
        $env:PHOTOCRAFT_NATIVE_ENGINE = $null
        $before = @(Get-Process python -ErrorAction SilentlyContinue | ForEach-Object Id)
        if ($ProfileComfy) {
            $portable = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable"
            Start-Process "$portable\python_embeded\python.exe" -ArgumentList "-s", "$sc\profile-comfy-upscale.py", "--root", "$portable\ComfyUI", "--work", $Out -RedirectStandardOutput "$Out\comfy.log" -RedirectStandardError "$Out\comfy.err" -WindowStyle Hidden | Out-Null
        } else {
            Start-Process powershell -ArgumentList "-NoProfile", "-File", "C:\Users\5090\ComfyUI\start-comfyui-fast.ps1" -WindowStyle Hidden | Out-Null
        }
        $up = $false
        foreach ($k in 1..120) { Start-Sleep -Seconds 2; try { Invoke-WebRequest http://127.0.0.1:8188/system_stats -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true; break } catch {} }
        if (-not $up) { "ComfyUI did not come up"; continue }
    }
    foreach ($name in $cases.Keys) {
        if ($Only.Count -and $Only -notcontains $name) { continue }
        $cold = Run-Case $route $name $cases[$name]
        $warm = Run-Case $route $name $cases[$name] 1
        $table["$name|$route"] = @($cold, $warm)
        "{0,-18} {1,-6} cold {2,7:N1} s  warm {3,7:N1} s  {4} {5}" -f $name, $route, $cold.secs, $warm.secs, $warm.template, $(if ($cold.ok -and $warm.ok) { "" } else { "FAILED " + $cold.err + " " + $warm.err })
    }
    if ($route -eq "native") {
        . "$sc\native-worker.ps1" -Stop
    } else {
        Get-Process python -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } | ForEach-Object { Stop-Process -Id $_.Id -Force -Confirm:$false }
    }
}
""
"| Feature (864x1152) | native cold | native warm | ComfyUI cold | ComfyUI warm | warm ratio | template (native / ComfyUI) |"
"|---|---|---|---|---|---|---|"
foreach ($name in $cases.Keys) {
    if ($Only.Count -and $Only -notcontains $name) { continue }
    $n = $table["$name|native"]; $c = $table["$name|comfy"]
    $f = { param($r) if ($r -and $r.ok) { "{0:N1}" -f $r.secs } elseif ($r) { "fail" } else { "-" } }
    if (-not $n) { $n = @($null, $null) }; if (-not $c) { $c = @($null, $null) }
    $same = $n -and $c -and $n[1].template -eq $c[1].template
    $ratio = if ($n -and $c -and $n[1].ok -and $c[1].ok) { if ($same) { "{0:N2}x" -f ($n[1].secs / $c[1].secs) } else { "different model" } } else { "" }
    $tpl = if ($n -and $c) { "{0} / {1}" -f $n[1].template, $c[1].template } else { "" }
    "| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f $name, (& $f $n[0]), (& $f $n[1]), (& $f $c[0]), (& $f $c[1]), $ratio, $tpl
}
