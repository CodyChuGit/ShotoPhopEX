# Rust rebuild R5 live check (goals/rust-rebuild.md §1.1, §1.2, §1.4): the portable zip of
# `package.ps1 -Native`, unpacked into a fresh folder and run as a new user would, against the
# native engine only. Every generative command of the feature table runs in its own CLI process
# (the app starts its own worker from the zip's `runtime\` folder), and a watcher records any
# python.exe that starts and every DLL the worker loads.
#
# What "a new user" means here: nothing of this account reaches the run. APPDATA, LOCALAPPDATA,
# USERPROFILE, TEMP and TMP point into $Work\profile (the qir cache, the preferences, the
# generation library all start empty), PATH is Windows' own folders only (no CUDA toolkit, no
# scoop, no Python), and every PHOTOCRAFT_*, GN_* and CUDA_* variable is cleared. ComfyUI is
# neither running nor configured (comfyPath and comfyServer empty); the models come from
# -Models, a ComfyUI `models` folder used as it is. A separate Windows account is the same test
# with the account's own folders; this script does not create one.
#
# Prints one line per case (time, result or failure), then the python.exe check and the worker's
# module check (onnxruntime*, cudnn*, python*, torch* must be absent), and the runtime folder's
# size. Results land in $Work\out.
param(
    [string]$Zip = "",
    [string]$Models = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models",
    [string]$Picture = "C:\Users\5090\ComfyUI\photocraft-tests\bench\swap\gen\gen-tshirt.png",
    [string]$Work = "$env:TEMP\pex-clean-profile",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..\..\..").Path
if (-not $Zip) {
    $Zip = Get-ChildItem (Join-Path $root "target\windows-package"), (Join-Path $root "dist") -Filter "photocraft-*-windows-x64-native-portable.zip" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Zip -or -not (Test-Path $Zip)) { throw "no portable zip (build one with packaging/windows/package.ps1 -Arch x64 -Native, or pass -Zip)" }
if (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -match "ComfyUI" }) { throw "a ComfyUI python is running: stop it first (the check is that none is needed)" }

if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
$app = Join-Path $Work "app"; $profileDir = Join-Path $Work "profile"; $out = Join-Path $Work "out"
New-Item -ItemType Directory -Force $app, $out, "$profileDir\AppData\Roaming", "$profileDir\AppData\Local", "$profileDir\Temp" | Out-Null
Expand-Archive -Path $Zip -DestinationPath $app
$cli = Get-ChildItem $app -Recurse -Filter "photocraft-cli.exe" | Select-Object -First 1 -ExpandProperty FullName
if (-not $cli) { throw "no photocraft-cli.exe in $Zip" }
$appDir = Split-Path -Parent $cli
"zip: $Zip"
"unpacked: $appDir"

# The fresh profile and a bare PATH, for this process and every child it starts.
$env:APPDATA = "$profileDir\AppData\Roaming"
$env:LOCALAPPDATA = "$profileDir\AppData\Local"
$env:USERPROFILE = $profileDir
$env:TEMP = "$profileDir\Temp"; $env:TMP = $env:TEMP
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
Get-ChildItem env: | Where-Object { $_.Name -match '^(PHOTOCRAFT_|GN_|CUDA_|CUDARC_|QIR_|PYTHON)' } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$env:RUST_LOG = "info"

function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
# Native only, nothing of ComfyUI; the runtime folder is the zip's own (runtimePath empty).
$prefs = @{
    "integrations.engine" = "native"; "integrations.comfyPath" = ""; "integrations.comfyServer" = ""
    "integrations.runtimePath" = ""; "integrations.modelsPath" = $Models
    "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false
}

# One case = one CLI process: a start (a picture or a new document), then its commands in order.
$cases = [ordered]@{
    "image"      = @{ new = @{ width = 1024; height = 1024 }; cmds = @(, @("generate.image", @{ prompt = "a red fox sitting in fresh snow at dawn, 35mm photo"; target = "layer"; width = 1024; height = 1024; seed = 11 })) }
    "byText"     = @{ cmds = @(, @("select.byText", @{ prompt = "t-shirt" })) }
    "byPoint"    = @{ cmds = @(, @("select.byPoint", @{ x = 512; y = 640 })) }
    "fill"       = @{ cmds = @(@("select.byText", @{ prompt = "t-shirt" }), @("generate.fill", @{ prompt = "a denim jacket"; seed = 12 })) }
    "remove"     = @{ cmds = @(@("select.rect", @{ x = 40; y = 40; width = 160; height = 160 }), @("generate.fill", @{ prompt = ""; erase = $true; seed = 13 })) }
    "edit"       = @{ cmds = @(, @("generate.edit", @{ prompt = "Make it winter: snow falling and a wool scarf around his neck."; wholePicture = $true; seed = 14 })) }
    "expand"     = @{ cmds = @(, @("generate.expand", @{ right = 256; prompt = ""; seed = 15 })) }
    "matte"      = @{ cmds = @(, @("generate.removeBackground", @{ prompt = "" })) }
    "angles"     = @{ cmds = @(, @("generate.angles", @{ rotation = 45; seed = 16 })) }
    "upscale"    = @{ cmds = @(, @("generate.upscale", @{ scale = 2; seed = 17 })) }
    "split"      = @{ cmds = @(, @("generate.splitLayers", @{ layers = 3; seed = 7 })) }
}

# The watcher: python.exe processes that appear during the run, and the modules of the worker
# (a photocraft process started from the unpacked folder with --engine).
$flag = Join-Path $Work "watch.stop"
$watch = Start-Job -ArgumentList $appDir, $flag -ScriptBlock {
    param($appDir, $flag)
    $before = @(Get-Process python -ErrorAction SilentlyContinue | ForEach-Object Id)
    $pythons = @{}; $modules = @{}
    while (-not (Test-Path $flag)) {
        Get-Process python -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } | ForEach-Object { $pythons[$_.Id] = $_.Path }
        Get-CimInstance Win32_Process -Filter "Name like 'photocraft%'" | Where-Object { $_.CommandLine -match '--engine' -and $_.ExecutablePath -like "$appDir*" } | ForEach-Object {
            try { (Get-Process -Id $_.ProcessId -Module -ErrorAction Stop) | ForEach-Object { $modules[$_.ModuleName.ToLowerInvariant()] = $_.FileName } } catch {}
        }
        Start-Sleep -Milliseconds 500
    }
    [pscustomobject]@{ pythons = $pythons; modules = $modules }
}

$results = [ordered]@{}
foreach ($name in $cases.Keys) {
    if ($Only.Count -and $Only -notcontains $name) { continue }
    $c = $cases[$name]
    $a = [System.Collections.Generic.List[string]]::new()
    if ($c.new) { $a.Add("--new"); $a.Add((Q $c.new)) } else { $a.Add($Picture) }
    $a.Add("--cmd"); $a.Add("prefs.set"); $a.Add("--params"); $a.Add((Q @{ values = $prefs }))
    foreach ($cmd in $c.cmds) { $a.Add("--cmd"); $a.Add($cmd[0]); $a.Add("--params"); $a.Add((Q $cmd[1])) }
    $a.Add("--out"); $a.Add("$out\$name.png")
    $t0 = Get-Date
    $lines = & $cli run @a 2>&1
    $secs = ((Get-Date) - $t0).TotalSeconds
    $last = $c.cmds[-1][0]
    $res = $lines | Where-Object { "$_" -match ('"command":"' + [regex]::Escape($last) + '"') } | Select-Object -First 1
    $ok = [bool]$res -and (Test-Path "$out\$name.png")
    $results[$name] = $ok
    $detail = if ($res) { "$res".Substring(0, [Math]::Min(200, "$res".Length)) } else { "FAILED " + (($lines | Select-Object -Last 3) -join " | ") }
    "{0,-8} {1,6:N1} s  {2}  {3}" -f $name, $secs, $(if ($ok) { "ok  " } else { "FAIL" }), $detail
}

New-Item -ItemType File $flag | Out-Null
$seen = Receive-Job $watch -Wait -AutoRemoveJob
""
$pythonCount = @($seen.pythons.Keys).Count
"python.exe started during the run: $pythonCount" + $(if ($pythonCount) { " (" + (($seen.pythons.Values | Sort-Object -Unique) -join ", ") + ")" } else { "" })
$bad = @($seen.modules.Keys | Where-Object { $_ -match '^(onnxruntime|cudnn|python|torch)' })
"worker modules seen: $(@($seen.modules.Keys).Count); onnxruntime/cudnn/python/torch among them: $($bad.Count)" + $(if ($bad.Count) { " (" + ($bad -join ", ") + ")" } else { "" })
$nvidia = @($seen.modules.Values | Where-Object { $_ -like "$appDir*" } | ForEach-Object { Split-Path -Leaf $_ } | Sort-Object -Unique)
"loaded from the zip's folder: " + ($nvidia -join ", ")
$rt = Join-Path $appDir "runtime"
if (Test-Path $rt) { "runtime folder: {0:N0} MB" -f ((Get-ChildItem $rt -Recurse -File | Measure-Object Length -Sum).Sum / 1MB) } else { "runtime folder: none beside the exes" }
$failed = @($results.Keys | Where-Object { -not $results[$_] })
"cases: $($results.Count), failed: $($failed.Count)" + $(if ($failed.Count) { " (" + ($failed -join ", ") + ")" } else { "" })
