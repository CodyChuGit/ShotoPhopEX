param(
    [int]$Width = 864,
    [int]$Height = 1152,
    [int]$Steps = 20,
    [string]$Label = 'baseline',
    [string]$InputPath = 'C:\Users\5090\ComfyUI\photocraft-tests\bench\layered\denim-3x4.png',
    [string]$Binary = '',
    [int]$ExportWidth = 832,
    [int]$ExportHeight = 1109,
    [switch]$Profile,
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
$env:CARGO_TARGET_DIR = "$PWD/target/layered-size-perf"
$env:CUDA_CACHE_PATH = "$PWD/target/layered-cuda-cache"
$env:CUDA_CACHE_MAXSIZE = '1073741824'
$env:QIR_LAYERED_INPUT = $InputPath
$env:QIR_LAYERED_WIDTH = "$Width"
$env:QIR_LAYERED_HEIGHT = "$Height"
$env:QIR_LAYERED_EXPORT_WIDTH = "$ExportWidth"
$env:QIR_LAYERED_EXPORT_HEIGHT = "$ExportHeight"
$env:QIR_LAYERED_STEPS = "$Steps"
$env:QIR_LAYERED_OUTPUT = "$PWD/target/layered-size-$Label-${Width}x$Height"
Remove-Item Env:QIR_PROFILE,Env:QIR_LAYERED_TIMING,Env:QIR_LAYERED_SYNC_STEPS -ErrorAction SilentlyContinue
if ($Profile) { $env:QIR_PROFILE = '1' }
if (!$Quiet) { $env:QIR_LAYERED_TIMING = '1' }
$executables = @(Get-ChildItem "$env:CARGO_TARGET_DIR/release/deps/layered_pipeline-*.exe")
if ($Binary) { $executables = @(Get-Item -LiteralPath $Binary) }
if ($executables.Count -ne 1) { throw 'Build the layered_pipeline integration test first (one matching exe required).' }
New-Item -ItemType Directory -Force target/layered-size-logs,$env:CUDA_CACHE_PATH | Out-Null
$log = "$PWD/target/layered-size-logs/$Label-${Width}x$Height.log"
# Let cmd merge native streams without PowerShell's NativeCommandError decoration.
$command = '"{0}" installed_benchmark --ignored --nocapture --test-threads=1 > "{1}" 2>&1' -f $executables[0].FullName, $log
& cmd /d /s /c "`"$command`""
if ($LASTEXITCODE -ne 0) { throw "benchmark failed: $LASTEXITCODE" }
Get-Content $log
