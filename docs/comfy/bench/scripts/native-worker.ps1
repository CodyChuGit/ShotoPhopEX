# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Start the native worker by hand so a run of many CLI processes shares one resident set of
# models: `photocraft-cli --engine --stay --runtime … --models …` with a token, its port read
# from its first stdout line. Prints `PHOTOCRAFT_NATIVE_ENGINE=port:token` for the caller to
# put in the environment (dot-source this script: `. native-worker.ps1` sets $env: too).
# -Stop ends a worker started this way (by the pid file it leaves in $env:TEMP).
# -TextQuant (on): the worker's `--text-quant`, the 2.1 text encoder in Q8_0 (8 GB instead of
# 16), so the worker's set, ComfyUI's vision-language model for the rewriter and the card's
# other tenants fit the 32 GB together (without it the rewriter paged for minutes, 2026-10-26).
# -Models: the models root (a ComfyUI `models` folder works since R4, 2026-10-27).
param([switch]$Stop, [string]$Log = "C:\Users\5090\ComfyUI\photocraft-tests\bench\native-worker.log", [bool]$TextQuant = $true, [string]$Models = "C:/Users/5090/Documents/_eval/native-models", [string]$SamPrecision = "")
$pidFile = Join-Path $env:TEMP "photocraft-native-worker.pid"
if ($Stop) {
    if (Test-Path $pidFile) {
        $id = [int](Get-Content $pidFile | Select-Object -First 1)
        Stop-Process -Id $id -Force -Confirm:$false -ErrorAction SilentlyContinue
        Remove-Item $pidFile -Force
        "worker $id stopped"
    }
    $env:PHOTOCRAFT_NATIVE_ENGINE = $null
    return
}
$cli = if ($env:PHOTOCRAFT_CLI) { $env:PHOTOCRAFT_CLI } else { "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe" }
$runtime = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib"
$token = -join ((1..48) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
$env:GN_ENGINE_TOKEN = $token
# The worker's phase timings (load, encode, sampling, decode) go to its stderr log.
if (-not $env:RUST_LOG) { $env:RUST_LOG = "info" }
# The worker needs the CUDA and ONNX Runtime libraries on its PATH (the app puts the runtime
# folders there when it starts the worker itself).
$env:PATH = ($runtime -replace "/", "\") + ";" + $env:PATH
$flags = @("--engine", "--stay", "--runtime", $runtime, "--models", $Models)
if ($TextQuant) { $flags += "--text-quant" }
# -SamPrecision f16|f32: SAM 3.1's compute precision on candle (the worker's `--sam-precision`).
if ($SamPrecision) { $flags += "--sam-precision", $SamPrecision }
$p = Start-Process -FilePath $cli -ArgumentList $flags -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddSeconds(60)
$port = $null
while ((Get-Date) -lt $deadline -and -not $port) {
    Start-Sleep -Milliseconds 500
    $line = Get-Content $Log -ErrorAction SilentlyContinue | Where-Object { $_ -match "gn-engine port=(\d+)" } | Select-Object -First 1
    if ($line -and $line -match "port=(\d+)") { $port = $Matches[1] }
    if ($p.HasExited) { "worker exited: " + (Get-Content "$Log.err" -Tail 5 -ErrorAction SilentlyContinue); return }
}
if (-not $port) { "no port line within 60 s"; return }
Set-Content $pidFile $p.Id
$env:PHOTOCRAFT_NATIVE_ENGINE = "${port}:$token"
"PHOTOCRAFT_NATIVE_ENGINE=$($env:PHOTOCRAFT_NATIVE_ENGINE) (pid $($p.Id))"
