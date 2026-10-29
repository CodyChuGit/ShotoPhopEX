# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# Rust rebuild R1 live check: Select Subject by text and by point on a worker over the native
# folders (SAM 3.1's ONNX graphs through ONNX Runtime) and on one over ComfyUI's `models` folder
# (SAM 3.1 on candle from ComfyUI's checkpoint). Prints each selection's result (bounds, pixel
# count) and wall time per request, the second request on a picture reusing its embedding.
param([string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\sam-live", [string]$SamPrecision = "", [string[]]$Only = @())
$sc = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe"
$stock = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
New-Item -ItemType Directory -Force $Out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
$native = @{ "integrations.engine" = "auto"; "integrations.runtimePath" = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = "C:/Users/5090/Documents/_eval/native-models" }
$roots = [ordered]@{ onnx = "C:/Users/5090/Documents/_eval/native-models"; candle = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models" }
$cases = @(
    @{ id = "dog"; image = "$stock\st-dog.jpg"; text = "dog"; point = @{ x = 930; y = 2140 } },
    @{ id = "bench"; image = "$stock\st-bench.jpg"; text = "bench"; point = @{ x = 1100; y = 690 } },
    @{ id = "cup"; image = "$stock\st-cup.jpg"; text = "coffee cup"; point = @{ x = 590; y = 870 } }
)
foreach ($r in $roots.Keys) {
    if ($Only.Count -and $Only -notcontains $r) { continue }
    $env:PHOTOCRAFT_NATIVE_ENGINE = $null
    . "$sc\native-worker.ps1" -Models $roots[$r] -Log "$Out\worker-$r.log" -SamPrecision $SamPrecision
    if (-not $env:PHOTOCRAFT_NATIVE_ENGINE) { "no worker for $r"; continue }
    foreach ($c in $cases) {
        foreach ($step in @(@("select.byText", @{ prompt = $c.text }), @("select.byPoint", @{ x = $c.point.x; y = $c.point.y }))) {
            $t0 = Get-Date
            $lines = & $cli run $c.image --cmd prefs.set --params (Q @{ values = $native }) --cmd $step[0] --params (Q $step[1]) 2>&1
            $res = $lines | Where-Object { "$_" -match ('"command":"' + [regex]::Escape($step[0]) + '"') } | Select-Object -First 1
            $txt = if ($res) { "$res".Substring(0, [Math]::Min(260, "$res".Length)) } else { "FAILED " + (($lines | Select-Object -Last 2) -join " | ") }
            "{0} {1} {2}: {3:N1} s {4}" -f $r, $c.id, $step[0], ((Get-Date) - $t0).TotalSeconds, $txt
        }
    }
    . "$sc\native-worker.ps1" -Stop
}
