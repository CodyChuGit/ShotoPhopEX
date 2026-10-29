# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
# Does a fill pay its "Model Initializing" again when the same graph runs twice in a row?
# Three generative Spot Healing dabs with the app's preferences on the lighthouse's boat:
# one candidate (the smoothed plain fill) twice, then Best of two (adapter + plain), printing
# the server-side run times; the server log's "Model Initialization complete!" lines give the
# staging time of each. Server up on 127.0.0.1:8188.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"auto\",\"integrations.runtimePath\":\"C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib\",\"integrations.modelsPath\":\"C:/Users/5090/Documents/_eval/native-models\",\"integrations.comfyPath\":\"C:\\\\Users\\\\5090\\\\ComfyUI\\\\ComfyUI_windows_portable\"}}'
$env:CM_ONE = '{\"type\":\"generative\",\"size\":56,\"hardness\":1,\"variations\":1,\"points\":[[300,664],[340,662],[380,660],[420,660],[450,662]]}'
$env:CM_TWO = '{\"type\":\"generative\",\"size\":56,\"hardness\":1,\"variations\":2,\"points\":[[300,664],[340,662],[380,660],[420,660],[450,662]]}'
function Show($line) {
    if ($line -match '"timings":(\[.*?\])') { "   timings $($Matches[1])" }
    elseif ($line -match '"template":"([^"]+)"') { "   template $($Matches[1])" }
    elseif ($line -match 'error') { "   $line" }
}
# `--%` takes the rest of its line, so the native call lives in its own function and the
# stream merge happens at the call.
function Invoke-Dab {
    & $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_DAB%"
}
foreach ($label in "one candidate, first", "one candidate, again", "best of two") {
    $env:CM_DAB = if ($label -eq "best of two") { $env:CM_TWO } else { $env:CM_ONE }
    "== $label"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-Dab 2>&1 | ForEach-Object { Show "$_" }
    "   ($($sw.ElapsedMilliseconds) ms)"
}
