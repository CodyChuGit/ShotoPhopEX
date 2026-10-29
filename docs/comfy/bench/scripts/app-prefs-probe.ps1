# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
# Reproduce the app's generative path from the CLI with the app's own preferences (engine
# auto, the native folders set, the server default): a generative Spot Healing stroke on the
# lighthouse's boat and a Krea 2 image, printing every result line and error.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_IMAGE = Join-Path $out "final-fill-s11.png"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"auto\",\"integrations.runtimePath\":\"C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64;C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib\",\"integrations.modelsPath\":\"C:/Users/5090/Documents/_eval/native-models\",\"integrations.comfyPath\":\"C:\\\\Users\\\\5090\\\\ComfyUI\\\\ComfyUI_windows_portable\"}}'
$env:CM_SPOT = '{\"type\":\"generative\",\"size\":56,\"hardness\":1,\"points\":[[300,664],[340,662],[380,660],[420,660],[450,662]]}'
$env:CM_IMG = '{\"prompt\":\"a red bicycle leaning on a white wall\",\"target\":\"layer\",\"width\":512,\"height\":512,\"seed\":3}'
"== health"
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd generate.health --params "{}"
"== spot healing (generative)"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd paint.spotHealing --params "%CM_SPOT%"
"   ($($sw.ElapsedMilliseconds) ms)"
"== generate.image (auto)"
$sw.Restart()
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd generate.image --params "%CM_IMG%"
"   ($($sw.ElapsedMilliseconds) ms)"
