# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The NVFP4 policy through the CLI against the live server: what generate.models reports, and
# which files the server loads for a Krea 2 image and a 2511 edit with the default `auto`.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$log = "C:\Users\5090\ComfyUI\comfyui-20261009-210900.log.err"
$env:NL_LIGHT = Join-Path $bench "final-fill-s11.png"
$lines = & $cli --% run --new "{\"width\":64,\"height\":64}" --cmd generate.models --params {}
foreach ($l in $lines) {
    if ("$l" -match '"templates"') {
        $r = ("$l" | ConvertFrom-Json).result
        "device: $($r.device)  blackwell: $($r.blackwell)"
        foreach ($t in $r.templates) { foreach ($m in $t.models) { if ($m.nvfp4) { "{0,-36} {1,-6} {2} installed={3}  nvfp4={4} installed={5}" -f $t.id, $m.placeholder, $m.file, $m.installed, $m.nvfp4, $m.nvfp4Installed } } }
    }
}
$mark = (Get-Content $log | Measure-Object -Line).Lines
$env:NL_PARAMS = '{\"prompt\":\"a red fox in deep snow, photorealistic\",\"seed\":3,\"target\":\"document\"}'
$env:NL_OUT = Join-Path $bench "nvfp4-live-fox.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run --new "{\"width\":64,\"height\":64}" --cmd generate.image --params "%NL_PARAMS%" --out %NL_OUT%
foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "krea2 image: {0} ms (wall {1} ms) via {2}" -f $r.ms, $sw.ElapsedMilliseconds, $r.template } elseif ("$l" -match "rror") { $l } }
$env:NL_PARAMS = '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":3}'
$env:NL_OUT = Join-Path $bench "nvfp4-live-edit.png"
$sw.Restart()
$lines = & $cli --% run %NL_LIGHT% --cmd generate.edit --params "%NL_PARAMS%" --out %NL_OUT%
foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "2511 edit: {0} ms (wall {1} ms) via {2}" -f $r.ms, $sw.ElapsedMilliseconds, $r.template } elseif ("$l" -match "rror") { $l } }
"---- model files the server loaded since the mark:"
Get-Content $log | Select-Object -Skip $mark | Select-String -Pattern "paths=\[" | ForEach-Object { ($_.Line -replace ".*models\\\\", "") -replace "'\]", "" } | Select-Object -Unique
