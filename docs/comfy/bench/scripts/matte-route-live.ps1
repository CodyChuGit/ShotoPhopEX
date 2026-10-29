# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Live: Remove Background without the research opt-in (SAM 3.1 route), and Select by Point.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:MR_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:MR_A = '{\"prompt\":\"the red boat\",\"seed\":1}'
$env:MR_B = '{\"seed\":1}'
$env:MR_P = '{\"x\":350,\"y\":665}'
$env:MR_OUT1 = Join-Path $out "matte-sam-boat.png"
$env:MR_OUT2 = Join-Path $out "matte-sam-subject.png"
$env:MR_OUT3 = Join-Path $out "select-point-boat.png"
$lines = & $cli --% run %MR_IMAGE% --cmd generate.removeBackground --params "%MR_A%" --out %MR_OUT1%
$lines | Where-Object { $_ -match '"runId"|rror' } | ForEach-Object { $r = ($_ | ConvertFrom-Json).result; "sam boat:    {0,6} ms bounds {1} pixels {2} via {3}" -f $r.ms, ($r.bounds -join ","), $r.pixels, $r.template }
$lines = & $cli --% run %MR_IMAGE% --cmd generate.removeBackground --params "%MR_B%" --out %MR_OUT2%
$lines | Where-Object { $_ -match '"runId"|rror' } | ForEach-Object { $r = ($_ | ConvertFrom-Json).result; "sam subject: {0,6} ms bounds {1} pixels {2} via {3}" -f $r.ms, ($r.bounds -join ","), $r.pixels, $r.template }
$lines = & $cli --% run %MR_IMAGE% --cmd select.byPoint --params "%MR_P%" --cmd edit.fill --params "{\"color\":\"#00ff00\"}" --out %MR_OUT3%
$lines | Where-Object { $_ -match '"selected"|rror' } | ForEach-Object { $r = ($_ | ConvertFrom-Json).result; "by point:    {0,6} ms count {1} bounds {2}" -f $r.ms, $r.count, ($r.instances[0].bounds -join ",") }
