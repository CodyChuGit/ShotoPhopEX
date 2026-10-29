# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Live: a fill, then Generate Similar on it (no selection), then the layer's info, in one process.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:SL_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:SL_RECT = '{\"x\":80,\"y\":120,\"width\":420,\"height\":300}'
$env:SL_FILL = '{\"prompt\":\"a hot air balloon drifting in the evening sky\",\"seed\":7}'
$env:SL_SIM = '{\"seed\":8}'
$env:SL_OUT = "C:\Users\5090\ComfyUI\photocraft-tests\bench\similar-balloon.png"
$lines = & $cli --% run %SL_IMAGE% --cmd select.rect --params "%SL_RECT%" --cmd generate.fill --params "%SL_FILL%" --cmd select.deselect --cmd generate.similar --params "%SL_SIM%" --cmd generate.info --out %SL_OUT%
foreach ($l in $lines) {
    if ("$l" -match '"command":"generate\.(fill|similar)"') { $r = ("$l" | ConvertFrom-Json); "{0,-18} {1,6} ms seed {2} layer {3} via {4}" -f $r.command, $r.result.ms, $r.result.seed, $r.result.layer, $r.result.template }
    elseif ("$l" -match '"command":"generate\.info"') { $r = ("$l" | ConvertFrom-Json); "info: layer {0} -> {1}" -f $r.result.layer, ($r.result.generative | ConvertTo-Json -Compress) }
    elseif ("$l" -match 'rror') { $l }
}
