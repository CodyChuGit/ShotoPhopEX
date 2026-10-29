# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# After the edge change: the three placement cases (ab-run), then expand timings back to back.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$sp = "C:\Users\5090\AppData\Local\Temp\claude\C--Users-5090-Projects-comfy-photocraft\b777eacf-44e3-45c1-98c9-821e291ceadf\scratchpad"
"== fills (ab-run, Lightning 8) =="
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $sp "ab-run.ps1") -Templates "qwen-edit-2511/fill-lightning-8"
"== expands back to back =="
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:EX_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png"
$cases = @(
    @{ tag = "expand-right-s6"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":6}' },
    @{ tag = "expand-right-s7"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":7}' },
    @{ tag = "expand-frame";    params = '{\"left\":192,\"right\":192,\"bottom\":192,\"seed\":5}' },
    @{ tag = "expand-right-s8"; params = '{\"right\":384,\"prompt\":\"more open sea and evening sky\",\"seed\":8}' }
)
foreach ($c in $cases) {
    $env:EX_PARAMS = $c.params
    $env:EX_OUT = Join-Path $out "$($c.tag).png"
    $lines = & $cli --% run %EX_IMAGE% --cmd generate.expand --params "%EX_PARAMS%" --out %EX_OUT%
    $r = $null; foreach ($l in $lines) { if ("$l" -match '"timings"') { $r = ("$l" | ConvertFrom-Json).result } }
    if ($null -eq $r) { "{0,-16} FAILED: {1}" -f $c.tag, (($lines | Select-Object -Last 1) -join " ") } else { "{0,-16} {1,6} ms (sampling {2} ms) canvas {3} sent {4}x{5} -> {6}" -f $c.tag, $r.ms, $r.timings[0].runMs, ($r.canvas -join "x"), $r.requestWidth, $r.requestHeight, $env:EX_OUT }
    try { $s = (Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json; $d = $s.devices[0]; "    vram free {0:N0} MB, torch free {1:N0} MB" -f ($d.vram_free/1MB), ($d.torch_vram_free/1MB) } catch {}
}
