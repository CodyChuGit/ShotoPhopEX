# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The ComfyUI side needs the last ComfyUI-capable app commit cdcedbd8, or must drive ComfyUI's HTTP API directly (removed preferences/commands).
# Swap benchmark (swap-cases.json): each case's object selected by name with SAM 3.1
# (select.byText, grown 12 px, holes filled), then replaced through one route, one picture per
# seed, into bench\swap\<case>-<route>-s<seed>.png. Routes:
#   fill21    Generative Fill on Qwen-Image-2.1 (qwen-2.1/fill), the native worker
#   edit21    Generative Edit on Qwen-Image-2.1 (qwen-2.1/edit), the instruction, masked to the selection
#   editk2    Generative Edit on Krea 2 Turbo with the General Edit V2 LoRA (krea2-turbo/edit), the same
#   fillnamed Generative Fill as the app runs it with the rewriter on: auto, the description, the
#             selection's thing named and the fill made by instruction on the 2.1 edit template
#   fillenh   the same with fillAsEdit off: the 2.1 fill template and its fill rewriter
#   fill2511  Generative Fill on Qwen-Image-Edit-2511, the 8-step tier through ComfyUI (the old default)
#   fill2511n the same 2511 fill natively (genai-native d40c536+), models from ComfyUI's folder
#   edit2511n Generative Edit on 2511 natively (qwen-edit-2511/edit-lightning-8), masked to the selection
# -Generate makes the outfit pictures first (Krea 2 on the worker, bench\swap\gen\). Native routes
# want a worker started by native-worker.ps1 (PHOTOCRAFT_NATIVE_ENGINE); fill2511 a ComfyUI server
# on 127.0.0.1:8188. The prompt rewriter is off so the routes see the same words.
param([string]$Route = "fill21", [string]$Seeds = "7,11", [string]$Only = "", [switch]$Generate)
$root = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$out = Join-Path $root "swap"
New-Item -ItemType Directory -Force (Join-Path $out "gen") | Out-Null
$cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe"
$spec = Get-Content (Join-Path $PSScriptRoot "swap-cases.json") -Raw | ConvertFrom-Json
# Windows PowerShell hands a native program its arguments with the double quotes stripped, and
# does not quote an argument that holds one, so a space would split it: spaces go as \u0020
# (compressed JSON has none outside its strings).
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
$native = @{ "integrations.engine" = "auto"; "integrations.runtimePath" = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = "C:/Users/5090/Documents/_eval/native-models"; "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false }
$comfy = @{ "integrations.engine" = "comfy"; "integrations.comfyServer" = "http://127.0.0.1:8188"; "integrations.enhancePrompts" = $false }
if ($Generate) {
    foreach ($g in $spec.generate) {
        $png = Join-Path $out "gen\$($g.id).png"
        if (Test-Path $png) { continue }
        $img = @{ prompt = $g.prompt; template = "krea2-turbo/image"; width = $g.width; height = $g.height; seed = $g.seed; target = "layer"; enhance = $false }
        & $cli run --new (Q @{ width = $g.width; height = $g.height }) --cmd prefs.set --params (Q @{ values = $native }) --cmd generate.image --params (Q $img) --out $png *> "$png.log"
        "gen $($g.id): exit $LASTEXITCODE"
    }
}
# The native 2511 routes read ComfyUI's models folder as it is (the 2511 files live there).
$native2511 = $native.Clone(); $native2511["integrations.modelsPath"] = "C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"; $native2511["integrations.engine"] = "native"
$prefs = if ($Route -eq "fill2511") { $comfy } elseif ($Route -like "*2511n") { $native2511 } else { $native }
foreach ($c in $spec.cases) {
    if ($Only -and ($Only -split ',') -notcontains $c.id) { continue }
    $src = Join-Path $root ($c.image -replace '^stock/', 'stock\' -replace '^gen/', 'swap\gen\')
    foreach ($seed in ($Seeds -split ',')) {
        $png = Join-Path $out "$($c.id)-$Route-s$seed.png"
        $sel = @{ prompt = $c.select; grow = 12; fillHoles = $true }
        $gen = switch ($Route) {
            "edit21" { @("generate.edit", @{ prompt = $c.edit; template = "qwen-2.1/edit"; seed = [int]$seed; enhance = $false }) }
            "editk2" { @("generate.edit", @{ prompt = $c.edit; template = "krea2-turbo/edit"; seed = [int]$seed; enhance = $false }) }
            "fillnamed" { @("generate.fill", @{ prompt = $c.fill; seed = [int]$seed; enhance = $true; fillAsEdit = $true }) }
            "fillenh" { @("generate.fill", @{ prompt = $c.fill; seed = [int]$seed; enhance = $true; fillAsEdit = $false }) }
            "fill2511" { @("generate.fill", @{ prompt = $c.fill; template = "qwen-edit-2511/fill-lightning-8"; seed = [int]$seed; enhance = $false }) }
            "fill2511n" { @("generate.fill", @{ prompt = $c.fill; template = "qwen-edit-2511/fill-lightning-8"; seed = [int]$seed; enhance = $false }) }
            "edit2511n" { @("generate.edit", @{ prompt = $c.edit; template = "qwen-edit-2511/edit-lightning-8"; seed = [int]$seed; enhance = $false }) }
            default { @("generate.fill", @{ prompt = $c.fill; template = "qwen-2.1/fill"; seed = [int]$seed; enhance = $false }) }
        }
        $t0 = Get-Date
        & $cli run $src --cmd prefs.set --params (Q @{ values = $prefs }) --cmd select.byText --params (Q $sel) --cmd $gen[0] --params (Q $gen[1]) --out $png *> "$png.log"
        $ok = if ($LASTEXITCODE -eq 0) { "ok" } else { "FAILED: " + ((Get-Content "$png.log" | Select-String -Pattern "error" | Select-Object -First 1) -replace '\s+', ' ') }
        "{0} {1} s{2}: {3:N1} s {4}" -f $c.id, $Route, $seed, ((Get-Date) - $t0).TotalSeconds, $ok
    }
}
