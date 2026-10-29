# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# R2 probe: can the native edit models (Qwen-Image-2.1 edit, Krea 2 General Edit V2) move the
# camera by a plain instruction, as the Multiple Angles LoRA does on 2511? Whole-picture edits.
$s = "C:\Users\5090\AppData\Local\Temp\claude\C--Users-5090-Projects-comfy-photocraft\23d7709e-573f-475c-a78d-721d5e6c037e\scratchpad"
$sc = "C:\Users\5090\Documents\PhotoshopEX\docs\comfy\bench\scripts"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$out = "$bench\angles"
$cli = "C:\Users\5090\Documents\PhotoshopEX\target\release\photocraft-cli.exe"
New-Item -ItemType Directory -Force $out | Out-Null
function Q([object]$v) { (ConvertTo-Json $v -Compress -Depth 6).Replace(' ', '\u0020').Replace('"', '\"') }
$native = @{ "integrations.engine" = "auto"; "integrations.runtimePath" = "C:/Users/5090/Documents/_eval/onnxruntime-cuda13/onnxruntime-win-x64-gpu_cuda13-1.31.0/lib;C:/Users/5090/scoop/apps/cuda/current/bin/x64"; "integrations.modelsPath" = "C:/Users/5090/Documents/_eval/native-models"; "integrations.allowResearchModels" = $true; "integrations.enhancePrompts" = $false }
$pics = [ordered]@{ denim = "$bench\swap\gen\gen-denim.png"; tshirt = "$bench\swap\gen\gen-tshirt.png"; bench = "$bench\stock\st-bench.jpg"; dog = "$bench\stock\st-dog.jpg" }
$cams = [ordered]@{ side2 = "Rotate the camera 90 degrees to the right around the subject, so it is seen from its right side. Keep the subject, its identity, its clothing and the setting the same."; back2 = "Move the camera behind the subject, so it is seen from the back. Keep the subject, its identity, its clothing and the setting the same."; high2 = "Raise the camera high above and to the front-right of the subject, looking down at it. Keep the subject, its identity, its clothing and the setting the same."; close2 = "Move the camera closer for a close-up of the subject. Keep the subject, its identity, its clothing and the setting the same." }
$routes = [ordered]@{ edit21 = "qwen-2.1/edit" }
$env:PHOTOCRAFT_NATIVE_ENGINE = $null
. "$sc\native-worker.ps1"
foreach ($r in $routes.Keys) {
    foreach ($p in $pics.Keys) {
        foreach ($c in $cams.Keys) {
            $prompt = $cams[$c]
            $q = @{ prompt = $prompt; template = $routes[$r]; wholePicture = $true; seed = 7; enhance = $false }
            $png = "$out\$p-$c-$r.png"
            $t0 = Get-Date
            $lines = & $cli run $pics[$p] --cmd prefs.set --params (Q @{ values = $native }) --cmd generate.edit --params (Q $q) --out $png 2>&1
            $ok = ($lines | Where-Object { "$_" -match '"command":"generate\.edit"' }) -ne $null
            "{0}-{1}-{2}: {3:N1} s {4}" -f $p, $c, $r, ((Get-Date) - $t0).TotalSeconds, $(if ($ok) { "ok" } else { "FAILED " + (($lines | Select-Object -Last 2) -join " | ") })
        }
    }
}
. "$sc\native-worker.ps1" -Stop
"=== done $(Get-Date -Format HH:mm:ss)"
