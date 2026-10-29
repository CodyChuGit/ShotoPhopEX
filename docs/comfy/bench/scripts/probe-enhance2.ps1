# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# TextGenerate again: sampling off (no nested inputs), and sampling on with dotted nested keys.
$srv = "http://127.0.0.1:8188"
$sp = $PSScriptRoot
$editSys = [IO.File]::ReadAllText((Join-Path $sp "enhance-edit-system.txt"))
$t2iSys = [IO.File]::ReadAllText((Join-Path $sp "enhance-t2i-system.txt"))
$upName = "probe-doodle.png"
function RunText($tag, $graph) {
    $body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing -TimeoutSec 60 } catch { $d = $_.ErrorDetails.Message; if (-not $d) { $d = $_.Exception.Message }; "$tag queue failed: " + $d; return }
    $pid2 = ($q.Content | ConvertFrom-Json).prompt_id
    $done = $null
    for ($i = 0; $i -lt 600; $i++) { Start-Sleep -Milliseconds 500; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
    $sw.Stop()
    if (-not $done) { "$tag timed out"; return }
    "== $tag : " + $done.status.status_str + " in " + $sw.ElapsedMilliseconds + " ms"
    if ($done.status.status_str -ne "success") { ($done.status.messages | ConvertTo-Json -Depth 6 -Compress).Substring(0, 600); return }
    "outputs: " + ($done.outputs | ConvertTo-Json -Depth 6 -Compress)
}
$offGraph = @{
    "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"; type = "qwen_image"; device = "default" } }
    "2" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = "make this hyper realistic"; max_length = 300; sampling_mode = "off"; image = @("2", 0); system_prompt = $editSys; use_default_template = $true } }
    "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
}
RunText "edit-enhance sampling off" $offGraph
$onGraph = @{
    "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"; type = "qwen_image"; device = "default" } }
    "2" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = "turn this drawing into a hyper realistic portrait that looks similar"; max_length = 300; sampling_mode = "on"; "sampling_mode.temperature" = 0.7; "sampling_mode.top_k" = 64; "sampling_mode.top_p" = 0.95; "sampling_mode.min_p" = 0.05; "sampling_mode.repetition_penalty" = 1.05; "sampling_mode.seed" = 2; image = @("2", 0); system_prompt = $editSys; use_default_template = $true } }
    "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
}
RunText "edit-enhance sampling on (dotted keys)" $onGraph
$t2iGraph = @{
    "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
    "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = "a red fox in autumn leaves"; max_length = 300; sampling_mode = "off"; system_prompt = $t2iSys; use_default_template = $true } }
    "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
}
RunText "t2i-enhance (Krea 2 encoder, sampling off)" $t2iGraph

