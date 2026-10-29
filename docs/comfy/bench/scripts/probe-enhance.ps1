# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Can the loaded text encoder rewrite prompts? TextGenerate on Qwen2.5-VL-7B (the 2511 encoder)
# with the picture and ComfyUI's shipped edit-enhancer system prompt; and the Krea 2 encoder
# (Qwen3-VL 4B) with the shipped text-to-image enhancer.
$srv = "http://127.0.0.1:8188"
$t = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\comfyui_workflow_templates_json\templates"
$sp = $PSScriptRoot
function SystemPromptOf($file, $needle) {
    $j = Get-Content (Join-Path $t $file) -Raw | ConvertFrom-Json
    $all = @($j.nodes) + @($j.definitions.subgraphs | ForEach-Object { $_.nodes })
    foreach ($n in $all) { if ($n.type -eq "PrimitiveStringMultiline" -and "$($n.widgets_values[0])" -match $needle) { return "$($n.widgets_values[0])" } }
    return $null
}
$editSys = SystemPromptOf "image_qwen_image_2_1_image_edit.json" "Edit Prompt Enhancer"
$t2iSys = SystemPromptOf "image_krea2_turbo_t2i.json" "expert prompt engineer"
$utf8 = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText((Join-Path $sp "enhance-edit-system.txt"), $editSys, $utf8)
[IO.File]::WriteAllText((Join-Path $sp "enhance-t2i-system.txt"), $t2iSys, $utf8)
"edit system prompt: {0} chars; t2i system prompt: {1} chars" -f $editSys.Length, $t2iSys.Length
# PreviewAny's schema (how text comes back).
$oi = (Invoke-WebRequest -Uri "$srv/object_info/PreviewAny" -UseBasicParsing -TimeoutSec 20).Content | ConvertFrom-Json
"PreviewAny inputs: " + ($oi.PreviewAny.input | ConvertTo-Json -Depth 5 -Compress) + " output_node: " + $oi.PreviewAny.output_node
# Upload the doodle.
$image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\user-drawing.png"
$bytes = [IO.File]::ReadAllBytes($image)
$boundary = [Guid]::NewGuid().ToString(); $lf = "`r`n"
$head = "--$boundary$lf" + "Content-Disposition: form-data; name=`"image`"; filename=`"probe-doodle.png`"$lf" + "Content-Type: image/png$lf$lf"
$tail = "$lf--$boundary--$lf"
$ms = New-Object IO.MemoryStream
$h = [Text.Encoding]::ASCII.GetBytes($head); $ms.Write($h, 0, $h.Length); $ms.Write($bytes, 0, $bytes.Length); $tb = [Text.Encoding]::ASCII.GetBytes($tail); $ms.Write($tb, 0, $tb.Length)
$up = Invoke-WebRequest -Uri "$srv/upload/image" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $ms.ToArray() -UseBasicParsing -TimeoutSec 60
$upName = ($up.Content | ConvertFrom-Json).name
function RunText($tag, $graph) {
    $body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60 } catch { "$tag queue failed: " + $_.ErrorDetails.Message; return }
    $pid2 = ($q.Content | ConvertFrom-Json).prompt_id
    $done = $null
    for ($i = 0; $i -lt 300; $i++) { Start-Sleep -Milliseconds 500; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
    $sw.Stop()
    if (-not $done) { "$tag timed out"; return }
    "== $tag : " + $done.status.status_str + " in " + $sw.ElapsedMilliseconds + " ms"
    if ($done.status.status_str -ne "success") { $done.status.messages | ConvertTo-Json -Depth 6 -Compress; return }
    "outputs: " + ($done.outputs | ConvertTo-Json -Depth 6 -Compress)
}
$editGraph = @{
    "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"; type = "qwen_image"; device = "default" } }
    "2" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = "make this hyper realistic"; max_length = 400; sampling_mode = "on"; temperature = 0.7; top_k = 64; top_p = 0.95; min_p = 0.05; repetition_penalty = 1.05; seed = 1; image = @("2", 0); system_prompt = $editSys; use_default_template = $true } }
    "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
}
RunText "edit-enhance (7B VL, with image)" $editGraph
$editGraph."3".inputs.prompt = "turn this drawing into a hyper realistic portrait that looks similar"
$editGraph."3".inputs.seed = 2
RunText "edit-enhance 2" $editGraph
$t2iGraph = @{
    "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
    "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = "a red fox in autumn leaves"; max_length = 400; sampling_mode = "on"; temperature = 0.7; top_k = 64; top_p = 0.95; min_p = 0.05; repetition_penalty = 1.05; seed = 1; system_prompt = $t2iSys; use_default_template = $true } }
    "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
}
RunText "t2i-enhance (Krea 2 encoder)" $t2iGraph
