# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The Krea 2 encoder (Qwen3-VL 4B) as the rewriter: text-to-image the official way (system text
# inside the prompt), and edit instructions with the picture attached.
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
    if ($done.status.status_str -ne "success") { ($done.status.messages | ConvertTo-Json -Depth 6 -Compress).Substring(0, 500); return }
    foreach ($o in $done.outputs.PSObject.Properties) { "  node {0}: {1}" -f $o.Name, (($o.Value.text -join " || ") -replace "\s+", " ") }
}
function Gen($prompt, $thinking, $withImage, $seed) {
    $g = @{
        "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
        "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = $prompt; max_length = 400; sampling_mode = "on"; "sampling_mode.temperature" = 0.7; "sampling_mode.top_k" = 64; "sampling_mode.top_p" = 0.95; "sampling_mode.min_p" = 0.05; "sampling_mode.repetition_penalty" = 1.05; "sampling_mode.seed" = $seed; thinking = $thinking; use_default_template = $true } }
        "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
        "5" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 1) } }
    }
    if ($withImage) { $g["2"] = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }; $g["3"].inputs["image"] = @("2", 0) }
    return $g
}
RunText "t2i, system in prompt, thinking on" (Gen ($t2iSys + "`na red fox in autumn leaves") $true $false 1)
RunText "t2i, system in prompt, thinking off" (Gen ($t2iSys + "`na red fox in autumn leaves") $false $false 1)
RunText "edit, with image: make this hyper realistic" (Gen ($editSys + "`nmake this hyper realistic") $false $true 1)
RunText "edit, with image: turn this drawing into a portrait" (Gen ($editSys + "`nturn this drawing into a hyper realistic portrait that looks similar") $false $true 2)
RunText "edit, with image: a red boat (fill-style description)" (Gen ($editSys + "`nadd a small red wooden rowing boat") $false $true 3)
