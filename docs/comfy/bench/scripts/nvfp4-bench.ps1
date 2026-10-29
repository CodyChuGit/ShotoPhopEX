# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# NVFP4 vs fp8 on the 5090: Comfy-Org's official NVFP4 files against the fp8 ones the templates
# ship with, through the engine's own graphs posted straight to the server (the CLI's `model`
# only swaps the first slot). Each case: purge, then three runs (cold, warm, warm); the time of
# each, free VRAM after. Seeds differ per run: ComfyUI caches a repeated graph (a 290 ms 'run'). Then the case that matters for the enhancer: the 4B encoder resident
# before a 2511 edit, with either 2511 encoder.
$srv = "http://127.0.0.1:8188"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$wf = "C:\Users\5090\Projects\comfy-photocraft\crates\genai\workflows"
function Upload($path, $name) {
    $bytes = [IO.File]::ReadAllBytes($path)
    $boundary = [Guid]::NewGuid().ToString(); $lf = "`r`n"
    $head = "--$boundary$lf" + "Content-Disposition: form-data; name=`"image`"; filename=`"$name`"$lf" + "Content-Type: image/png$lf$lf"
    $tail = "$lf--$boundary--$lf"
    $ms = New-Object IO.MemoryStream
    $h = [Text.Encoding]::ASCII.GetBytes($head); $ms.Write($h, 0, $h.Length); $ms.Write($bytes, 0, $bytes.Length); $tb = [Text.Encoding]::ASCII.GetBytes($tail); $ms.Write($tb, 0, $tb.Length)
    $up = Invoke-WebRequest -Uri "$srv/upload/image" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $ms.ToArray() -UseBasicParsing -TimeoutSec 60
    return ($up.Content | ConvertFrom-Json).name
}
function Free() {
    $body = '{"unload_models": true, "free_memory": true}'
    Invoke-WebRequest -Uri "$srv/free" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60 | Out-Null
    Start-Sleep -Seconds 2
}
function VramFree() { $s = (Invoke-WebRequest -UseBasicParsing "$srv/system_stats").Content | ConvertFrom-Json; return [math]::Round($s.devices[0].vram_free / 1GB, 1) }
function Post($graphJson, $tag) {
    $body = '{"prompt": ' + $graphJson + ', "client_id": "nvfp4-bench"}'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing -TimeoutSec 60 } catch { $d = $_.ErrorDetails.Message; if (-not $d) { $d = $_.Exception.Message }; return @{ err = "queue failed: " + $d.Substring(0, [Math]::Min(400, $d.Length)) } }
    $pid2 = ($q.Content | ConvertFrom-Json).prompt_id
    $done = $null
    for ($i = 0; $i -lt 3000; $i++) { Start-Sleep -Milliseconds 250; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
    $sw.Stop()
    if (-not $done) { return @{ err = "timed out" } }
    if ($done.status.status_str -ne "success") { return @{ err = "error " + ($done.status.messages | ConvertTo-Json -Depth 6 -Compress).Substring(0, 400) } }
    $img = $null
    foreach ($o in $done.outputs.PSObject.Properties) { if ($o.Value.images) { $img = $o.Value.images[0] } }
    if ($img -and $tag) {
        $url = "$srv/view?filename=" + [Uri]::EscapeDataString($img.filename) + "&subfolder=" + [Uri]::EscapeDataString($img.subfolder) + "&type=" + $img.type
        Invoke-WebRequest -Uri $url -OutFile (Join-Path $bench "nvfp4-$tag.png") -UseBasicParsing -TimeoutSec 60
    }
    return @{ ms = $sw.ElapsedMilliseconds }
}
function Fill($template, $map) {
    $j = Get-Content (Join-Path $wf $template) -Raw | ConvertFrom-Json
    $g = $j.graph | ConvertTo-Json -Depth 10 -Compress
    foreach ($k in $map.Keys) {
        $v = $map[$k]
        if ($v -is [string]) { $g = $g.Replace('"{{' + $k + '}}"', ('"' + $v.Replace('\', '\\').Replace('"', '\"') + '"')) }
        else { $g = $g.Replace('"{{' + $k + '}}"', "$v") }
    }
    return $g
}
$light = Upload (Join-Path $bench "final-fill-s11.png") "bench-lighthouse.png"
$editPrompt = "make the sky dark and stormy with heavy clouds. Keep everything else exactly as it is."
$foxPrompt = "A fox stands alert in deep snow, its fur thick and russet-brown, paws sinking slightly into the white. Its ears twitch, eyes sharp and focused. Soft, diffused daylight illuminates the scene from above, casting gentle shadows. The snow stretches endlessly around it. Rendered as a photorealistic image, the palette is muted whites and browns, evoking stillness and cold."
function Edit2511($te, $seed, $tag) {
    Fill "qwen-edit-2511-edit-lightning-8.json" @{ unet = "qwen_image_edit_2511_fp8mixed.safetensors"; clip = $te; vae = "qwen_image_vae.safetensors"; lora = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"; image = $light; prompt = $editPrompt; negative = " "; seed = $seed; steps = 8; cfg = 1.0; prefix = "photocraft/nvfp4-$tag" }
}
function Krea2($unet, $seed, $tag) {
    Fill "krea2-turbo-image.json" @{ unet = $unet; clip = "qwen3vl_4b_fp8_scaled.safetensors"; vae = "qwen_image_vae.safetensors"; prompt = $foxPrompt; seed = $seed; steps = 8; cfg = 1.0; width = 1024; height = 1024; prefix = "photocraft/nvfp4-$tag" }
}
function Enhancer($seed) {
    Fill "enhance-image.json" @{ clip = "qwen3vl_4b_fp8_scaled.safetensors"; prompt = "a fox in snow"; system = "Expand the idea into one sentence."; seed = $seed; prefix = "photocraft/nvfp4-enh" }
}
function Series($name, $make, $tag) {
    Free
    $times = @()
    for ($i = 1; $i -le 3; $i++) {
        $r = & $make $i $(if ($i -eq 1) { $tag } else { $null })
        if ($r.err) { "  $name run $i : $($r.err)"; return }
        $times += $r.ms
    }
    "{0,-44} cold {1,6} ms, warm {2,6} / {3,6} ms, free VRAM after {4} GB" -f $name, $times[0], $times[1], $times[2], (VramFree)
}
"==== 2511 Lightning edit, 1024x1024 lighthouse, 8 steps"
Series "2511 + fp8 encoder" { param($i, $t) Post (Edit2511 "qwen_2.5_vl_7b_fp8_scaled.safetensors" (20 + $i) $t) $t } "2511-te-fp8"
Series "2511 + NVFP4 encoder" { param($i, $t) Post (Edit2511 "qwen_2.5_vl_7b_nvfp4.safetensors" (20 + $i) $t) $t } "2511-te-nvfp4"
"==== Krea 2 Turbo, 1024x1024, 8 steps"
Series "Krea 2 fp8" { param($i, $t) Post (Krea2 "krea2_turbo_fp8_scaled.safetensors" (6 + $i) $t) $t } "krea2-fp8"
Series "Krea 2 NVFP4" { param($i, $t) Post (Krea2 "krea2_turbo_nvfp4.safetensors" (6 + $i) $t) $t } "krea2-nvfp4"
"==== 2511 edit with the enhancer's 4B encoder resident"
foreach ($te in @("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_2.5_vl_7b_nvfp4.safetensors")) {
    Free
    $e = Post (Enhancer 1) $null
    if ($e.err) { "  enhancer: $($e.err)"; continue }
    $times = @()
    for ($i = 1; $i -le 3; $i++) { $r = Post (Edit2511 $te (30 + $i) $null) $null; if ($r.err) { "  $te run $i : $($r.err)"; break }; $times += $r.ms }
    "{0,-44} enhancer {1} ms, edits {2} / {3} / {4} ms, free VRAM after {5} GB" -f ("4B resident + 2511 + " + $(if ($te -match "nvfp4") { "NVFP4" } else { "fp8" }) + " encoder"), $e.ms, $times[0], $times[1], $times[2], (VramFree)
}
