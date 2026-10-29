# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A short rewriter prompt for the 4B vision model, on the doodle and the lighthouse, plus a
# fill case where the selection is tinted red; then a 2511 edit right after, for the timing.
$srv = "http://127.0.0.1:8188"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$editSys = @"
You rewrite instructions for an image-editing AI. You can see the image.
Rewrite the user's instruction as one clear sentence of at most 60 words that:
- starts with the action verb (Change, Replace, Add, Remove, Turn, Make, Give),
- names things by what they are in the image instead of "this", "it" or "here" (for example "the hand-drawn smiley face", "the white lighthouse", "the woman in the red coat"),
- says concretely what the result should look like,
- ends with what must stay unchanged.
Keep the user's intent. Add nothing they did not ask for. Keep any quoted text exactly. If a region is tinted red, the change happens inside that region and the sentence must describe what appears there.
Reply with the sentence only: no explanation, no quotes, no markdown.
"@
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
# The lighthouse with a red tint over the sky region (a fill's selection).
Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Bitmap]::FromFile((Join-Path $bench "final-fill-s11.png"))
$ov = New-Object System.Drawing.Bitmap $src.Width, $src.Height
$g = [System.Drawing.Graphics]::FromImage($ov); $g.DrawImage($src, 0, 0, $src.Width, $src.Height); $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(110, 255, 0, 0)); $g.FillRectangle($brush, 80, 120, 420, 300); $g.Dispose(); $src.Dispose()
$ovPath = Join-Path $env:TEMP "lighthouse-tinted.png"; $ov.Save($ovPath, [System.Drawing.Imaging.ImageFormat]::Png); $ov.Dispose()
$doodle = Upload (Join-Path $bench "user-drawing.png") "probe-doodle.png"
$light = Upload (Join-Path $bench "final-fill-s11.png") "probe-lighthouse.png"
$tinted = Upload $ovPath "probe-lighthouse-tinted.png"
function RunText($tag, $graph) {
    $body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing -TimeoutSec 60 } catch { $d = $_.ErrorDetails.Message; if (-not $d) { $d = $_.Exception.Message }; "$tag queue failed: " + $d; return }
    $pid2 = ($q.Content | ConvertFrom-Json).prompt_id
    $done = $null
    for ($i = 0; $i -lt 600; $i++) { Start-Sleep -Milliseconds 300; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
    $sw.Stop()
    if (-not $done) { "$tag timed out"; return }
    if ($done.status.status_str -ne "success") { "== $tag : error " + ($done.status.messages | ConvertTo-Json -Depth 6 -Compress).Substring(0, 400); return }
    "== {0} ({1} ms): {2}" -f $tag, $sw.ElapsedMilliseconds, (($done.outputs."4".text -join " || ") -replace "\s+", " ")
}
function Gen($instruction, $image, $seed) {
    @{
        "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
        "2" = @{ class_type = "LoadImage"; inputs = @{ image = $image; upload = "image" } }
        "3" = @{ class_type = "TextGenerate"; inputs = @{ clip = @("1", 0); prompt = ($editSys + "`nInstruction: " + $instruction); max_length = 120; sampling_mode = "on"; "sampling_mode.temperature" = 0.3; "sampling_mode.top_k" = 40; "sampling_mode.top_p" = 0.9; "sampling_mode.min_p" = 0.05; "sampling_mode.repetition_penalty" = 1.05; "sampling_mode.seed" = $seed; image = @("2", 0); thinking = $false; use_default_template = $true } }
        "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
    }
}
RunText "doodle: make this hyper realistic" (Gen "make this hyper realistic" $doodle 1)
RunText "doodle: turn this drawing into a hyper realistic portrait that looks similar" (Gen "turn this drawing into a hyper realistic portrait that looks similar" $doodle 1)
RunText "lighthouse: make the sky stormy" (Gen "make the sky stormy" $light 1)
RunText "lighthouse: turn it blue" (Gen "turn it blue" $light 1)
RunText "tinted fill: a hot air balloon" (Gen "a hot air balloon" $tinted 1)
RunText "tinted fill: remove this" (Gen "remove this" $tinted 1)
# Now a 2511 edit right after the small model ran: does it still take ~15 s?
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$env:PE_IMAGE = Join-Path $bench "final-fill-s11.png"
$env:PE_PARAMS = '{\"prompt\":\"make the sky dark and stormy with heavy clouds\",\"seed\":21}'
$env:PE_OUT = Join-Path $bench "after-enhancer-edit.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %PE_IMAGE% --cmd generate.edit --params "%PE_PARAMS%" --out %PE_OUT%
foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "2511 edit after the enhancer: {0} ms (wall {1} ms)" -f $r.ms, $sw.ElapsedMilliseconds } }
RunText "lighthouse again after 2511: make the sky stormy" (Gen "make the sky stormy" $light 2)
