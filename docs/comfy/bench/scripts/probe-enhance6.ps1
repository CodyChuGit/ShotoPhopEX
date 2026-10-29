# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The rewriter with the fixed template (thinking=true: a plain assistant turn for the
# non-thinking 4B Instruct model; the instructions as a real system turn) and a refined edit
# prompt that handles "make this realistic" on a doodle: more instructions, two seeds.
$srv = "http://127.0.0.1:8188"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$editSys = @"
You rewrite instructions for an image-editing AI. You can see the image.
Rewrite the user's instruction as one clear sentence of at most 60 words that:
- starts with the action verb (Change, Replace, Add, Remove, Turn, Make, Give),
- names things by what they are in the image instead of "this", "it" or "here" (for example "the hand-drawn smiley face", "the white lighthouse", "the woman in the red coat"),
- says concretely what the result should look like,
- ends with what must stay unchanged, if anything should.
When the instruction changes the style, medium or realism of the picture ("make this realistic", "turn it into a painting"), name the real-world subject the result shows and say that the drawing's strokes are replaced by it (a doodle of a face made realistic becomes "a photograph of a real human face with the same expression and pose, replacing the black outline strokes with real skin, hair and facial features"); never ask to keep the drawing's lines.
Keep the user's intent. Add nothing they did not ask for. Keep any quoted text exactly.
Reply with the sentence only: no explanation, no quotes, no markdown.
"@
$fillNote = "The user selected the {place} of the image; the instruction says what should be generated inside that selection so that it blends with its surroundings. Describe what appears there."
$imageSys = @"
You are an expert prompt writer for a text-to-image model. Expand the user's idea into one paragraph of 60 to 120 words the model can follow: the subject and what it does first, then materials and clothing, pose and framing, lighting, the setting, and finally the style, palette and mood.
Keep every subject, action, colour and spatial relation the user gave. Add no new objects, characters or animals. Keep any medium the user named (photo, painting, sketch, 3D render). Put any words to be rendered in the image in double quotes. If the idea is already detailed, polish it lightly and keep its wording.
Reply with the paragraph only: no title, no bullets, no quotes around it, no markdown.
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
$doodle = Upload (Join-Path $bench "user-drawing.png") "probe-doodle.png"
$light = Upload (Join-Path $bench "final-fill-s11.png") "probe-lighthouse.png"
function RunText($graph) {
    $body = @{ prompt = $graph; client_id = "probe6" } | ConvertTo-Json -Depth 8 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing -TimeoutSec 60 } catch { $d = $_.ErrorDetails.Message; if (-not $d) { $d = $_.Exception.Message }; return @{ err = "queue failed: " + $d.Substring(0, [Math]::Min(300, $d.Length)) } }
    $pid2 = ($q.Content | ConvertFrom-Json).prompt_id
    $done = $null
    for ($i = 0; $i -lt 600; $i++) { Start-Sleep -Milliseconds 200; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
    $sw.Stop()
    if (-not $done) { return @{ err = "timed out" } }
    if ($done.status.status_str -ne "success") { return @{ err = "error " + ($done.status.messages | ConvertTo-Json -Depth 6 -Compress).Substring(0, 300) } }
    return @{ ms = $sw.ElapsedMilliseconds; text = (($done.outputs."4".text -join " || ") -replace "\s+", " ") }
}
function Gen($system, $user, $image, $seed, $maxLen) {
    $inputs = @{ clip = @("1", 0); prompt = $user; system_prompt = $system; max_length = $maxLen; sampling_mode = "on"; "sampling_mode.temperature" = 0.3; "sampling_mode.top_k" = 40; "sampling_mode.top_p" = 0.9; "sampling_mode.min_p" = 0.05; "sampling_mode.repetition_penalty" = 1.05; "sampling_mode.seed" = $seed; thinking = $true; use_default_template = $true }
    $g = @{
        "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
        "3" = @{ class_type = "TextGenerate"; inputs = $inputs }
        "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
    }
    if ($image) { $g["2"] = @{ class_type = "LoadImage"; inputs = @{ image = $image; upload = "image" } }; $inputs.image = @("2", 0) }
    return $g
}
function Show($tag, $r) { if ($r.err) { "  {0,-34} {1}" -f $tag, $r.err } else { "  {0,-34} {1,5} ms: {2}" -f $tag, $r.ms, $r.text } }
"==== edits (system turn, thinking=true)"
foreach ($seed in 1..2) {
    Show "doodle: make this hyper realistic s$seed" (RunText (Gen $editSys "make this hyper realistic" $doodle $seed 120))
    Show "doodle: portrait that looks similar s$seed" (RunText (Gen $editSys "turn this drawing into a hyper realistic portrait that looks similar" $doodle $seed 120))
    Show "doodle: watercolor s$seed" (RunText (Gen $editSys "make it a watercolor painting" $doodle $seed 120))
    Show "light: make the sky stormy s$seed" (RunText (Gen $editSys "make the sky stormy" $light $seed 120))
    Show "light: turn it blue s$seed" (RunText (Gen $editSys "turn it blue" $light $seed 120))
    Show "light: remove the boat s$seed" (RunText (Gen $editSys "remove the boat" $light $seed 120))
    Show "light: add a flock of birds s$seed" (RunText (Gen $editSys "add a flock of birds" $light $seed 120))
    Show "light: vintage postcard s$seed" (RunText (Gen $editSys "make it look like a vintage postcard" $light $seed 120))
    Show "light: text sign s$seed" (RunText (Gen $editSys 'put a sign on the lighthouse that says "OPEN"' $light $seed 120))
}
"==== fills"
$note = $fillNote.Replace("{place}", "upper part")
Show "fill upper: a hot air balloon" (RunText (Gen ($editSys + "`n" + $note) "a hot air balloon" $light 1 120))
Show "fill upper: remove this" (RunText (Gen ($editSys + "`n" + $note) "remove this" $light 1 120))
Show "fill upper: birds" (RunText (Gen ($editSys + "`n" + $note) "birds" $light 1 120))
"==== image ideas (no picture)"
foreach ($seed in 1..2) {
    Show "idea: a fox in snow s$seed" (RunText (Gen $imageSys "a fox in snow" $null $seed 220))
    Show "idea: neon sign s$seed" (RunText (Gen $imageSys 'a neon sign that says "OPEN LATE" above a diner' $null $seed 220))
}
Show "idea: detailed" (RunText (Gen $imageSys "A close-up photo of a weathered fisherman mending a net on a wooden pier at golden hour, shallow depth of field, 85mm lens" $null 1 220))
