# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Why the 4B rewriter answers nothing half the time: the chat template. With thinking=false
# ComfyUI appends "<think>\n\n</think>\n\n" after the assistant turn, which the non-thinking
# Qwen3-VL-4B-Instruct does not expect (it then emits <|im_end|> at once, or restarts the turn
# with "assistant"). Matrix: thinking on/off x instructions in the user turn vs a real system
# turn (TextGenerate's system_prompt), 3 seeds each; count empties and role echoes.
param([string]$Which = "all")
$srv = "http://127.0.0.1:8188"
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$editSys = @"
You rewrite instructions for an image-editing AI. You can see the image.
Rewrite the user's instruction as one clear sentence of at most 60 words that:
- starts with the action verb (Change, Replace, Add, Remove, Turn, Make, Give),
- names things by what they are in the image instead of "this", "it" or "here" (for example "the hand-drawn smiley face", "the white lighthouse", "the woman in the red coat"),
- says concretely what the result should look like,
- ends with what must stay unchanged.
Keep the user's intent. Add nothing they did not ask for. Keep any quoted text exactly.
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
$doodle = Upload (Join-Path $bench "user-drawing.png") "probe-doodle.png"
$light = Upload (Join-Path $bench "final-fill-s11.png") "probe-lighthouse.png"
function RunText($graph) {
    $body = @{ prompt = $graph; client_id = "probe5" } | ConvertTo-Json -Depth 8 -Compress
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
function Gen($instruction, $image, $seed, $thinking, $systemTurn) {
    $inputs = @{ clip = @("1", 0); max_length = 120; sampling_mode = "on"; "sampling_mode.temperature" = 0.3; "sampling_mode.top_k" = 40; "sampling_mode.top_p" = 0.9; "sampling_mode.min_p" = 0.05; "sampling_mode.repetition_penalty" = 1.05; "sampling_mode.seed" = $seed; image = @("2", 0); thinking = $thinking; use_default_template = $true }
    if ($systemTurn) { $inputs.system_prompt = $editSys; $inputs.prompt = $instruction } else { $inputs.prompt = ($editSys + "`nInstruction: " + $instruction) }
    @{
        "1" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_4b_fp8_scaled.safetensors"; type = "krea2"; device = "default" } }
        "2" = @{ class_type = "LoadImage"; inputs = @{ image = $image; upload = "image" } }
        "3" = @{ class_type = "TextGenerate"; inputs = $inputs }
        "4" = @{ class_type = "PreviewAny"; inputs = @{ source = @("3", 0) } }
    }
}
$cases = @(
    @{ tag = "doodle portrait"; img = $doodle; instr = "turn this drawing into a hyper realistic portrait that looks similar" },
    @{ tag = "doodle realistic"; img = $doodle; instr = "make this hyper realistic" },
    @{ tag = "light stormy"; img = $light; instr = "make the sky stormy" },
    @{ tag = "light blue"; img = $light; instr = "turn it blue" }
)
foreach ($variant in @(@{ name = "think=false, user turn"; t = $false; s = $false }, @{ name = "think=true, user turn"; t = $true; s = $false }, @{ name = "think=true, system turn"; t = $true; s = $true }, @{ name = "think=false, system turn"; t = $false; s = $true })) {
    if ($Which -ne "all" -and $variant.name -notlike "*$Which*") { continue }
    "==== $($variant.name)"
    $empty = 0; $echo = 0; $ok = 0; $n = 0; $ms = 0
    foreach ($c in $cases) {
        foreach ($seed in 1..3) {
            $r = RunText (Gen $c.instr $c.img $seed $variant.t $variant.s)
            if ($r.err) { "  $($c.tag) s$seed : $($r.err)"; continue }
            $n++; $ms += $r.ms
            if ($r.text -eq "") { $empty++; $kind = "EMPTY" } elseif ($r.text -match "^(assistant|Assistant)") { $echo++; $kind = "echo " } else { $ok++; $kind = "ok   " }
            "  {0} {1,-16} s{2} {3,5} ms: {4}" -f $kind, $c.tag, $seed, $r.ms, $r.text.Substring(0, [Math]::Min(150, $r.text.Length))
        }
    }
    "  => {0} runs: {1} ok, {2} role echoes, {3} empty; mean {4} ms" -f $n, $ok, $echo, $empty, [int]($ms / [Math]::Max(1, $n))
}
