# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Probe: run ComfyUI's official Qwen-Image-2.1 background-removal recipe on a test image and
# report whether the saved PNG carries an alpha channel.
param([string]$Image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png", [string]$Prompt = "Remove the background, and output a PNG image", [int]$Seed = 7, [string]$Tag = "probe-rgba")
$srv = "http://127.0.0.1:8188"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
Invoke-WebRequest -Uri "$srv/free" -Method Post -ContentType "application/json" -Body '{"unload_models": true, "free_memory": true}' -UseBasicParsing -TimeoutSec 30 | Out-Null
# upload
$bytes = [IO.File]::ReadAllBytes($Image)
$name = "probe-" + [IO.Path]::GetFileName($Image)
$boundary = [Guid]::NewGuid().ToString()
$lf = "`r`n"
$head = "--$boundary$lf" + "Content-Disposition: form-data; name=`"image`"; filename=`"$name`"$lf" + "Content-Type: image/png$lf$lf"
$tail = "$lf--$boundary--$lf"
$ms = New-Object IO.MemoryStream
$h = [Text.Encoding]::ASCII.GetBytes($head); $ms.Write($h, 0, $h.Length); $ms.Write($bytes, 0, $bytes.Length); $t = [Text.Encoding]::ASCII.GetBytes($tail); $ms.Write($t, 0, $t.Length)
$up = Invoke-WebRequest -Uri "$srv/upload/image" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $ms.ToArray() -UseBasicParsing -TimeoutSec 60
$upName = ($up.Content | ConvertFrom-Json).name
"uploaded as $upName"
$graph = @{
    "1" = @{ class_type = "UNETLoader"; inputs = @{ unet_name = "qwen_image_2.1_int8_convrot.safetensors"; weight_dtype = "default" } }
    "2" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_8b_int8_convrot.safetensors"; type = "qwen_image"; device = "default" } }
    "3" = @{ class_type = "VAELoader"; inputs = @{ vae_name = "qwen_image_2.1_vae_bf16.safetensors" } }
    "4" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "5" = @{ class_type = "QwenImage21Cache"; inputs = @{ model = @("1", 0); device = "auto"; dtype = "default" } }
    "6" = @{ class_type = "TextEncodeQwenImage21"; inputs = @{ clip = @("2", 0); vae = @("3", 0); prompt = $Prompt; negative_prompt = ""; resolution = 0; "images.image_1" = @("4", 0) } }
    "10" = @{ class_type = "KSampler"; inputs = @{ model = @("5", 0); positive = @("6", 0); negative = @("6", 1); latent_image = @("6", 2); seed = $Seed; steps = 25; cfg = 1.0; sampler_name = "euler"; scheduler = "simple"; denoise = 1.0 } }
    "11" = @{ class_type = "VAEDecode"; inputs = @{ samples = @("10", 0); vae = @("3", 0) } }
    "12" = @{ class_type = "SaveImage"; inputs = @{ images = @("11", 0); filename_prefix = $Tag } }
}
$body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
$sw = [Diagnostics.Stopwatch]::StartNew()
$q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60
$pid2 = ($q.Content | ConvertFrom-Json).prompt_id
"queued $pid2"
$done = $null
for ($i = 0; $i -lt 600; $i++) {
    Start-Sleep -Seconds 2
    $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json
    $entry = $hist.$pid2
    if ($entry) { $done = $entry; break }
}
$sw.Stop()
if (-not $done) { "timed out"; exit 1 }
"status: " + $done.status.status_str + " in " + [int]$sw.Elapsed.TotalSeconds + " s"
if ($done.status.status_str -ne "success") { $done.status.messages | ConvertTo-Json -Depth 6 -Compress; exit 1 }
$img = $done.outputs."12".images[0]
$url = "$srv/view?filename=" + [Uri]::EscapeDataString($img.filename) + "&subfolder=" + [Uri]::EscapeDataString($img.subfolder) + "&type=" + $img.type
$dst = Join-Path $out "$Tag-s$Seed.png"
Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 120
$png = [IO.File]::ReadAllBytes($dst)
# IHDR: width(4) height(4) depth(1) colour type(1) at offset 16..
$w = [BitConverter]::ToUInt32([byte[]]($png[16..19] | Sort-Object -Descending { [array]::IndexOf($png[16..19], $_) }), 0)
$ct = $png[25]; $depth = $png[24]
$ctName = switch ([int]$ct) { 0 { "gray" } 2 { "RGB" } 3 { "palette" } 4 { "gray+alpha" } 6 { "RGBA" } default { "?" } }
"saved $dst  colour type $ct ($ctName), depth $depth, " + $png.Length + " bytes"
Add-Type -AssemblyName System.Drawing
$bmp = [System.Drawing.Bitmap]::FromFile($dst)
"size " + $bmp.Width + "x" + $bmp.Height + ", pixel format " + $bmp.PixelFormat
$transparent = 0; $opaque = 0; $partial = 0
for ($y = 0; $y -lt $bmp.Height; $y += 4) { for ($x = 0; $x -lt $bmp.Width; $x += 4) { $a = $bmp.GetPixel($x, $y).A; if ($a -eq 0) { $transparent++ } elseif ($a -eq 255) { $opaque++ } else { $partial++ } } }
$bmp.Dispose()
"alpha histogram (sampled): transparent $transparent, partial $partial, opaque $opaque"
