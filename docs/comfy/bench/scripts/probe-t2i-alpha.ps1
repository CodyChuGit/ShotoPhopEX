# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Probe: does Qwen-Image-2.1 text-to-image give an RGBA PNG when asked for a transparent background?
param([string]$Prompt = "A small red wooden rowing boat, product photo, isolated on a transparent background, output a PNG image", [int]$Seed = 3, [string]$Tag = "probe-t2i-alpha")
$srv = "http://127.0.0.1:8188"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$graph = @{
    "1" = @{ class_type = "UNETLoader"; inputs = @{ unet_name = "qwen_image_2.1_int8_convrot.safetensors"; weight_dtype = "default" } }
    "2" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen3vl_8b_int8_convrot.safetensors"; type = "qwen_image"; device = "default" } }
    "3" = @{ class_type = "VAELoader"; inputs = @{ vae_name = "qwen_image_2.1_vae_bf16.safetensors" } }
    "4" = @{ class_type = "QwenImage21Cache"; inputs = @{ model = @("1", 0); device = "auto"; dtype = "default" } }
    "5" = @{ class_type = "TextEncodeQwenImage21"; inputs = @{ clip = @("2", 0); prompt = $Prompt; negative_prompt = ""; resolution = 1024 } }
    "6" = @{ class_type = "EmptyLatentImage"; inputs = @{ width = 1024; height = 1024; batch_size = 1 } }
    "7" = @{ class_type = "KSampler"; inputs = @{ model = @("4", 0); positive = @("5", 0); negative = @("5", 1); latent_image = @("6", 0); seed = $Seed; steps = 25; cfg = 1.0; sampler_name = "euler"; scheduler = "simple"; denoise = 1.0 } }
    "8" = @{ class_type = "VAEDecode"; inputs = @{ samples = @("7", 0); vae = @("3", 0) } }
    "9" = @{ class_type = "SaveImage"; inputs = @{ images = @("8", 0); filename_prefix = $Tag } }
}
$body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
$sw = [Diagnostics.Stopwatch]::StartNew()
$q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60
$pid2 = ($q.Content | ConvertFrom-Json).prompt_id
$done = $null
for ($i = 0; $i -lt 600; $i++) { Start-Sleep -Seconds 2; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
$sw.Stop()
if (-not $done) { "timed out"; exit 1 }
"status: " + $done.status.status_str + " in " + [int]$sw.Elapsed.TotalSeconds + " s"
if ($done.status.status_str -ne "success") { $done.status.messages | ConvertTo-Json -Depth 6 -Compress; exit 1 }
$img = $done.outputs."9".images[0]
$url = "$srv/view?filename=" + [Uri]::EscapeDataString($img.filename) + "&subfolder=" + [Uri]::EscapeDataString($img.subfolder) + "&type=" + $img.type
$dst = Join-Path $out "$Tag-s$Seed.png"
Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 120
$png = [IO.File]::ReadAllBytes($dst)
$ct = $png[25]
$ctName = switch ([int]$ct) { 0 { "gray" } 2 { "RGB" } 3 { "palette" } 4 { "gray+alpha" } 6 { "RGBA" } default { "?" } }
"saved $dst  colour type $ct ($ctName), " + $png.Length + " bytes"
Add-Type -AssemblyName System.Drawing
$bmp = [System.Drawing.Bitmap]::FromFile($dst)
$transparent = 0; $opaque = 0; $partial = 0
for ($y = 0; $y -lt $bmp.Height; $y += 4) { for ($x = 0; $x -lt $bmp.Width; $x += 4) { $a = $bmp.GetPixel($x, $y).A; if ($a -eq 0) { $transparent++ } elseif ($a -eq 255) { $opaque++ } else { $partial++ } } }
$bmp.Dispose()
"alpha histogram (sampled): transparent $transparent, partial $partial, opaque $opaque"
