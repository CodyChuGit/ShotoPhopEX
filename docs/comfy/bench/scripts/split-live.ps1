# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Live: Split into Layers on the lighthouse through PhotoCraft (integration + timing), and the
# same graph straight against ComfyUI to save each layer as its own PNG for a look.
param([int]$Layers = 3, [int]$Seed = 5)
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$srv = "http://127.0.0.1:8188"
$model = "C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\qwen_image_layered_fp8mixed.safetensors"
for ($i = 0; $i -lt 360; $i++) { $sz = (Get-Item $model -ErrorAction SilentlyContinue).Length; if ($sz -ge 19100000000) { break }; Start-Sleep -Seconds 10 }
"model file: {0:N2} GB" -f ((Get-Item $model).Length / 1GB)
Invoke-WebRequest -Uri "$srv/free" -Method Post -ContentType "application/json" -Body '{"unload_models": true, "free_memory": true}' -UseBasicParsing -TimeoutSec 30 | Out-Null
$env:SP_IMAGE = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png"
$env:SP_PARAMS = '{\"layers\":' + $Layers + ',\"prompt\":\"a white lighthouse on dark rocks, a small red rowing boat on calm water, a pink evening sky\",\"seed\":' + $Seed + '}'
$env:SP_OUT = Join-Path $out "split-composite.png"
$sw = [Diagnostics.Stopwatch]::StartNew()
$lines = & $cli --% run %SP_IMAGE% --cmd generate.splitLayers --params "%SP_PARAMS%" --out %SP_OUT%
$sw.Stop()
foreach ($l in $lines) { if ("$l" -match '"runId"') { $r = ("$l" | ConvertFrom-Json).result; "photocraft: {0} layers in {1} ms (wall {2} ms), sent {3}x{4} via {5}" -f $r.count, $r.ms, $sw.ElapsedMilliseconds, $r.requestWidth, $r.requestHeight, $r.template } elseif ("$l" -match "rror") { $l } }
# The same request straight at the server, saving each layer.
Add-Type -AssemblyName System.Drawing
$src = [System.Drawing.Bitmap]::FromFile($env:SP_IMAGE)
$scale = 640.0 / [Math]::Max($src.Width, $src.Height)
$rw = [int]([Math]::Round($src.Width * $scale) / 16) * 16; $rh = [int]([Math]::Round($src.Height * $scale) / 16) * 16
$small = New-Object System.Drawing.Bitmap $rw, $rh
$g = [System.Drawing.Graphics]::FromImage($small); $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic; $g.DrawImage($src, 0, 0, $rw, $rh); $g.Dispose(); $src.Dispose()
$tmp = Join-Path $env:TEMP "split-probe-input.png"; $small.Save($tmp, [System.Drawing.Imaging.ImageFormat]::Png); $small.Dispose()
$bytes = [IO.File]::ReadAllBytes($tmp)
$boundary = [Guid]::NewGuid().ToString(); $lf = "`r`n"
$head = "--$boundary$lf" + "Content-Disposition: form-data; name=`"image`"; filename=`"split-probe-input.png`"$lf" + "Content-Type: image/png$lf$lf"
$tail = "$lf--$boundary--$lf"
$ms = New-Object IO.MemoryStream
$h = [Text.Encoding]::ASCII.GetBytes($head); $ms.Write($h, 0, $h.Length); $ms.Write($bytes, 0, $bytes.Length); $t = [Text.Encoding]::ASCII.GetBytes($tail); $ms.Write($t, 0, $t.Length)
$up = Invoke-WebRequest -Uri "$srv/upload/image" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $ms.ToArray() -UseBasicParsing -TimeoutSec 60
$upName = ($up.Content | ConvertFrom-Json).name
$prompt = "a white lighthouse on dark rocks, a small red rowing boat on calm water, a pink evening sky"
$graph = @{
    "1" = @{ class_type = "UNETLoader"; inputs = @{ unet_name = "qwen_image_layered_fp8mixed.safetensors"; weight_dtype = "default" } }
    "2" = @{ class_type = "CLIPLoader"; inputs = @{ clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"; type = "qwen_image"; device = "default" } }
    "3" = @{ class_type = "VAELoader"; inputs = @{ vae_name = "qwen_image_layered_vae.safetensors" } }
    "4" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "5" = @{ class_type = "VAEEncode"; inputs = @{ pixels = @("4", 0); vae = @("3", 0) } }
    "6" = @{ class_type = "CLIPTextEncode"; inputs = @{ clip = @("2", 0); text = $prompt } }
    "7" = @{ class_type = "CLIPTextEncode"; inputs = @{ clip = @("2", 0); text = "" } }
    "8" = @{ class_type = "ModelSamplingAuraFlow"; inputs = @{ model = @("1", 0); shift = 1.0 } }
    "9" = @{ class_type = "ReferenceLatent"; inputs = @{ conditioning = @("6", 0); latent = @("5", 0) } }
    "10" = @{ class_type = "ReferenceLatent"; inputs = @{ conditioning = @("7", 0); latent = @("5", 0) } }
    "11" = @{ class_type = "EmptyQwenImageLayeredLatentImage"; inputs = @{ width = $rw; height = $rh; layers = $Layers; batch_size = 1 } }
    "12" = @{ class_type = "KSampler"; inputs = @{ model = @("8", 0); positive = @("9", 0); negative = @("10", 0); latent_image = @("11", 0); seed = $Seed; steps = 20; cfg = 2.5; sampler_name = "euler"; scheduler = "simple"; denoise = 1.0 } }
    "13" = @{ class_type = "LatentCut"; inputs = @{ samples = @("12", 0); dim = "t"; index = 1; amount = 16384 } }
    "14" = @{ class_type = "LatentCutToBatch"; inputs = @{ samples = @("13", 0); dim = "t"; slice_size = 1 } }
    "15" = @{ class_type = "VAEDecode"; inputs = @{ samples = @("14", 0); vae = @("3", 0) } }
    "16" = @{ class_type = "SaveImage"; inputs = @{ images = @("15", 0); filename_prefix = "split-probe" } }
}
$body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
$sw = [Diagnostics.Stopwatch]::StartNew()
try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60 } catch { "queue failed: " + $_.ErrorDetails.Message; exit 1 }
$pid2 = ($q.Content | ConvertFrom-Json).prompt_id
$done = $null
for ($i = 0; $i -lt 600; $i++) { Start-Sleep -Seconds 2; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
$sw.Stop()
"probe: " + $done.status.status_str + " in " + [int]$sw.Elapsed.TotalSeconds + " s"
if ($done.status.status_str -ne "success") { $done.status.messages | ConvertTo-Json -Depth 6 -Compress; exit 1 }
$k = 0
foreach ($img in $done.outputs."16".images) {
    $url = "$srv/view?filename=" + [Uri]::EscapeDataString($img.filename) + "&subfolder=" + [Uri]::EscapeDataString($img.subfolder) + "&type=" + $img.type
    $dst = Join-Path $out ("split-layer-{0}.png" -f $k)
    Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 120
    $bmp = [System.Drawing.Bitmap]::FromFile($dst)
    $tr = 0; $op = 0; $pa = 0
    for ($y = 0; $y -lt $bmp.Height; $y += 4) { for ($x = 0; $x -lt $bmp.Width; $x += 4) { $a = $bmp.GetPixel($x, $y).A; if ($a -eq 0) { $tr++ } elseif ($a -eq 255) { $op++ } else { $pa++ } } }
    "layer {0}: {1}x{2} {3}, alpha transparent {4} partial {5} opaque {6} -> {7}" -f $k, $bmp.Width, $bmp.Height, $bmp.PixelFormat, $tr, $pa, $op, $dst
    $bmp.Dispose(); $k++
}
