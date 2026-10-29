# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Probe: SAM3_Detect with a literal positive_coords point (API format) on the lighthouse photo.
param([string]$Image = "C:\Users\5090\ComfyUI\photocraft-tests\bench\final-fill-s11.png", [int]$X = 350, [int]$Y = 665, [string]$Tag = "probe-sam-point")
$srv = "http://127.0.0.1:8188"
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$bytes = [IO.File]::ReadAllBytes($Image)
$name = "probe-" + [IO.Path]::GetFileName($Image)
$boundary = [Guid]::NewGuid().ToString(); $lf = "`r`n"
$head = "--$boundary$lf" + "Content-Disposition: form-data; name=`"image`"; filename=`"$name`"$lf" + "Content-Type: image/png$lf$lf"
$tail = "$lf--$boundary--$lf"
$ms = New-Object IO.MemoryStream
$h = [Text.Encoding]::ASCII.GetBytes($head); $ms.Write($h, 0, $h.Length); $ms.Write($bytes, 0, $bytes.Length); $t = [Text.Encoding]::ASCII.GetBytes($tail); $ms.Write($t, 0, $t.Length)
$up = Invoke-WebRequest -Uri "$srv/upload/image" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $ms.ToArray() -UseBasicParsing -TimeoutSec 60
$upName = ($up.Content | ConvertFrom-Json).name
$points = '[{"x":' + $X + ',"y":' + $Y + '}]'
$graph = @{
    "1" = @{ class_type = "CheckpointLoaderSimple"; inputs = @{ ckpt_name = "sam3.1_multiplex_fp16.safetensors" } }
    "2" = @{ class_type = "LoadImage"; inputs = @{ image = $upName; upload = "image" } }
    "4" = @{ class_type = "SAM3_Detect"; inputs = @{ model = @("1", 0); image = @("2", 0); threshold = 0.5; refine_iterations = 2; individual_masks = $true; positive_coords = $points } }
    "5" = @{ class_type = "MaskToImage"; inputs = @{ mask = @("4", 0) } }
    "6" = @{ class_type = "SaveImage"; inputs = @{ images = @("5", 0); filename_prefix = $Tag } }
}
$body = @{ prompt = $graph; client_id = "probe" } | ConvertTo-Json -Depth 8 -Compress
$sw = [Diagnostics.Stopwatch]::StartNew()
try { $q = Invoke-WebRequest -Uri "$srv/prompt" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 60 } catch { "queue failed: " + $_.ErrorDetails.Message; exit 1 }
$pid2 = ($q.Content | ConvertFrom-Json).prompt_id
$done = $null
for ($i = 0; $i -lt 300; $i++) { Start-Sleep -Seconds 1; $hist = (Invoke-WebRequest -Uri "$srv/history/$pid2" -UseBasicParsing -TimeoutSec 30).Content | ConvertFrom-Json; $entry = $hist.$pid2; if ($entry) { $done = $entry; break } }
$sw.Stop()
if (-not $done) { "timed out"; exit 1 }
"status: " + $done.status.status_str + " in " + [int]$sw.Elapsed.TotalSeconds + " s"
if ($done.status.status_str -ne "success") { $done.status.messages | ConvertTo-Json -Depth 6 -Compress; exit 1 }
$imgs = $done.outputs."6".images
"masks: " + $imgs.Count
Add-Type -AssemblyName System.Drawing
$k = 0
foreach ($img in $imgs) {
    $url = "$srv/view?filename=" + [Uri]::EscapeDataString($img.filename) + "&subfolder=" + [Uri]::EscapeDataString($img.subfolder) + "&type=" + $img.type
    $dst = Join-Path $out ("{0}-{1}.png" -f $Tag, $k)
    Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 120
    $bmp = [System.Drawing.Bitmap]::FromFile($dst)
    $n = 0; $x0 = 99999; $y0 = 99999; $x1 = -1; $y1 = -1
    for ($y = 0; $y -lt $bmp.Height; $y += 2) { for ($x = 0; $x -lt $bmp.Width; $x += 2) { if ($bmp.GetPixel($x, $y).R -gt 127) { $n++; if ($x -lt $x0) { $x0 = $x }; if ($y -lt $y0) { $y0 = $y }; if ($x -gt $x1) { $x1 = $x }; if ($y -gt $y1) { $y1 = $y } } } }
    $bmp.Dispose()
    "mask {0}: {1} px (sampled /4), bbox {2},{3}-{4},{5} -> {6}" -f $k, $n, $x0, $y0, $x1, $y1, $dst
    $k++
}
