# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Contact sheets of the robustness set: a row per scene (original, half, whole), cropped
# around the object (its box doubled, at least 400 px), stacked -PerSheet rows per sheet.
# Output: bench\rb-sheet-<n>.png. -Tags a,b,c limits the scenes.
# -Dir, -Prefix and -Objects point it at another set, as robust-strokes.ps1's do.
# -Suffix compares a second run's outputs (<Prefix>-<tag>-half<Suffix>.png) in place of the whole stroke.
# -Compare half|whole picks which stroke the suffixed run is compared on; -Base is the suffix of
# the run it is compared against (default: the unsuffixed one).
param([string]$Tags = "", [int]$PerSheet = 7, [string]$Dir = "C:\Users\5090\ComfyUI\photocraft-tests\bench", [string]$Prefix = "rb", [string]$Objects = "rb-objects.json", [string]$Suffix = "", [string]$Compare = "half", [string]$Base = "")
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$out = $Dir
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$rel = if ($out -eq $bench) { "" } else { ($out.Substring($bench.Length).TrimStart('\')) + "\" }
$sheet = (Resolve-Path "docs\comfy\bench\scripts\sheet.ps1").Path
$set = Get-Content (Join-Path $out $Objects) -Raw | ConvertFrom-Json
$want = @($Tags -split ',' | Where-Object { $_ -ne "" })
Add-Type -AssemblyName System.Drawing
$strips = @()
foreach ($p in $set.PSObject.Properties) {
    $tag = $p.Name; $o = $p.Value
    if ($null -eq $o) { continue }
    if ($want.Count -gt 0 -and $want -notcontains $tag) { continue }
    $image = Get-ChildItem (Join-Path $out "$Prefix-$tag.*") | Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' } | Select-Object -First 1
    if ($null -eq $image) { continue }
    $bmp = [System.Drawing.Bitmap]::FromFile($image.FullName); $iw = $bmp.Width; $ih = $bmp.Height; $bmp.Dispose()
    $x = $o.bounds[0]; $y = $o.bounds[1]; $w = $o.bounds[2]; $h = $o.bounds[3]
    $side = [Math]::Min([Math]::Min($iw, $ih), [Math]::Max(400, [Math]::Max($w, $h) * 2))
    $cx = $x + $w / 2; $cy = $y + $h / 2
    $x0 = [int][Math]::Max(0, [Math]::Min($iw - $side, $cx - $side / 2)); $y0 = [int][Math]::Max(0, [Math]::Min($ih - $side, $cy - $side / 2))
    $strip = Join-Path $out "$Prefix-$tag-strip.png"
    $second = if ($Suffix -eq "") { "$rel$Prefix-$tag-half.png" } else { "$rel$Prefix-$tag-$Compare$Base.png" }
    $third = if ($Suffix -eq "") { "$rel$Prefix-$tag-whole.png" } else { "$rel$Prefix-$tag-$Compare$Suffix.png" }
    & powershell -NoProfile -File $sheet -Names "$rel$($image.Name),$second,$third" -Cols 3 -Crop "$x0,$y0,$side,$side" -Tile "400,400" -Out $strip | Out-Null
    $strips += $strip
}
$n = 0
for ($i = 0; $i -lt $strips.Count; $i += $PerSheet) {
    $n++
    $part = $strips[$i..([Math]::Min($i + $PerSheet, $strips.Count) - 1)]
    $imgs = $part | ForEach-Object { [System.Drawing.Bitmap]::FromFile($_) }
    $wmax = [int](($imgs | ForEach-Object { $_.Width } | Measure-Object -Maximum).Maximum)
    $htot = [int](($imgs | ForEach-Object { $_.Height } | Measure-Object -Sum).Sum + 6 * ($imgs.Count - 1))
    $bmp = New-Object System.Drawing.Bitmap -ArgumentList $wmax, $htot
    $g = [System.Drawing.Graphics]::FromImage($bmp); $g.Clear([System.Drawing.Color]::Black)
    $yy = 0
    foreach ($im in $imgs) { $g.DrawImage($im, 0, $yy, $im.Width, $im.Height); $yy += $im.Height + 6; $im.Dispose() }
    $g.Dispose()
    $file = Join-Path $out "$Prefix-sheet-$n.png"
    $bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
    $file
}
