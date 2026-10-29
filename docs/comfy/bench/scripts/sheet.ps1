# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Contact sheet: -Names a,b,c (files in the bench dir), -Cols n, -Crop x,y,w,h (optional), -Out path, -Tile w,h
param([string]$Names, [int]$Cols = 3, [string]$Crop = "", [string]$Out, [string]$Tile = "420,320")
$list = @($Names -split ',' | Where-Object { $_ -ne "" })
$cropv = @($Crop -split ',' | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
$tsz = @($Tile -split ',' | ForEach-Object { [int]$_ })
Add-Type -AssemblyName System.Drawing
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$tw = $tsz[0]; $th = $tsz[1]
$rows = [int][Math]::Ceiling($list.Count / $Cols)
$sheet = New-Object System.Drawing.Bitmap ($tw * $Cols + 8 * ($Cols - 1)), ($th * $rows + 8 * ($rows - 1))
$g = [System.Drawing.Graphics]::FromImage($sheet)
$g.Clear([System.Drawing.Color]::Black)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$font = New-Object System.Drawing.Font "Consolas", 12
for ($i = 0; $i -lt $list.Count; $i++) {
    $img = [System.Drawing.Bitmap]::FromFile((Join-Path $bench $list[$i]))
    $src = if ($cropv.Count -eq 4) { New-Object System.Drawing.Rectangle $cropv[0], $cropv[1], $cropv[2], $cropv[3] } else { New-Object System.Drawing.Rectangle 0, 0, $img.Width, $img.Height }
    $s = [Math]::Min($tw / $src.Width, $th / $src.Height)
    $w = [int]($src.Width * $s); $h = [int]($src.Height * $s)
    $x = ($i % $Cols) * ($tw + 8); $y = [int][Math]::Floor($i / $Cols) * ($th + 8)
    $g.DrawImage($img, (New-Object System.Drawing.Rectangle $x, $y, $w, $h), $src, [System.Drawing.GraphicsUnit]::Pixel)
    $g.DrawString(($list[$i] -replace '\.png$', ''), $font, [System.Drawing.Brushes]::Yellow, $x + 4, $y + 4)
    $img.Dispose()
}
$g.Dispose()
$sheet.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$sheet.Dispose()
$Out
