# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# One row a picture: the source, then each camera on each route. Labels on top.
param([string]$Routes = "edit21,editk2", [string]$Out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\angles\sheet.png", [int]$Tile = 220)
Add-Type -AssemblyName System.Drawing
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$pics = [ordered]@{ denim = "$bench\swap\gen\gen-denim.png"; tshirt = "$bench\swap\gen\gen-tshirt.png"; bench = "$bench\stock\st-bench.jpg"; dog = "$bench\stock\st-dog.jpg" }
$cams = "side2", "back2", "high2", "close2"
$routeList = $Routes.Split(",")
$cols = 1 + $cams.Count * $routeList.Count
$head = 22
$bmp = New-Object System.Drawing.Bitmap ($cols * $Tile), ($head + $pics.Count * $Tile)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::White)
$font = New-Object System.Drawing.Font "Segoe UI", 9
$labels = @("source") + ($routeList | ForEach-Object { $r = $_; $cams | ForEach-Object { "$_ $r" } })
for ($i = 0; $i -lt $labels.Count; $i++) { $g.DrawString($labels[$i], $font, [System.Drawing.Brushes]::Black, ($i * $Tile + 4), 3) }
$row = 0
foreach ($p in $pics.Keys) {
    $files = @($pics[$p]) + ($routeList | ForEach-Object { $r = $_; $cams | ForEach-Object { "$bench\angles\$p-$_-$r.png" } })
    for ($i = 0; $i -lt $files.Count; $i++) {
        if (-not (Test-Path $files[$i])) { continue }
        $im = [System.Drawing.Image]::FromFile($files[$i])
        $sc = [Math]::Min($Tile / $im.Width, $Tile / $im.Height)
        $w = [int]($im.Width * $sc); $h = [int]($im.Height * $sc)
        $g.DrawImage($im, ($i * $Tile + ($Tile - $w) / 2), ($head + $row * $Tile + ($Tile - $h) / 2), $w, $h)
        $im.Dispose()
    }
    $row++
}
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
$Out
