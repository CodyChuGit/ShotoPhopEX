# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Contact sheets for the swap benchmark (swap-run.ps1): one row per case, the original then each
# route's picture for one seed, into bench\swap\sheet-<kind>-s<seed>.png.
param([string]$Routes = "fill21,edit21,editk2,fill2511", [string]$Seeds = "7,11", [string]$Tile = "300,300")
Add-Type -AssemblyName System.Drawing
$root = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$out = Join-Path $root "swap"
$spec = Get-Content (Join-Path $PSScriptRoot "swap-cases.json") -Raw | ConvertFrom-Json
$routeList = @($Routes -split ',')
$tsz = @($Tile -split ',' | ForEach-Object { [int]$_ })
$tw = $tsz[0]; $th = $tsz[1]
$font = New-Object System.Drawing.Font "Consolas", 10
foreach ($seed in ($Seeds -split ',')) {
    foreach ($kind in "item", "outfit") {
        $cases = @($spec.cases | Where-Object { $_.kind -eq $kind })
        $cols = 1 + $routeList.Count
        $sheet = New-Object System.Drawing.Bitmap ($tw * $cols + 6 * ($cols - 1)), ($th * $cases.Count + 6 * ($cases.Count - 1))
        $g = [System.Drawing.Graphics]::FromImage($sheet)
        $g.Clear([System.Drawing.Color]::Black)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        for ($r = 0; $r -lt $cases.Count; $r++) {
            $c = $cases[$r]
            $src = Join-Path $root ($c.image -replace '^stock/', 'stock\' -replace '^gen/', 'swap\gen\')
            $cells = New-Object System.Collections.Generic.List[object]
            $cells.Add([pscustomobject]@{ Path = $src; Label = "$($c.id): $($c.fill)" })
            foreach ($rt in $routeList) { $cells.Add([pscustomobject]@{ Path = (Join-Path $out "$($c.id)-$rt-s$seed.png"); Label = $rt }) }
            for ($k = 0; $k -lt $cells.Count; $k++) {
                $x = $k * ($tw + 6); $y = $r * ($th + 6)
                if (Test-Path $cells[$k].Path) {
                    $img = [System.Drawing.Bitmap]::FromFile($cells[$k].Path)
                    $sc = [Math]::Min($tw / $img.Width, $th / $img.Height)
                    $g.DrawImage($img, (New-Object System.Drawing.Rectangle $x, $y, ([int]($img.Width * $sc)), ([int]($img.Height * $sc))))
                    $img.Dispose()
                } else {
                    $g.DrawString("(missing)", $font, [System.Drawing.Brushes]::Red, $x + 4, $y + 20)
                }
                $g.FillRectangle((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(160, 0, 0, 0))), $x, $y, $tw, 16)
                $g.DrawString($cells[$k].Label, $font, [System.Drawing.Brushes]::Yellow, $x + 2, $y + 1)
            }
        }
        $g.Dispose()
        $path = Join-Path $out "sheet-$kind-s$seed.png"
        $sheet.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        $sheet.Dispose()
        $path
    }
}
