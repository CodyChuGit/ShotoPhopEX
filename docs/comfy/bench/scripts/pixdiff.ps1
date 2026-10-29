# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Mean absolute RGB difference (sampled every 4th pixel) between pairs "a.png=b.png,c.png=d.png" in the bench dir.
param([string]$Pairs)
Add-Type -AssemblyName System.Drawing
$bench = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
foreach ($pair in ($Pairs -split ',')) {
    $ab = $pair -split '='
    $pa = Join-Path $bench $ab[0]; $pb = Join-Path $bench $ab[1]
    if (-not (Test-Path $pa) -or -not (Test-Path $pb)) { "{0}: missing file" -f $pair; continue }
    $ia = [System.Drawing.Bitmap]::FromFile($pa); $ib = [System.Drawing.Bitmap]::FromFile($pb)
    if ($ia.Width -ne $ib.Width -or $ia.Height -ne $ib.Height) { "{0}: size differs" -f $pair; $ia.Dispose(); $ib.Dispose(); continue }
    $sum = 0.0; $n = 0; $max = 0
    for ($y = 0; $y -lt $ia.Height; $y += 4) { for ($x = 0; $x -lt $ia.Width; $x += 4) {
        $p = $ia.GetPixel($x, $y); $q = $ib.GetPixel($x, $y)
        $d = [Math]::Abs($p.R - $q.R) + [Math]::Abs($p.G - $q.G) + [Math]::Abs($p.B - $q.B)
        $sum += $d; $n++; if ($d -gt $max) { $max = $d }
    } }
    $ia.Dispose(); $ib.Dispose()
    "{0,-28} vs {1,-28} mean |d| = {2:N2} / 765, max {3}" -f $ab[0], $ab[1], ($sum / $n), $max
}
