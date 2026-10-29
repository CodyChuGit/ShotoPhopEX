# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# The removal rewriter's naming alone, on a running native worker (2026-10-26): the two views
# the engine shows it (the surroundings left, a close-up of the stroke right, as
# `generative_remove` crops them) made here for each scene's half and whole stroke, then one
# question each through `chat_probe` (crates/genai/examples). Seconds a wording, where a full
# removal run is minutes.
#   -System   the system turn, a text file (the engine's edit system plus the removal note)
#   -Close    the close-up's margin around the grown stroke, a share of its longer side (0.25)
#   -Wide     the surroundings' least side in pixels (384, `REMOVE_MIN_REQUEST`)
#   -WideScale when above 0, the surroundings are the close-up this many times as wide and high
#   -Dim      when above 0, the close-up outside the grown stroke darkened by this much (0..1)
#   -Tag      names the made pictures apart
param([string]$System, [string]$Tags = "glasses,watch,tattoo,cup,dog,boat,bicycle,hat,umbrella,motorbike",
    [double]$Close = 0.25, [int]$Wide = 384, [double]$WideScale = 0, [double]$Dim = 0, [string]$Tag = "base", [int]$Seed = 7,
    [string]$Dir = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock", [string]$Prefix = "st", [string]$Objects = "st-objects.json",
    [string]$Probe = "C:\Users\5090\Documents\PhotoshopEX\target\release\examples\chat_probe.exe",
    [string]$Prompt = "Remove the object in the selected area and fill its place with the surrounding scene continued seamlessly (its background, surfaces, textures, patterns and lines), as if nothing had ever been there")
Add-Type -AssemblyName System.Drawing
$out = Join-Path $Dir "naming-$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$objs = Get-Content (Join-Path $Dir $Objects) -Raw | ConvertFrom-Json
function Clamp([int]$v, [int]$lo, [int]$hi) { [Math]::Max($lo, [Math]::Min($hi, $v)) }
function Grow-Rect($r, [int]$m, $W, $H) { @((Clamp ($r[0] - $m) 0 $W), (Clamp ($r[1] - $m) 0 $H), (Clamp ($r[2] + $m) 0 $W), (Clamp ($r[3] + $m) 0 $H)) }
function AtLeast($r, [int]$min, $W, $H) {
    $span = { param($lo, $hi, $chi) $len = $hi - $lo; if ($len -ge $min) { return @($lo, $hi) }; $want = [Math]::Min($min, $chi); $lo2 = [Math]::Min([Math]::Max(0, $lo - [int](($want - $len) / 2)), $chi - $want); @($lo2, ($lo2 + $want)) }
    $x = & $span $r[0] $r[2] $W; $y = & $span $r[1] $r[3] $H; @($x[0], $y[0], $x[1], $y[1])
}
function Crop($bmp, $r) { $w = $r[2] - $r[0]; $h = $r[3] - $r[1]; $bmp.Clone((New-Object System.Drawing.Rectangle $r[0], $r[1], $w, $h), $bmp.PixelFormat) }
$pictures = @()
foreach ($t in $Tags.Split(",")) {
    $o = $objs.$t
    if (-not $o) { continue }
    $src = Get-ChildItem $Dir -Filter "$Prefix-$t.*" | Where-Object { $_.Extension -match "jpg|png" } | Select-Object -First 1
    $bmp = [System.Drawing.Bitmap]::FromFile($src.FullName)
    $IW = $bmp.Width; $IH = $bmp.Height
    $x, $y, $w, $h = $o.bounds
    $s = [Math]::Round([Math]::Min([Math]::Max([Math]::Min($w, $h) / 2, 12), 64))
    $cx = $x + [int]($w / 2); $cy = $y + [int]($h / 2)
    $strokes = @{}
    if ($w -ge $h) { $strokes.half = @(($x + [int](0.05 * $w) - $s / 2), ($cy - $s / 2), ($x + [int](0.5 * $w) + $s / 2), ($cy + $s / 2)) }
    else { $strokes.half = @(($cx - $s / 2), ($y + [int](0.05 * $h) - $s / 2), ($cx + $s / 2), ($y + [int](0.5 * $h) + $s / 2)) }
    $strokes.whole = @($x, $y, ($x + $w), ($y + $h))
    foreach ($k in "half", "whole") {
        $hole = $strokes.$k | ForEach-Object { [int]$_ }
        $side = [Math]::Max($hole[2] - $hole[0], $hole[3] - $hole[1])
        $grow = Clamp ([Math]::Round($side * 0.15)) 16 96
        $g = Grow-Rect $hole $grow $IW $IH
        $gs = [Math]::Max($g[2] - $g[0], $g[3] - $g[1])
        $closeR = Grow-Rect $g ([int]($Close * $gs)) $IW $IH
        if ($WideScale -gt 0) {
            $cw = $closeR[2] - $closeR[0]; $ch = $closeR[3] - $closeR[1]
            $wideR = AtLeast @($closeR[0], $closeR[1], $closeR[2], $closeR[3]) ([int]([Math]::Max($cw, $ch) * $WideScale)) $IW $IH
        } else { $wideR = AtLeast $closeR $Wide $IW $IH }
        $wideR = AtLeast $wideR $Wide $IW $IH
        $left = Crop $bmp $wideR; $right = Crop $bmp $closeR
        if ($Dim -gt 0) {
            # the close-up's surroundings of the stroke dimmed (a spotlight on what was painted)
            $rg = [System.Drawing.Graphics]::FromImage($right)
            $region = New-Object System.Drawing.Region (New-Object System.Drawing.Rectangle 0, 0, $right.Width, $right.Height)
            $region.Exclude((New-Object System.Drawing.Rectangle ($g[0] - $closeR[0]), ($g[1] - $closeR[1]), ($g[2] - $g[0]), ($g[3] - $g[1])))
            $rg.FillRegion((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb([int](255 * $Dim), 0, 0, 0))), $region)
            $rg.Dispose()
        }
        $lh = $left.Height
        $rw = [Math]::Max(1, [int][Math]::Round($right.Width * $lh / [Math]::Max(1, $right.Height)))
        $tw = $left.Width + $rw
        # the request size: at most about a megapixel, as `request_size` keeps it
        $scale = [Math]::Min(1.0, [Math]::Sqrt(1048576.0 / ($tw * $lh)))
        $ow = [Math]::Max(64, [int]($tw * $scale)); $oh = [Math]::Max(64, [int]($lh * $scale))
        $pic = New-Object System.Drawing.Bitmap $ow, $oh
        $gr = [System.Drawing.Graphics]::FromImage($pic)
        $gr.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $lw = [int]($left.Width * $scale)
        $gr.DrawImage($left, 0, 0, $lw, $oh)
        $gr.DrawImage($right, $lw, 0, $ow - $lw, $oh)
        $gr.Dispose()
        $file = Join-Path $out "$t-$k.png"
        $pic.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
        $pic.Dispose(); $left.Dispose(); $right.Dispose()
        $pictures += $file
    }
    $bmp.Dispose()
}
& $Probe $System $Prompt $Seed @pictures
