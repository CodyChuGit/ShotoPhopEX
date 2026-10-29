# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# A real-photograph set for the generative removal: one public-domain or CC0 photograph per
# subject below from Wikimedia Commons (its search API, the first hit whose licence the file's
# own metadata gives as CC0 or Public domain and whose width is at least 1200 px), saved at
# 1600 px on the longer side as bench\stock\st-<tag>.jpg, with bench\stock\st-sources.txt
# listing each file's Commons page, author and licence. Nothing is downloaded whose licence is
# not one of those two. -Only a,b,c limits the subjects.
param([string]$Only = "")
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench\stock"
New-Item -ItemType Directory -Force $out | Out-Null
$ua = @{ "User-Agent" = "PhotoshopEX-bench/1.0 (https://github.com/CodyChuGit/comfy-photocraft)" }
$api = "https://commons.wikimedia.org/w/api.php"
$subjects = @(
    @("glasses",   "glasses",     "portrait man wearing glasses photograph"),
    @("watch",     "wristwatch",  "wristwatch on wrist photograph"),
    @("necklace",  "necklace",    "woman wearing necklace portrait photograph"),
    @("earring",   "earring",     "earring ear close-up photograph"),
    @("ring",      "ring",        "ring on finger hand photograph"),
    @("tattoo",    "tattoo",      "tattoo on arm photograph"),
    @("car",       "car",         "car parked street photograph"),
    @("sign",      "sign",        "street sign on wall photograph"),
    @("bird",      "bird",        "bird on wire photograph"),
    @("cup",       "cup",         "coffee cup on table photograph"),
    @("poster",    "poster",      "poster on wall photograph"),
    @("dog",       "dog",         "dog on grass lawn photograph"),
    @("lamppost",  "lamp post",   "lamp post park photograph"),
    @("beach",     "person",      "person walking on beach photograph"),
    @("cat",       "cat",         "cat on sofa photograph"),
    @("bicycle",   "bicycle",     "bicycle leaning against wall photograph"),
    @("boat",      "boat",        "small boat on lake photograph"),
    @("cone",      "traffic cone","traffic cone on road photograph"),
    @("umbrella",  "umbrella",    "person with umbrella street photograph"),
    @("hat",       "hat",         "man wearing hat portrait photograph"),
    @("bench",     "bench",       "park bench photograph"),
    @("bottle",    "bottle",      "bottle on table photograph"),
    @("cow",       "cow",         "cow in field photograph"),
    @("hydrant",   "fire hydrant","fire hydrant sidewalk photograph"),
    @("kite",      "kite",        "kite in sky photograph"),
    @("motorbike", "motorcycle",  "motorcycle parked photograph")
)
$want = @($Only -split ',' | Where-Object { $_ -ne "" })
$sources = New-Object System.Collections.ArrayList
foreach ($s in $subjects) {
    $tag = $s[0]; $name = $s[1]; $query = $s[2]
    if ($want.Count -gt 0 -and $want -notcontains $tag) { continue }
    $file = Join-Path $out "st-$tag.jpg"
    if (Test-Path $file) { "{0,-10} already there" -f $tag; continue }
    $picked = $null
    foreach ($q in @("$query incategory:CC-Zero", "$query incategory:PD-self", $query)) {
        $r = Invoke-RestMethod -Headers $ua -Uri ($api + "?action=query&list=search&srnamespace=6&srlimit=12&format=json&srsearch=" + [uri]::EscapeDataString("$q filemime:image/jpeg"))
        foreach ($hit in $r.query.search) {
            $title = $hit.title
            $info = Invoke-RestMethod -Headers $ua -Uri ($api + "?action=query&prop=imageinfo&iiprop=url|size|extmetadata&iiurlwidth=1600&format=json&titles=" + [uri]::EscapeDataString($title))
            $page = $info.query.pages.PSObject.Properties.Value | Select-Object -First 1
            $ii = $page.imageinfo | Select-Object -First 1
            if ($null -eq $ii) { continue }
            $lic = "$($ii.extmetadata.LicenseShortName.value)"
            if ($lic -notmatch '^(CC0|Public domain)' ) { continue }
            if ($ii.width -lt 1200 -or $ii.height -lt 800) { continue }
            $picked = @{ title = $title; url = $ii.thumburl; lic = $lic; author = ("$($ii.extmetadata.Artist.value)" -replace '<[^>]+>', '').Trim(); w = $ii.width; h = $ii.height; page = $ii.descriptionurl }
            break
        }
        if ($null -ne $picked) { break }
    }
    if ($null -eq $picked) { "{0,-10} nothing with a usable licence found" -f $tag; continue }
    Invoke-WebRequest -Headers $ua -Uri $picked.url -OutFile $file
    $kb = [int]((Get-Item $file).Length / 1KB)
    "{0,-10} {1,5} KB  {2} ({3}x{4}, {5}) {6}" -f $tag, $kb, $picked.title, $picked.w, $picked.h, $picked.lic, $picked.page
    [void]$sources.Add(("{0}`t{1}`t{2}`t{3}`t{4}" -f $tag, $picked.title, $picked.lic, $picked.author, $picked.page))
    Start-Sleep -Milliseconds 300
}
if ($sources.Count -gt 0) { $sources | Out-File (Join-Path $out "st-sources.txt") -Append -Encoding utf8 }
