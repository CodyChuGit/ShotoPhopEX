# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# .NET's current directory is not PowerShell's: use absolute paths for [IO.File].
$dir = "C:\Users\5090\Projects\comfy-photocraft\crates\genai\workflows"
Set-Location $dir
[IO.Directory]::SetCurrentDirectory($dir)
$utf8 = New-Object System.Text.UTF8Encoding $false
$noun = "Add {prompt} to this image, fitting the scene's perspective, lighting and surroundings naturally. Change nothing else."
$imp = "{prompt}. Fit the result to the scene's perspective, lighting and surroundings naturally, and change nothing else."
foreach ($f in @("qwen-edit-2511-fill.json", "qwen-edit-2511-fill-lightning-8.json", "qwen-edit-2511-fill-lightning-4.json")) {
    $t = [IO.File]::ReadAllText($f, [Text.Encoding]::UTF8)
    if ($t -notmatch '"promptFormat"') {
        $t = $t -replace '(\r?\n\s*)"notes":', ('$1"promptFormat": "' + $noun + '",$1"promptFormatImperative": "' + $imp + '",$1"notes":')
        [IO.File]::WriteAllText($f, $t, $utf8)
        "${f}: wrappers added"
    } else { "${f}: already has promptFormat" }
}
$g = "qwen-edit-2511-fill-guided.json"
$t = [IO.File]::ReadAllText($g, [Text.Encoding]::UTF8)
if ($t -notmatch '"promptFormatImperative"') {
    $gimp = "Image 2 is a mask of image 1: the white area marks the only part of image 1 to change. Inside that area: {prompt}. Fit the result to the scene naturally and keep everything outside that area exactly as it is."
    $t = $t -replace '(\r?\n\s*)"notes":', ('$1"promptFormatImperative": "' + $gimp + '",$1"notes":')
    [IO.File]::WriteAllText($g, $t, $utf8)
    "${g}: imperative wrapper added"
}
$q = "qwen-2.1-fill.json"
$t = [IO.File]::ReadAllText($q, [Text.Encoding]::UTF8)
if ($t -notmatch '"promptFormatImperative"') {
    $qimp = "Edit image 1. Image 2 is a mask of image 1: change only the area that is white in image 2, as follows: {prompt}. Keep everything outside that area exactly as it is in image 1, with the same lighting, perspective and style."
    $t = $t -replace '(\r?\n\s*)"placeholders":', ('$1"promptFormatImperative": "' + $qimp + '",$1"placeholders":')
    [IO.File]::WriteAllText($q, $t, $utf8)
    "${q}: imperative wrapper added"
}
Select-String -Path *.json -Pattern '"promptFormat' | ForEach-Object { "$($_.Filename):$($_.LineNumber): " + $_.Line.Trim().Substring(0, [Math]::Min(80, $_.Line.Trim().Length)) }
