# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Appends rows from pref-translations.tsv (lang<TAB>source<TAB>translation, UTF-8) to each
# language's TSV, keeping that file's line ending and writing UTF-8 without a BOM.
$utf8 = New-Object System.Text.UTF8Encoding $false
$data = [IO.File]::ReadAllText("$PSScriptRoot\pref-translations.tsv", [Text.Encoding]::UTF8)
$byLang = @{}
foreach ($line in ($data -split "`n")) {
    $line = $line.TrimEnd("`r")
    if ($line -eq "") { continue }
    $cols = $line -split "`t"
    if ($cols.Count -ne 3) { throw "bad row: $line" }
    if (-not $byLang.ContainsKey($cols[0])) { $byLang[$cols[0]] = New-Object System.Collections.ArrayList }
    [void]$byLang[$cols[0]].Add("`t" + $cols[1] + "`t" + $cols[2])
}
foreach ($lang in $byLang.Keys) {
    $path = Join-Path $PSScriptRoot "..\..\..\..\crates\ui-egui\src\i18n\$lang.tsv"
    if (-not (Test-Path $path)) { throw "missing $path" }
    $raw = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
    $nl = if ($raw.Contains("`r`n")) { "`r`n" } else { "`n" }
    $prefix = if ($raw.EndsWith("`n")) { "" } else { $nl }
    $rows = $byLang[$lang] | ForEach-Object { $_ }
    $skipped = @($rows | Where-Object { $raw.Contains($_) })
    $rows = @($rows | Where-Object { -not $raw.Contains($_) })
    if ($rows.Count -gt 0) {
        [IO.File]::AppendAllText($path, $prefix + ($rows -join $nl) + $nl, $utf8)
    }
    "{0,-8} appended {1} rows, skipped {2} (already present), line ending {3}" -f $lang, $rows.Count, $skipped.Count, $(if ($nl -eq "`r`n") { "CRLF" } else { "LF" })
}
