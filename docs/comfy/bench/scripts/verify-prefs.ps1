# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\shims;$env:PATH"
$env:CARGO_HOME = "C:\Users\5090\scoop\persist\rustup\.cargo"
$env:RUSTUP_HOME = "C:\Users\5090\scoop\persist\rustup\.rustup"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
"=== TSV line endings (mixed after appends) -> one convention per file ==="
$utf8 = New-Object System.Text.UTF8Encoding $false
foreach ($f in Get-ChildItem crates\ui-egui\src\i18n\*.tsv) {
    $raw = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8)
    $crlf = ([regex]::Matches($raw, "`r`n")).Count
    $lf = ([regex]::Matches($raw, "(?<!`r)`n")).Count
    if ($crlf -gt 0 -and $lf -gt 0) {
        # Majority wins; git stores LF in the index either way.
        $norm = $raw -replace "`r`n", "`n"
        if ($crlf -ge $lf) { $norm = $norm -replace "`n", "`r`n" }
        [IO.File]::WriteAllText($f.FullName, $norm, $utf8)
        "{0,-12} CRLF {1,5}  LF {2,5}  -> normalised to {3}" -f $f.Name, $crlf, $lf, $(if ($crlf -ge $lf) { "CRLF" } else { "LF" })
    } else {
        "{0,-12} CRLF {1,5}  LF {2,5}  (consistent)" -f $f.Name, $crlf, $lf
    }
}
cargo fmt --all
"=== ui-egui lib tests (all) ==="
cargo test -p photocraft-ui-egui --lib 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|test result|FAILED|panicked" | Out-String -Width 220
"=== clippy ui-egui ==="
cargo clippy -p photocraft-ui-egui --all-targets -- -D warnings 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|Finished" | Out-String -Width 220
"=== i18n coverage ==="
cargo xtask i18n-coverage 2>&1 | Select-Object -Last 15 | Out-String
"=== release snapshot build + Preferences > AI Integrations ==="
cargo build --release -p photocraft-ui-egui --example snapshot 2>&1 | Select-String -Pattern "^error|Finished" | Out-String -Width 220
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-prefs-integrations.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"edit.preferences.integrations\"}]]"
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-prefs-integrations-de.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"prefs.set\",\"params\":{\"values\":{\"interface.language\":\"de\"}}}],[\"ui.menu.invoke\",{\"id\":\"edit.preferences.integrations\"}]]"
"exit: $LASTEXITCODE"
