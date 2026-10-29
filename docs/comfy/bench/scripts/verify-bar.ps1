# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\shims;$env:PATH"
$env:CARGO_HOME = "C:\Users\5090\scoop\persist\rustup\.cargo"
$env:RUSTUP_HOME = "C:\Users\5090\scoop\persist\rustup\.rustup"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
cargo fmt --all
"=== ui-egui lib tests (all) ==="
cargo test -p photocraft-ui-egui --lib 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|^\s+\|.*\^|test result|FAILED|panicked" | Out-String -Width 220
"=== engine tests (all, incl. prefs audit) ==="
cargo test -p photocraft-engine 2>&1 | Select-String -Pattern "^error|^\s+-->|test result: FAILED|FAILED|panicked|unread|hidden" | Out-String -Width 220
cargo test -p photocraft-engine 2>&1 | Select-String -Pattern "test result" | Measure-Object | ForEach-Object { "engine test binaries: $($_.Count)" }
"=== clippy ==="
cargo clippy -p photocraft-ui-egui -p photocraft-engine --all-targets -- -D warnings 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|Finished" | Out-String -Width 220
"=== xtask gates ==="
cargo xtask scorecard 2>&1 | Select-Object -Last 1 | Out-String
cargo xtask parity 2>&1 | Select-Object -Last 1 | Out-String
cargo xtask layers 2>&1 | Select-Object -Last 1 | Out-String
cargo xtask i18n-coverage 2>&1 | Select-String -Pattern "^(cs|de|ja|ko)\s" | Out-String
"=== wasm gate ==="
cargo xtask wasm 2>&1 | Select-String -Pattern "ui-egui|engine|genai|error|FAIL" | Out-String -Width 200
