# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\shims;$env:PATH"
$env:CARGO_HOME = "C:\Users\5090\scoop\persist\rustup\.cargo"
$env:RUSTUP_HOME = "C:\Users\5090\scoop\persist\rustup\.rustup"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
cargo fmt --all
"=== ui-egui lib tests (all) ==="
cargo test -p photocraft-ui-egui --lib 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|test result|FAILED|panicked|failures:" | Out-String -Width 220
"=== clippy ui-egui ==="
cargo clippy -p photocraft-ui-egui --all-targets -- -D warnings 2>&1 | Select-String -Pattern "^(error|warning)|^\s+-->|Finished" | Out-String -Width 220
"=== release snapshot build ==="
cargo build --release -p photocraft-ui-egui --example snapshot 2>&1 | Select-String -Pattern "^error|Finished" | Out-String -Width 220
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genfill-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":200,\"y\":200,\"width\":400,\"height\":300}}],[\"ui.menu.invoke\",{\"id\":\"generate.fill\"}]]"
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-selecttext-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"select.byText\"}]]"
"exit: $LASTEXITCODE"
