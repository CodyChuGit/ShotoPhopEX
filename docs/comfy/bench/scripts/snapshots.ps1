# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\shims;$env:PATH"
$env:CARGO_HOME = "C:\Users\5090\scoop\persist\rustup\.cargo"
$env:RUSTUP_HOME = "C:\Users\5090\scoop\persist\rustup\.rustup"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$out = "C:\Users\5090\ComfyUI\photocraft-tests"
# 1. Generative Fill dialog over a selection (Edit > Generative Fill...)
cargo run --release -p photocraft-ui-egui --example snapshot --% -- --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genfill-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":200,\"y\":200,\"width\":400,\"height\":300}}],[\"ui.menu.invoke\",{\"id\":\"generate.fill\"}]]"
# 2. Generate Image dialog
cargo run --release -p photocraft-ui-egui --example snapshot --% -- --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genimage-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"generate.image\"}]]"
# 3. Select by Text dialog
cargo run --release -p photocraft-ui-egui --example snapshot --% -- --out C:\Users\5090\ComfyUI\photocraft-tests\ui-selecttext-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"select.byText\"}]]"
Get-ChildItem $out\ui-*.png | Select-Object Name, Length, LastWriteTime
