# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-edit-menu.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --click-at 90,16
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-select-menu.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --click-at 261,16
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-selection-context.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":200,\"y\":200,\"width\":400,\"height\":300}}],[\"ui.set\",{\"tool\":\"marquee_rect\"}]]" --right-click-at 500,400
"exit: $LASTEXITCODE"
