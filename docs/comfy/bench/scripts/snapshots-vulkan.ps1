# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genfill-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":200,\"y\":200,\"width\":400,\"height\":300}}],[\"ui.menu.invoke\",{\"id\":\"generate.fill\"}]]"
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genimage-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"generate.image\"}]]"
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-selecttext-dialog.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"ui.menu.invoke\",{\"id\":\"select.byText\"}]]"
"exit: $LASTEXITCODE"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-edit-menu.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --click-at 52,12
"exit: $LASTEXITCODE"
Get-ChildItem C:\Users\5090\ComfyUI\photocraft-tests\ui-*.png | Select-Object Name, Length, LastWriteTime
