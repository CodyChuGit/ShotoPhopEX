# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-selection-context.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":200,\"y\":200,\"width\":400,\"height\":300}}],[\"ui.pointer\",{\"tool\":\"RectMarquee\",\"button\":\"secondary\",\"events\":[{\"kind\":\"down\",\"x\":400,\"y\":350},{\"kind\":\"up\",\"x\":400,\"y\":350}]}]]"
"exit: $LASTEXITCODE"
