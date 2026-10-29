# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;C:\Users\5090\scoop\shims;$env:PATH"
$env:CARGO_HOME = "C:\Users\5090\scoop\persist\rustup\.cargo"
$env:RUSTUP_HOME = "C:\Users\5090\scoop\persist\rustup\.rustup"
Set-Location "C:\Users\5090\Projects\comfy-photocraft"
cargo build --release -p photocraft-ui-egui --example snapshot 2>&1 | Select-String -Pattern "^error|Finished" | Out-String -Width 200
$env:WGPU_BACKEND = "vulkan"
$exe = "C:\Users\5090\Projects\comfy-photocraft\target\release\examples\snapshot.exe"
"--- 1. the bar under a fresh selection, prompt typed ---"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genbar-idle.png --size 1440x900 --scale 1 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":320,\"y\":560,\"width\":384,\"height\":320}}],[\"ui.set\",{\"generativePrompt\":\"a small red wooden rowing boat floating on the water\",\"generativeVariations\":2}]]"
"exit: $LASTEXITCODE"
"--- 2. after a live 2-variation fill (background job, long settle) ---"
$t = Get-Date
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genbar-results.png --size 1440x900 --scale 1 --background-jobs --settle-ms 90000 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"prefs.set\",\"params\":{\"values\":{\"integrations.allowResearchModels\":true}}}],[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":320,\"y\":560,\"width\":384,\"height\":320}}],[\"ui.set\",{\"generativePrompt\":\"a small red wooden rowing boat floating on the water\",\"generativeTemplate\":\"qwen-2.1/fill\",\"generativeVariations\":2}],[\"engine.execute\",{\"command\":\"generate.fill\",\"params\":{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"template\":\"qwen-2.1/fill\",\"variations\":2,\"seed\":7},\"wait\":false}]]"
"exit: $LASTEXITCODE  ($([int]((Get-Date) - $t).TotalSeconds) s)"
"--- 3. mid-job (warm server, short settle) ---"
& $exe --% --out C:\Users\5090\ComfyUI\photocraft-tests\ui-genbar-running.png --size 1440x900 --scale 1 --background-jobs --settle-ms 2500 --open C:\Users\5090\ComfyUI\photocraft-tests\q21-image.png --script "[[\"engine.execute\",{\"command\":\"prefs.set\",\"params\":{\"values\":{\"integrations.allowResearchModels\":true}}}],[\"engine.execute\",{\"command\":\"select.rect\",\"params\":{\"x\":320,\"y\":560,\"width\":384,\"height\":320}}],[\"ui.set\",{\"generativePrompt\":\"a small red wooden rowing boat floating on the water\",\"generativeTemplate\":\"qwen-2.1/fill\",\"generativeVariations\":2}],[\"engine.execute\",{\"command\":\"generate.fill\",\"params\":{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"template\":\"qwen-2.1/fill\",\"variations\":2,\"seed\":7},\"wait\":false}]]"
"exit: $LASTEXITCODE"
Start-Sleep -Seconds 1
try { Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8188/interrupt -TimeoutSec 5 | Out-Null; "interrupted the leftover prompt" } catch { "interrupt: $($_.Exception.Message)" }
