# Any ComfyUI started here is external, for comparison benchmarks only; PhotoshopEX itself no longer uses ComfyUI.
# Removed app preferences/routes require the last ComfyUI-capable commit cdcedbd8 for that benchmark side, or direct ComfyUI HTTP API calls.
# A hard test picture for the generative removal: a portrait with a necklace, made by the app's
# own text-to-image (generate.image, Quality balanced), then the necklace located by Select by
# Text so the strokes can be planned. Output: bench\necklace.png and the necklace's bounds.
Set-Location "C:\Users\5090\Documents\PhotoshopEX"
$cli = (Resolve-Path "target\release\photocraft-cli.exe").Path
$out = "C:\Users\5090\ComfyUI\photocraft-tests\bench"
$env:CM_PREFS = '{\"values\":{\"integrations.engine\":\"comfy\",\"integrations.enhancePrompts\":false,\"integrations.quality\":\"balanced\"}}'
$env:CM_NEW = '{\"width\":1024,\"height\":1024}'
$env:CM_GEN = '{\"prompt\":\"close-up studio portrait photograph of a woman with her hair up, wearing a delicate gold chain necklace with a small round pendant over bare collarbones, plain dark grey top, soft natural light, shallow depth of field, photorealistic\",\"target\":\"document\",\"width\":1024,\"height\":1024,\"seed\":21}'
$env:CM_OUT = Join-Path $out "necklace.png"
& $cli --% run --new "%CM_NEW%" --cmd prefs.set --params "%CM_PREFS%" --cmd generate.image --params "%CM_GEN%" --out %CM_OUT% | Where-Object { $_ -match '"runId"|error' } | ForEach-Object { if ($_ -match '"ms":(\d+)') { "image: $($Matches[1]) ms" } else { $_ } }
$env:CM_IMAGE = $env:CM_OUT
$env:CM_TXT = '{\"prompt\":\"necklace\"}'
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_TXT%" | Where-Object { $_ -match '"instances"|error' }
$env:CM_TXT = '{\"prompt\":\"pendant\"}'
& $cli --% run %CM_IMAGE% --cmd prefs.set --params "%CM_PREFS%" --cmd select.byText --params "%CM_TXT%" | Where-Object { $_ -match '"instances"|error' }
