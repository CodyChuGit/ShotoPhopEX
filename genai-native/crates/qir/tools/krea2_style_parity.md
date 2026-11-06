# Krea 2 style reference CPU parity

Worktree changes are uncommitted. The style route is `Krea2Pipeline::generate_style`,
served as `krea2-turbo/image-style` (task image, family Krea2). It takes one or two
references and explicit dimensions. The default adapter is `loras/krea2_style_reference.safetensors`,
at strength one; the worker only advertises the template when that file exists.

The worker accepts the template model slot as `params.models.lora` (or `params.lora`).
PhotoshopEX currently drops `Request.models` in `crates/genai/src/native.rs` when constructing
`RunRequest`. That file is outside this assignment: forward those name/file pairs into
`params.models` as an object before sending the native request. The default adapter works
without this upstream change.

Style-specific differences: stock EditPlus area resizes at 384-square and 1024-square pixel
areas; Python rounding including ties; independent VAE references; EditPlus's explicit chat
override and Picture N labels; text/target/reference token order; uncentered positive frame
indices; circular patch padding of odd reference grids; zero-timestep reference modulation;
the template's RGB qwen_image_vae checkpoint for encoding and decoding; requested-size ModelSamplingFlux shift; ConditioningZeroOut CFG; Euler/simple, eight steps
by default. Legacy image/edit generation retains its previous route, grid and schedule.
The style vision entry point uses torch.linspace's symmetric position grid without changing
legacy vision forwarding.

## Reproduce on CPU

From this worktree in PowerShell. Run one command at a time; do not run installed tests in parallel.

```powershell
$env:RAYON_NUM_THREADS='8'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$env:CARGO_TARGET_DIR="$PWD/target-krea2-style-cpu"
$py='C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
& $py -I crates/qir/tools/krea2_style_fixtures.py
& $py -I crates/qir/tools/krea2_style_encoder_fixtures.py
& $py -I crates/qir/tools/krea2_style_model_fixtures.py
cargo test --release -p qwen-image-rs -p gn-diffusion --lib --test krea2_style --no-default-features -j 6 -- --nocapture --test-threads=1
cargo test --release -p qwen-image-rs --test krea2_style -j 6 stock_editplus_preprocessing -- --ignored --nocapture
cargo test --release -p qwen-image-rs --test krea2_style -j 6 installed_style_encoder -- --ignored --nocapture
cargo test --release -p qwen-image-rs --test krea2_style -j 6 installed_style_reference_latents -- --ignored --nocapture
cargo test --release -p qwen-image-rs --test krea2_style -j 6 installed_style_dit_and_two_steps -- --ignored --nocapture
```

The large generated tensors live in `crates/qir/tests/generated/` and are ignored. Small
synthetic model fixtures cover the actual ComfyUI module architecture, including unequal
references, odd reference padding, adapters, clearing adapters, and legacy image/edit wiring.
They are regression coverage, not substitutes for the installed-model gates.
CPU installed oracles dequantize FP8 matrices through bf16 and run float32 math; the DiT
oracle uses ComfyUI's LoRAAdapter merged float32 weights, while Rust uses its runtime side path.
The portable Python scripts set torch threads to eight and never allocate CUDA tensors.

## Measurements

Relative RMS is sqrt(sum((actual-oracle)^2)/sum(oracle^2)); comparisons use complete tensors.

| Stage | Max absolute | Relative RMS | Status |
|---|---:|---:|---|
| Vision patches, square | 2.38419e-7 | 1.48624e-7 | pass |
| Vision patches, landscape | 3.57628e-7 | 2.98956e-7 | pass |
| VAE pixels, square | 1.19209e-7 | 1.12064e-7 | pass |
| VAE pixels, landscape | 1.19209e-7 | 1.35827e-7 | pass |
| Installed two-step latent, chained native inputs | 5.84960e-4 | 2.58653e-5 | pass (2e-3 RMS) |
| Installed Euler latent after step one, chained native inputs | 7.89165e-5 | 1.18142e-5 | pass (2e-3 RMS) |
| Installed DiT, real LoRA, both references | 4.86612e-4 | 6.25458e-5 | pass (1e-3 RMS) |
| Installed reference latent, square | 3.09944e-6 | 4.84602e-7 | pass |
| Installed reference latent, landscape | 2.02656e-6 | 2.47436e-7 | pass |
| Installed text, identical vision features | 1.29395e-2 | 6.39578e-6 | pass (0.02 absolute, 5e-5 RMS) |
| Conditioning with identical tower-input patches | 2.58331e-2 | 4.58287e-5 | diagnostic: absolute cap still misses |
| Installed conditioning, native vision | 3.06015e-2 | 4.34451e-5 | absolute gate fails (0.02); RMS passes (5e-5) |

Installed inputs: 256x256 and 320x240 deterministic RGBA references (alpha dropped),
256x256 target, prompt `a ceramic bird`, seed 123, two steps, CFG one, real Ostris adapter
at strength one. The fixed references are deliberately nonuniform high-frequency pixel patterns.
The full installed gate is **not accepted while any listed gate fails**.
The failed combined-conditioning absolute cap is retained rather than widened. Its text-only
isolation passes, indicating accumulated vision floating-point differences rather than a
chat-template, token-order or tap mismatch. Feeding identical input patches to the native
tower still gives max conditioning error 0.0258331, so fixing byte-to-float or resize rounding
alone does not resolve the absolute gate. Further alignment of the existing float32 vision
tower is required. No threshold was relaxed.

Regression verification: 91 qir unit tests and eight worker unit tests pass. The new small
ComfyUI-module legacy image comparison has max abs 7.15256e-7 / relative RMS 1.51734e-7;
legacy edit with boost four has max abs 7.15256e-7 / relative RMS 1.54817e-7. The checkout
had no pre-existing installed Krea image/edit parity integration tests; these are added
architectural regression checks, not a claim that the previous full-size GPU app suite was rerun.
ComfyUI oracle peak working sets: encoder 21.34 GiB, VAE plus streamed DiT 19.77 GiB.
The Rust streamed DiT process reached 11.38 GB peak working set (Windows process counters),
and its three full passes completed in 2332.21 seconds. GPU inference was not used.
The installed reference-latent and DiT/two-step tests pass; the installed encoder test remains
red because of the combined-conditioning absolute cap. The default (non-ignored) suites pass.

Remaining work: align the complete vision-to-conditioning result to the 0.02 absolute cap;
forward PhotoshopEX `Request.models` as described above for custom model-slot overrides;
then run the deferred GPU comparison through the rebuilt app. The default adapter route is
implemented, but the full CPU acceptance gate is not marked complete.

## Files changed

- `crates/qir/src/krea2_pipeline.rs`: style request, preprocessing, VAE selection, CFG/schedule.
- `crates/qir/src/model/krea2.rs`: EditPlus template, multi-reference split modulation, streaming parity runner.
- `crates/qir/src/model/vision.rs`: opt-in torch.linspace grid; legacy forwarding remains on its previous grid.
- `crates/gn-diffusion/src/lib.rs`: registration, availability, references and model-slot adapter routing.
- `crates/qir/tests/krea2_style.rs`: small and ignored installed-model complete-tensor checks.
- `crates/qir/tests/fixtures/krea2_style{,_weights,_lora}.safetensors`, `krea2_style_template.json`, and local `.gitignore` exceptions: small ComfyUI fixtures.
- `crates/qir/tests/generated/.gitignore`: regenerated large fixtures are local only.
- `crates/qir/tools/krea2_style{,_encoder,_model}_fixtures.py`: CPU ComfyUI fixture generators.
- `crates/qir/tools/krea2_style_app_compare.py` and this report: deferred app comparison and reproduction.

## GPU app comparison for Claude

After rebuilding PhotoshopEX against this worktree and making the GPU available, run both
routes through the app. Start ComfyUI separately. This script has not been executed here.

```powershell
& C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe -I crates/qir/tools/krea2_style_app_compare.py --app C:/Users/5090/Documents/PhotoshopEX/target-app/release/photocraft.exe --models C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI/models --runtime 'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib' --image1 crates/qir/tests/generated/krea2_style_ref1.png --image2 crates/qir/tests/generated/krea2_style_ref2.png --out runs/krea2-style-app
```

This uses 1024x1024, eight steps, seed 123, CFG one, enhancement disabled, and the default
style adapter. It explicitly binds the FP8 base model, preventing the ComfyUI route from
automatically choosing the installed NVFP4 alternative or a saved model preference. It saves both app outputs, stdout/stderr, PSNR and relative RMS. Adjust the
runtime directory to the same CUDA/cuDNN runtime used by the rebuilt native app if different.
The script changes the app's integration preferences for each run.

The sandbox denied staging because the worktree index is outside its writable root. After
review, stage the three small binary fixtures explicitly (each is below 2 MB):

```powershell
git add -f crates/qir/tests/fixtures/krea2_style.safetensors crates/qir/tests/fixtures/krea2_style_weights.safetensors crates/qir/tests/fixtures/krea2_style_lora.safetensors
```
