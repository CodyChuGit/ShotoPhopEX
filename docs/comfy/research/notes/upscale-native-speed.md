# Generative Upscale: native speed investigation, 2026-10-28

**The speed target is still open.** The default native upscale now measures **47.5 s warm**
without profiling. ComfyUI measures **39.1 s** with its default settings and **39.0 s with the
same Restore LoRA explicitly enabled at 0.5**. The two small app optimizations save about
**0.7–0.8 s of CPU work**, with byte-identical final images on both routes. They do not solve
the native worker's LoRA cost. No settings, prompts, tile positions, seeds, steps, precision,
alignment decisions, or blend arithmetic were changed to obtain this result.

The PhotoshopEX worktree is `C:/Users/5090/.codex/worktrees/6530/PhotoshopEX`. Nothing was
committed. No genai-native source was changed. Creating its required separate worktree failed:

```text
git -C C:/Users/5090/Documents/genai-native worktree add C:/Users/5090/.codex/worktrees/6530/PhotoshopEX/target/genai-native-upscale HEAD
Preparing worktree (detached HEAD 03fb72e)
fatal: could not create directory of '.git/worktrees/genai-native-upscale': Permission denied
```

This session can write the PhotoCraft worktree but cannot write the second repository's Git
metadata. It cannot request elevated access. Native-worker implementation and its performance
harness are consequently left for Claude/a session with access to create that worktree.

## Setup and boundaries

RTX 5090, driver 617.42; one inference process at a time. The benchmark scripts stopped each
native worker/ComfyUI process before the next route started. All inference measurements below
finished before Cargo tests started. Source portrait, template `qwen-2.1/edit`, 864×1152 input,
scale 2, default tileMegapixels 1, Restore prompt, 25 Euler/simple steps, CFG 1. Cold seed 20;
warm seed **21**, as prescribed by `speed-table.ps1` to avoid ComfyUI result-cache hits.

The enlarged document is 1728×2304. Both routes send **six 992×992 tiles**, in this order:
`(0,0), (736,0), (0,864), (736,864), (0,1312), (736,1312)`. Resolution-0 conditioning keeps
this 32-aligned size: 3844 reference latent tokens and 961 merged vision tokens. The nominal
blend strip is 128 px; the final flush-to-edge tiles overlap more. No 2048-wide image crosses
the worker channel in this case. Each raw RGBA request/result is 3,936,256 bytes.

`build-native.cmd photocraft-cli` was used in this worktree. Its directory selection now follows
the script location instead of silently building the main PhotoCraft checkout. The user-level
Cargo patch still resolves gn-diffusion/qir to **the main genai-native checkout**, HEAD
`03fb72e5424639a79d34c30e7e2602e772a31ce2`, with pre-existing modifications to Cargo.lock,
`comfy_files.rs`, `comfy_int8.rs`, and `pipeline.rs`. Both instrumented binaries used those same
sources; these measurements must not be attributed to clean HEAD alone. Source hashes/status
are retained in `target/upscale-native-source-state.json`.

The frozen pre-optimization binary is `target/upscale-baseline.exe`. Local benchmark inputs,
logs, images and phase JSON remain ignored under `target/upscale-*`.

## End-to-end observations

Seconds from the speed-table script, including CLI startup and, where requested, PNG export.
The phase totals below use the command's own timer, which excludes CLI startup/export.

| Run | Cold | Warm | Interpretation |
|---|---:|---:|---|
| Supplied historical native baseline | — | 54.1 | Original task's app measurement |
| Existing main CLI, initial unprofiled observation | 85.2 | 48.4 | CPU compilation overlapped; contextual observation only |
| Instrumented native before | 79.9 | 55.0 | `QIR_PHASE_PROFILE=1`, saved images |
| Instrumented native after | 88.9 | 49.5 | Same profiling, saved images |
| Instrumented ComfyUI before | 59.5 | 42.4 | Node/CUDA profiler, saved images, default no LoRA |
| Instrumented ComfyUI after | 57.5 | 39.1 | Same profiler, saved images, default no LoRA |
| Native after, unprofiled | 72.4 | **47.5** | Default Restore 0.5 retained |
| Native after, diagnostic `loras: []` | 55.7 | **30.8** | Different output/settings; **not a fix or parity claim** |
| ComfyUI after, explicit Restore 0.5 | 57.6 | **39.0** | Matched adapter setting; 192 attached patches logged |

Do not attribute the full profiled before/after reduction to the app changes: unchanged worker
sampling also became faster between runs. Profiling synchronizes native phases, including
each step; it is not an unprofiled speed claim. Cold results include kernel/model initialization
and are not suitable for judging this warm optimization. Native remains about 22% slower than
the matched Restore ComfyUI observation, and misses the historical 40.8 s target by 6.7 s.

## App phases, before → after

Warm command totals, seconds, six tiles combined. Native worker uses raw frames; ComfyUI uses
PNG + HTTP/WebSocket. `request` contains backend execution and transfers; it is not additive
with the more detailed backend rows.

| Phase | Native before | Native after | ComfyUI before | ComfyUI after |
|---|---:|---:|---:|---:|
| Bicubic enlargement | .052 | .050 | .057 | .051 |
| Render enlarged composite | .019 | .018 | .020 | .018 |
| Crop / prepare tile pixels | .006 | .006 | .006 | .006 |
| All backend requests | 53.613 | 48.810 | 41.019 | 38.412 |
| Request encode/copy, inside requests | <.006 | .005 | .035 | .038 |
| Upload, inside requests | unavailable separately | unavailable separately | .021 | .024 |
| Worker/server execution, inside requests | 53.578 | 48.767 | 40.245 | 37.347 |
| Download + result materialization, inside requests | <.006 materialization only | <.006 materialization only | .605 | .881 |
| Alignment / shift | .723 | .153 | .763 | .137 |
| Color matching | .189 | .053 | .195 | .055 |
| Strip blending | .025 | .030 | .026 | .030 |
| Final normalize / layer write / insertion | .153 | .173 | .155 | .171 |
| Entire command | **54.789** | **49.305** | **42.249** | **38.889** |

Native round-trip residual is approximately **5–8 ms/tile** above `DoneInfo.elapsed_ms`; it
includes IPC in both directions and app dispatch/materialization. The current EngineClient
does not expose separate send/receive timers. Native `uploadMs: 0` means **not separately
instrumented**, not zero physical transfer time. Its `downloadMs` measures only the final RGBA
copy after EngineClient returns. ComfyUI's download includes HTTP fetch and PNG decode. This
report does not invent directional wire timings from those aggregate boundaries.

Warm per-tile app measurements, milliseconds (`request` / `alignment` / `color`):

| Tile | Native before | Native after | ComfyUI before | ComfyUI after |
|---|---|---|---|---|
| 0 | 8909.8 / 120.9 / 32.7 | 8153.7 / 35.0 / 8.8 | 6973.8 / 122.8 / 34.6 | 6520.3 / 23.0 / 8.9 |
| 1 | 8955.1 / 118.7 / 31.5 | 8115.3 / 23.5 / 8.9 | 6689.6 / 134.6 / 32.0 | 6251.1 / 22.6 / 9.1 |
| 2 | 8994.3 / 119.3 / 31.2 | 8163.7 / 22.6 / 8.8 | 6996.3 / 121.3 / 31.3 | 6533.6 / 23.0 / 9.1 |
| 3 | 8922.7 / 121.9 / 31.4 | 8123.8 / 22.4 / 8.9 | 6751.3 / 131.2 / 31.7 | 6267.9 / 23.4 / 9.5 |
| 4 | 8915.8 / 120.9 / 31.2 | 8128.9 / 23.4 / 8.8 | 6934.0 / 122.1 / 33.4 | 6549.4 / 22.8 / 9.1 |
| 5 | 8915.2 / 121.2 / 31.3 | 8124.6 / 26.4 / 9.0 | 6674.1 / 130.8 / 32.4 | 6289.8 / 22.2 / 9.4 |

Crop is .8–1.3 ms/tile; blend is 2.7–7.2 ms/tile in these warm runs. Full values, including
positions and backend timings, are in each run's `upscale-2x-*-1.log`.

## Worker phases and the Restore difference

Seconds for the six warm requests. Native phase scopes synchronize CUDA. ComfyUI node times
synchronize at node boundaries; prefix-block CUDA events are a **subset** of KSampler time,
not a separately additive node. Prefix totals have different boundaries: native includes the
complete prefix routine, ComfyUI's event subtotal covers prefix transformer blocks.

| Worker phase | Native before | Native after | ComfyUI before | ComfyUI after |
|---|---:|---:|---:|---:|
| Reference pixels | .006 | .008 | inside conditioning | inside conditioning |
| Reference VAE encode | 1.061 | .973 | inside conditioning | inside conditioning |
| Vision tower | 1.689 | 1.548 | inside conditioning | inside conditioning |
| Text encode | .848 | .763 | inside conditioning | inside conditioning |
| Conditioning node (ComfyUI positive + negative) | — | — | 8.141 | 7.629 |
| DiT prefix | 4.405 | 3.959 | .940 block CUDA subtotal | .870 block CUDA subtotal |
| 150 denoising steps | 40.822 | 37.042 | 28.398 target CUDA subtotal | 26.247 target CUDA subtotal |
| KSampler entire node (prefix + steps + overhead) | — | — | 29.831 | 27.642 |
| VAE decode | 4.487 | 4.140 | 1.155 | 1.128 |
| SaveImage PNG node | raw RGBA readback instead | raw RGBA readback instead | .489 | .540 |

Per-tile worker scopes, seconds, in the same tile order. Cells are
**conditioning / prefix / steps / decode**; the boundary caveats above apply.

| Tile | Native before | Native after | ComfyUI before | ComfyUI after |
|---|---|---|---|---|
| 0 | 0.616 / 0.729 / 6.797 / 0.736 | 0.573 / 0.659 / 6.190 / 0.690 | 1.363 / 0.158 / 4.737 / 0.192 | 1.263 / 0.144 / 4.363 / 0.186 |
| 1 | 0.614 / 0.731 / 6.822 / 0.756 | 0.564 / 0.659 / 6.158 / 0.690 | 1.373 / 0.157 / 4.708 / 0.194 | 1.270 / 0.144 / 4.366 / 0.190 |
| 2 | 0.627 / 0.741 / 6.845 / 0.744 | 0.570 / 0.659 / 6.203 / 0.690 | 1.362 / 0.159 / 4.734 / 0.197 | 1.275 / 0.145 / 4.370 / 0.188 |
| 3 | 0.620 / 0.739 / 6.788 / 0.744 | 0.567 / 0.660 / 6.160 / 0.690 | 1.362 / 0.160 / 4.763 / 0.198 | 1.272 / 0.145 / 4.376 / 0.187 |
| 4 | 0.616 / 0.733 / 6.787 / 0.750 | 0.567 / 0.661 / 6.168 / 0.690 | 1.346 / 0.146 / 4.717 / 0.194 | 1.269 / 0.145 / 4.384 / 0.191 |
| 5 | 0.615 / 0.732 / 6.782 / 0.757 | 0.567 / 0.661 / 6.162 / 0.691 | 1.335 / 0.158 / 4.738 / 0.181 | 1.279 / 0.146 / 4.388 / 0.186 |

For native, `references` also has small orchestration/transfer costs; reference pixel/VAE/vision
subscopes are nested within it. Full phase logs are `worker.log.err`; parsed generations are
`worker-phases.json`. ComfyUI's `comfy-phases.jsonl` retains every node and sampled CUDA scope
for all twelve tile requests. Both routes execute conditioning for **all six warm tiles**;
ComfyUI does not retain six complete conditioning entries in this actual sequence either.

The most useful controlled diagnostic changes only `loras` in the native app request:

| Unprofiled warm native phase | Restore 0.5 | No Restore | Difference |
|---|---:|---:|---:|
| Conditioning + prefix (`encode_s`) | 7.005 | 6.327 | .678 |
| 150 denoising steps (`denoise_s`) | 35.621 | 21.090 | **14.531** |
| Decode (`decode_s`) | 4.163 | 2.617 | **1.546** |
| CLI wall | 47.5 | 30.8 | **16.7** |

The installed Restore file contains **192 rank-32 adapter pairs, BF16**, 159,436,584 bytes.
Native loads the file once into the worker's six-file cache, but `set_loras` recreates slots
between requests, and every adapted forward computes `x Aᵀ`, then `... Bᵀ`, scale, and add.
It is **not merged once into the base weight**. The worker's warm time outside its existing
encode/denoise/decode counters is only about 25–30 ms/tile, including adapter setup/readback;
repeated file loading/setup cannot explain the observed 14.5 s denoising difference.

Restore also changes the memory outcome. Warm Restore logs select tiled decode with roughly
4.5–4.8 GiB-equivalent MiB counts available; the no-Restore run has 4885–5082 MiB and selects
whole decode. The logged whole-image requirement is 3844 MiB **plus the 1024 MiB margin**.
Do not lower that safety margin without measuring real allocation peaks and output parity.

## Source evidence: what ComfyUI does differently

Paths prefixed `genai-native/` refer to the main second checkout. `ComfyUI/` refers to
`C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI`.

| Behavior | Evidence |
|---|---|
| Default adapter mismatch | `crates/engine/src/generate_upscale.rs:130`: automatic Restore requires `runs_native`. ComfyUI's default gets no automatic Restore. `crates/genai/src/comfy.rs:379` can add it explicitly; `:400` inserts LoraLoaderModelOnly |
| File cache versus runtime application | `genai-native/crates/gn-diffusion/src/lib.rs:273` caches loaded LoraFile by path; `:743` builds the request stack. `genai-native/crates/qir/src/pipeline.rs:716` calls set_loras each request; `model/dit.rs:1139` binds the slots; `lora.rs:51` computes the two low-rank GEMMs; `:118` adds the delta |
| ComfyUI applies patches to weights and reuses resident materializations | `ComfyUI/nodes.py:772` loads the model-only adapter; `comfy/model_patcher.py:1936` installs LowVramPatch on dynamic modules. `comfy/ops.py:169` checks the resident signature and skips materialization on a hit; `:240` stages patches; `:312` applies them during materialization. `comfy/model_patcher.py:920` is the ordinary weight-patching path. This is not native's per-activation A/B side path |
| Matched adapter really ran | `target/upscale-comfy-restore/comfy-phases.jsonl` executes `pcx_lora_1_0`; stderr says `192 patches attached`. Warm conditioning/KSampler/decode are 7.634 / 27.644 / 1.132 s, essentially level with the no-LoRA route |
| Conditioning misses | `genai-native/crates/qir/src/pipeline.rs:580` keeps one exact picture entry. Changed tile pixels miss. ComfyUI events execute node 6 on every warm tile; `crates/genai/workflows/qwen-2.1-edit.json:48` supplies resolution 0. No request-size or step discrepancy was found |
| Prefix attention and cache | `genai-native/crates/qir/src/model/dit.rs:769` uses 1024-query f32 attention chunks; `:1282` builds the prefix. `ComfyUI/comfy/ldm/qwen_image21/model.py:277` resets per-generation cache, `:283` selects storage, `:389` runs the prefix blocks, `:397` reuses their K/V. Its `:197` block-causal attention dispatches through the installed PyTorch backend |
| Decode policy | `genai-native/crates/qir/src/model/vae.rs:410`, `:426`, `:957` implement the 1 GiB margin/auto selection; pipeline drops prefix caches before decode at `pipeline.rs:818`. ComfyUI's VAEDecode node is `crates/genai/workflows/qwen-2.1-edit.json:50` |
| IPC is raw, not PNG | `crates/genai/src/native.rs:315` builds binary blobs. `genai-native/crates/gn-engine/src/client.rs:268` writes blobs then JSON and reads progress/results; send/receive timing is not exposed. ComfyUI PNG encode/upload/fetch/decode live at `crates/genai/src/comfy.rs:695` and `:752` |
| Order/overlap | The same app tile loop invokes both routes sequentially. ComfyUI overlaps internal model prefetch: `comfy/ldm/qwen_image21/model.py:385`, `:388`, `:397`. No cross-tile overlap was found that could account for the native-only Restore penalty |

## Changes and parity

- `crates/engine/src/generate_realign.rs`: evaluate independent candidate rows with Rayon;
  retain each candidate's original serial floating-point sum and collect in original order,
  preserving strict-less-than tie selection. Keep the old sequential search as a test oracle.
- `crates/engine/src/generate_upscale.rs`: a 256-entry correction table per byte channel;
  retain the same f64 mean/variance/gain and rounding. Add per-tile/app phase timings to the
  command result, using the existing wasm-safe jobs clock. Add timing/lookup regression tests.
- `crates/genai/src/native.rs`: measure request preparation/raw-copy and final result-copy
  time; keep worker execution timing and document the unsplit IPC boundary.
- `docs/comfy/bench/scripts/build-native.cmd`, `speed-table.ps1`: honor the worktree and
  optional `PHOTOCRAFT_CLI`, retain command logs, optional `-SaveImages`, optional
  `-UpscaleLora default|none|restore` for controlled comparisons.
- `docs/comfy/bench/scripts/profile-comfy-upscale.py`: optional `-ProfileComfy` server launcher
  with node wall timings and asynchronous CUDA events; redirects writable server folders into
  the benchmark output directory. It patches the running process only, not installed files.
- This report, the fork dev log, roadmap follow-up, and benchmark link record the open target.

Complete exported 1728×2304 RGBA images were compared before/after on **each route at seeds
20 and 21**. All four comparisons have **RGB MAE = 0, RGBA MAE = 0, maximum difference = 0**:
byte-identical, stronger than the ≤1/255 gate. Local record: `target/upscale-parity.json`.
The no-Restore diagnostic is intentionally excluded from this parity claim.

## Reproduction

```powershell
cmd /c docs\comfy\bench\scripts\build-native.cmd photocraft-cli
$env:PHOTOCRAFT_CLI = "$PWD/target/release/photocraft-cli.exe"
$env:QIR_PHASE_PROFILE = '1'
powershell -File docs/comfy/bench/scripts/speed-table.ps1 -Routes native -Only "upscale 2x" -Out "$PWD/target/upscale-after-native" -SaveImages
$env:QIR_PHASE_PROFILE = $null
powershell -File docs/comfy/bench/scripts/speed-table.ps1 -Routes comfy -Only "upscale 2x" -Out "$PWD/target/upscale-after-comfy" -SaveImages -ProfileComfy
powershell -File docs/comfy/bench/scripts/speed-table.ps1 -Routes native -Only "upscale 2x" -Out "$PWD/target/upscale-after-native-unprofiled"
powershell -File docs/comfy/bench/scripts/speed-table.ps1 -Routes native -Only "upscale 2x" -Out "$PWD/target/upscale-native-no-restore" -UpscaleLora none
powershell -File docs/comfy/bench/scripts/speed-table.ps1 -Routes comfy -Only "upscale 2x" -Out "$PWD/target/upscale-comfy-restore" -ProfileComfy -UpscaleLora restore
```

Each invocation starts/stops its own inference server. Do not run these in parallel. The ComfyUI
profiler also supplies writable input/output/temp/user directories, necessary in this restricted
session. Re-running into the same directory replaces its logs/images.

## Remaining work

1. Create the native repository worktree with sufficient filesystem access. Optimize the
   **runtime LoRA side path first**, using the actual six tile requests and Restore 0.5 in a
   qir/gn-diffusion harness. Investigate shared/batched down projections and fused scaled-add
   materialization. Keep native before/after rounding or prove ≤1/255 final-image MAE. Blindly
   merging and requantizing base INT8 weights is not an established parity-preserving change.
2. Cache unchanged adapter bindings/transposed layouts to reduce setup allocations, but do not
   present this small setup cost as the principal fix. Measure live/pool memory: retaining raw
   and transposed adapter tensors can also affect the decode threshold.
3. Measure prefix f32 attention and decode memory under Restore. Decode alone accounts for
   about 1.5 s of its measured penalty. Preserve the safety margin unless real peak-memory
   evidence supports a revised policy; any different tiling policy needs final-image parity.
4. Rebuild/re-measure the app after integrating native changes; this report does **not** claim
   the requested speed target or any native-worker implementation was completed.
5. Separate directional IPC timings would require instrumentation in gn-engine's client/worker;
   the measured combined residual is already too small to explain the remaining seconds.

## Validation

- Engine unit suite: **1935 passed, 15 ignored**. After the final test-only iterator lint fix,
  the generative/alignment subset was rerun: **104 passed, 1 ignored**, including upscale.
- Strict clippy passed for both touched crates, **all targets**, with `-D warnings`.
- `cargo xtask layers`: **32 crates, no violations**. `cargo xtask wasm`: passed all listed
  L0?L6 crates, including ui-egui and the heif feature check.
- Adversarial command check: **1 passed**, no command panic/hang.
- Genai: **28 passed** with the Windows detached-child test excluded. The full run had that
  single failure at `sidecar.rs:584` (`pid_alive`); it uses `tasklist`, which independently
  returns **ERROR: Access denied** in this session. That file was not changed. Standalone
  genai unit tests need `--features fake-server` for this checkout's existing fake helpers.
- The broad release integration-suite build was interrupted before execution: it links 69
  integration binaries. This report claims the unit/subset and panic-check results above,
  **not** completion of the entire engine integration suite.

Release unit/test builds used the command-line-only override
`--config 'profile.release.package.photocraft-engine.codegen-units=16'` to avoid a single
large codegen unit; no Cargo profile source was changed. Clippy uses
`target/upscale-checks`; layer/wasm/perf checks use `target/upscale-xtask`.

```powershell
cargo test --release -p photocraft-engine -p photocraft-genai --lib --config 'profile.release.package.photocraft-engine.codegen-units=16'
cargo test --release -p photocraft-engine --lib generate_ --config 'profile.release.package.photocraft-engine.codegen-units=16'
cargo test --release -p photocraft-engine --test panic_hunt --config 'profile.release.package.photocraft-engine.codegen-units=16' -- --ignored
cargo test --release -p photocraft-genai --features fake-server --lib -- --skip sidecar::tests::a_detached_child_is_tracked_by_pid_and_leaves_no_handle_behind
cargo clippy -p photocraft-engine -p photocraft-genai --all-targets -- -D warnings
cargo xtask layers
cargo xtask wasm
```

Check logs are retained as `target/upscale-*-tests.log`, `upscale-clippy.log`,
`upscale-panic-hunt.log`, `upscale-layers.log`, and `upscale-wasm.log`.

The final native-diffusion CLI rebuild passed using `build-native.cmd photocraft-cli`.
`rustfmt --check` on all three touched Rust files and `git diff --check` passed.
`cargo xtask perf --quick` completed successfully; results are retained at
`target/upscale-xtask/perf/results.json` and `target/upscale-perf-quick.log`. Quick mode does
not enforce full-size budgets, and no baseline/budget/scorecard was changed. Its build used
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16` and six build jobs in that separate target directory.

The release alignment benchmark ran after the other builds/benchmarks finished, comparing
both searches in one process and asserting identical drift decisions:

| Synthetic input | Previous sequential search | Parallel search | Speedup |
|---|---:|---:|---:|
| 992?992 tile | 101.782 ms | 17.558 ms | 5.80? |
| 6000?4000 (24 MP) | 3155.633 ms | 466.966 ms | 6.76? |

A preceding Cargo-driven run measured 99.786 ? 17.870 ms and 3148.538 ? 456.601 ms,
respectively. The repeat reused the passing release unit executable. Final log:
`target/upscale-alignment-performance.log`.

```powershell
cargo test --release -p photocraft-engine --lib generate_realign::tests::alignment_performance --config 'profile.release.package.photocraft-engine.codegen-units=16' -- --ignored --nocapture
```

Generated local path-patch changes to Cargo.lock were discarded; dependency manifests and
profiles are unchanged. Native source hashes were rechecked against the recorded snapshot
and remain unchanged. No commit was made.
