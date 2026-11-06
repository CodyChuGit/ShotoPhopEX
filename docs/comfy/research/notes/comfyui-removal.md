# ComfyUI removal — 2026-10-29

**Status: landed 2026-10-29 (checked by Claude in the main checkout).** On the real GPU
(RTX 5090), `cargo test --workspace --no-fail-fast` fails only the PC's known GPU-preview tests
(full_res_preview 3, pattern_preview_canvas 1) and `photocraft-gpu --test timing` (1–2 of 5;
that crate is unchanged, measured while another session's build loaded the machine); the engine
lib passes 1,928. The 39 graphics failures below are Codex's CPU-only run on the software
rasterizer and do not occur on the GPU. Added after Codex: the `comfyPath` → Models folder
migration on load (`prefs::tests::an_old_comfyui_install_becomes_the_models_folder_once`: the
portable `ComfyUI\models` or a checkout's `models`, only when no Models folder is set) and the
Generate panel's "Set Models Folder…" note when there is none.

Owner decision: PhotoshopEX uses only `photocraft --engine`, the gn-engine worker in the separate,
pinned genai-native repository. That repository was not changed.

Logical change groups:

1. Genai backend removal: deleted the ComfyUI HTTP/WebSocket client and process sidecar, their
   dependencies and protocol-only tests; kept the native worker and shared request/result types.
2. Engine routing: every generation, vision-language question and model query uses the native
   worker. Removed auto/ComfyUI route selection, research fallback hints, model-set tracking,
   NVFP4 choice and pre-run purges. Kept native status/start/restart/purge/stop and Generate Free.
3. Prefs and UI: removed comfyServer, comfyPath, comfyArgs, engine, engineStartOnLaunch,
   engineStopOnQuit, purgeModels and modelPrecision, plus their UI rows and obsolete translations.
   The launch/quit toggles controlled only the old sidecar. Status dot/page now show only native.
4. Templates: all 28 files are metadata-only. Every ID exists in the pinned gn-diffusion or
   gn-perception template index, including Krea image-style/edit, Qwen 2.1, 2511, Layered, SAM 3.1
   and enhance. There were no unserved template files to delete in this checkout. Removed graphs,
   minComfyVersion, saveNode, node/reference placeholders and NVFP4 alternatives. Native parameter
   declarations retain point/box prompt semantics.
5. Tests: FakeNative implements gn_engine::server::Engine and runs through server::serve over a
   bound loopback listener. It records typed RunRequests and raw blobs; supports echo/solid
   pictures, matte rectangles, per-prompt segmentation masks, chat, injected errors, progress and
   cancellation. Command/UI/CLI assertions check native requests. ComfyUI-only protocol, graph,
   upload, sidecar, version, file-precision and automatic-purge tests were removed.
6. Docs: current setup and architecture updated, progress entries added, and all 128 benchmark
   programs retained with comparison-only headers. Scripts using retired preferences/routes
   identify cdcedbd8 or direct ComfyUI HTTP calls as the way to run their historical comparison.

`integrations.modelsPath` (Models folder) and runtimePath remain. The implicit comfyPath/models fallback is
gone (old files migrate once on load, above). Existing weights retain ComfyUI’s
models layout; ComfyUI itself is not required. Generative timeout is retained and now cancels native worker runs at the configured deadline.
The packaged runtime fallback and externally
attached native-worker channel remain. Plain builds still require native-diffusion to serve
diffusion models; there is no automatic fallback to another engine.

Old-prefs regression: `prefs::tests::old_comfy_preferences_are_ignored_and_other_values_survive`
loads a JSON containing every retired key. load_prefs_json discards those keys before lenient
validation, reports no invalid preferences, preserves Models/runtime folders, quality, research
opt-in and an unrelated file-handling setting, and omits retired keys when saving.

Validation is CPU-only. Graphics tests temporarily request fallback adapters; the offscreen egui
renderer uses a local dependency override changing adapter selection to the software rasterizer.
Those validation overrides are not part of the delivered change. No inference or benchmarks run
on the GPU. The sibling craftrules checkout is absent from this worktree.

Gate results (own target directory: `target/comfy-removal`; no physical GPU used):

| Check | Result |
|---|---|
| `cargo test --workspace --no-fail-fast` | **FAIL**: 9,790 passed, 41 failed, 84 ignored, across 318 parent binary/doc-test summaries |
| Complete genai + engine + CLI unit suites after assertion corrections | **PASS**: 1,946 passed, 0 failed, 15 ignored (CLI 5, engine 1,927, genai 14) |
| `cargo clippy --workspace --all-targets -- -D warnings` | **PASS**, zero warnings |
| `cargo xtask layers` | **PASS**, 32 crates, zero violations |
| `cargo xtask wasm` | **PASS**, 26 successful checks, including the codecs HEIF feature |
| `cargo xtask parity` | **PASS**, 643/643 menu entries live; regenerated checklist |
| `cargo xtask scorecard --check` | **PASS**; regenerated scorecard: 187 settings (was 195), 26 inactive/hidden settings unchanged |
| `cargo build -p photocraft-cli` | **PASS**, plain CLI, no diffusion/CUDA build feature |
| Ignored `panic_hunt` adversarial command test | **PASS**, 1 test, no command panic or hang |
| Offscreen preferences snapshot | **PASS**, visually inspected at 1280×900: Models folder, runtime folders and native status; retired controls absent |
| Catalog validation / formatting / diff whitespace | **PASS**, 20 valid TSV catalogs with no duplicate keys; `cargo fmt`; `git diff --check` |

The full invocation used a local `egui-wgpu` adapter-selection override and temporarily forced
fallback adapters in seven graphics test fixtures. The override selects **Microsoft Basic
Render Driver, DX12, Cpu** (driver 10.0.19041.5794); the screenshot also reports `gpu:false` and
`canvas:cpu`. All seven fixtures were restored. Cargo.toml has no renderer patch, Cargo.lock
uses the normal registry dependency, and no graphics-source change is delivered. Logs and the
snapshot remain under gitignored `plan/` (`workspace-final.log`, `core-retest.log`,
`clippy-post.log`, `native-integrations.png`).

Two of the 41 failures were test-migration assertions and are now fixed: CLI batching counts
matte runs separately from the native worker’s subject-naming questions; the image-reference
assertion checks that exactly its one supplied reference is sent. Their complete unit targets
passed in the subsequent 1,946-test run. The other **39 failures are graphics tests on WARP**:

| Graphics target | Failed | Relation to the supplied PC baseline |
|---|---:|---|
| `photocraft-gpu/tests/parity.rs` | 23 | Additional |
| `photocraft-gpu/tests/vector_mask.rs` | 1 | Additional |
| `photocraft-ui-egui --lib` | 4 | Additional: three blend-hover cases and the navigator GPU texture case |
| `ui-egui/tests/adjust_preview_gpu.rs` | 2 | Additional |
| `ui-egui/tests/arbitrary_rotation_preview.rs` | 1 | Additional |
| `ui-egui/tests/canvas_16f.rs` | 3 | Additional |
| `ui-egui/tests/full_res_preview.rs` | 3 | Known baseline |
| `ui-egui/tests/layout_perf.rs` | 1 | Additional |
| `ui-egui/tests/pattern_preview_canvas.rs` | 1 | Known baseline |

Thus **35 additional graphics failures beyond the four known preview failures remain** under
the CPU-only restriction. They include `GPU readback failed: MapRangeError(... Buffer is not
mapped ...)` and pixel comparison mismatches in unchanged graphics code. A single-thread FXC
probe of `adjustment_layers` reproduced the same U16 buffer-readback failure (0 passed,
1 failed); a DXC probe stalled and was stopped. These experiments did not establish a working
software-renderer workaround. The normal physical-GPU baseline was deliberately not run.
The full-workspace acceptance gate remains blocked; this report does not claim a green tree.

The workspace test emits the Windows CLI linker warning LNK4199 for its unchanged
Windows.Data.Pdf delay-load setup. Wasm emits three dead-constant warnings in untouched
`display_color.rs`/`file_cmds.rs`; strict host clippy is warning-free. No real-model output or
inference performance was measured.

User-visible/API changes: generation has no ComfyUI fallback or engine choice; Models folder
is explicit for app-managed workers; retired settings load silently and disappear on save.
Lifecycle status now contains only its `native` object. `generate.models` drops the ComfyUI
Blackwell/NVFP4 alternatives and graph-reference fields. Native requests carry template defaults
and model-slot overrides in `params.models`. The retained generative timeout now cancels native
runs and reports a timeout while leaving the worker usable. Invalid external-worker endpoint
strings fail explicitly rather than bypassing folder validation.


Failing graphics test names from the full invocation:

- **tests/parity.rs**: `adjustment_layers`, `blend_if`, `channel_restrictions`, `documents_larger_than_the_texture_limit`, `duplicates_copy_their_source_texture`, `exterior_effects_merge_from_the_shape_at_any_fill`, `fills_and_dissolve`, `groups_masks_and_clipping`, `hdr_adjustments_preserve_range_and_cpu_parity`, `layer_effects_on_groups_clipping_and_fills`, `layer_effects_update_incrementally`, `pages_are_evicted_under_the_budget`, `pass_through_clipping_base_with_an_adjustment`, `rgba16f_fallback_path_renders`, `root_checkpoints_keep_clipping_groups_and_adjustments_whole`, `runs_of_normal_layers_blend_with_normal_code_alone`, `runs_of_plain_layers_blend_in_one_pass`, `small_layers_composite_over_their_bounds`, `stroke_effects_on_filled_and_stroked_shapes`, `stroked_shapes_with_clipped_layers`, `the_focused_view_stays_resident_after_a_full_refresh`, `type_layers_blend_with_text_gamma`, `xor_rounding_on_half_float_targets`.
- **tests/vector_mask.rs**: `vector_masks_match_the_cpu`.
- **photocraft-ui-egui --lib**: `canvas::blend_viewport_regression::blend_hover_multiple_layers_tracks_visible_canvas`, `canvas::blend_viewport_regression::blend_hover_pattern_tiles_and_multiple_views_reuse_current_pixels`, `canvas::blend_viewport_regression::blend_hover_refresh_is_limited_to_visible_canvas`, `navigator_gpu_tests::navigator_draws_the_gpu_canvas_texture_and_follows_edits`.
- **tests/adjust_preview_gpu.rs**: `preview_layer_composites_on_the_gpu_like_the_command`, `proxy_preview_composites_on_the_gpu`.
- **tests/arbitrary_rotation_preview.rs**: `size_changing_rotation_preview_is_visible_and_preserves_the_panned_center`.
- **tests/canvas_16f.rs**: `sixteen_bit_gradient_keeps_more_than_256_levels`, `sixteen_bit_gradient_on_screen`, `thirty_two_bit_preview_exposes_values_above_one`.
- **tests/full_res_preview.rs**: `adjustment_dialog_zoomed_out_skips_the_reduced_copy_when_low_res_is_off`, `adjustment_layer_drag_previews_at_full_resolution_when_low_res_is_off`, `filter_dialog_previews_at_full_resolution_when_low_res_is_off`.
- **tests/layout_perf.rs**: `layout_document_composites_on_the_gpu_like_the_cpu`.
- **tests/pattern_preview_canvas.rs**: `pattern_preview_repeats_the_document_on_gpu_and_cpu_canvases`.
