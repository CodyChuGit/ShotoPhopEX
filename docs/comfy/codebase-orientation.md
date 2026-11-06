# Codebase orientation for generative work

The parts of PhotoCraft a generative feature touches, with the exact types and files. Written
2026-10-08 against upstream commit `d836e34` (0.5.0); line numbers drift, names rarely do. The
analysis behind this page was produced by a read-only Codex pass over the workspace and then
verified by hand on the files cited; upstream's own [`docs/architecture.md`](../architecture.md)
remains the authoritative design document and [`AGENTS.md`](../../AGENTS.md) the rulebook.

## 0. Size and shape

24 library crates (`crates/`), 3 apps (`apps/`), one task runner (`xtask/`), ~250 k lines of Rust.
Measured on 2026-10-08 (`.rs` lines under `src/`):

| Crate | Lines | What it is |
|---|---:|---|
| `ui-egui` | 75 388 | the egui shell: panels, dialogs, canvas, menus, i18n, control protocol |
| `engine` | 61 733 | `Session`, the command registry (137 files, 70 `*_cmds.rs` modules), jobs, prefs |
| `algo` | 30 032 | imaging algorithms (filters, selection, content-aware fill, segmentation) |
| `io` | 11 655 | document ⇄ PSD/TIFF/flat formats |
| `psd` | 10 204 | standalone PSD/PSB reader/writer |
| `codecs` | 7 877 | PNG/JPEG/TIFF/WebP/… encode/decode |
| `compose` | 7 101 | the CPU compositor (reference oracle) |
| `text` 6 408 · `raw` 5 890 · `paint` 5 304 · `gpu` 4 389 · `cms` 4 039 · `doc` 3 976 · `format` 3 091 · `automation` 2 842 · `vector` 2 755 · `color` 974 · `raster` 956 · `geom` 939 · `plugins` 914 · `tablet` 857 · `ops` 333 · `heif` 194 · `testkit` 177 | | |
| `apps/photocraft` 4 003 · `apps/photocraft-cli` 536 · `apps/photocraft-web` 254 | | |

Layering (`xtask/src/layers.rs:35-72`, enforced by `cargo xtask layers`): L0 `geom cms color
raster` (+ standalone `psd codecs heif raw tablet`), L1 `doc`, L2 `ops paint algo text vector`,
L3 `compose gpu format`, L4 `io plugins` (+ reserved `tools viewport ml`), L5 `engine`, L6
`ui-egui automation` (+ reserved `platform`), apps exempt. UI crates (`egui eframe winit rfd
egui_kittest bevy*`) are only allowed from L6 up (`UI_CRATES`, `UI_MIN_LAYER`, lines 100–103). A
new crate must be added to `TABLE` or the check fails with "unknown workspace crate".

## 1. Commands

Everything user-visible is a `CommandSpec` (`crates/engine/src/commands.rs:15-28`):

```rust
type Run = fn(&mut Session, &Value) -> Result<Value>;                       // :11
type Enabled = fn(&Session) -> std::result::Result<(), String>;            // :12
pub struct CommandSpec {
    pub id: &'static str,            // "filter.blur.gaussianBlur": Photoshop's menu path as an id
    pub label: &'static str,
    pub menu: &'static [&'static str],   // ["Filter", "Blur"]; empty = not in menus
    pub shortcut: Option<&'static str>,  // "Cmd+Shift+N", mapped per platform by the UI
    pub params: &'static str,            // JSON-ish doc string agents read: {"radius":px=4}
    pub enabled: Enabled,
    pub run: Run,
    pub journal: bool,                   // false for queries
}
```

- Modules expose `pub fn specs() -> Vec<CommandSpec>` and are collected in `build()` with one
  `v.extend(crate::<module>::specs())` line each (`commands.rs:1020-1100`). `command_specs()`
  (`:116`) is a `OnceLock`'d static list; `find(id)` (`:121`) is the lookup.
- Param helpers live next to the struct (`commands.rs:127-154`: `bad`, `int`, `int_i32`,
  `u32_id`…). Each module validates its own params and returns `EngineError::BadParams`.
- `Session::execute(id, params)` (`crates/engine/src/lib.rs:401`) runs synchronously;
  `Session::start` (`jobs.rs:398`) runs job-capable commands in the background. Both go through
  `dispatch` (`jobs.rs:590-650`): find the spec, require an object (or null) for params, run the
  session's `authorize` gate (untrusted MCP/control sessions, `lib.rs:311`), drop a floating
  selection, inject the Channels-panel target, check `enabled` and job conflicts, refuse hidden
  targets, then call `run` inside `catch_unwind` (the last-resort never-crash guard).
- `Session::edit(label, |doc, active| …)` (`lib.rs:438-465`) is the one-undo-step API: clone the
  `Arc<Document>` (COW tiles make this O(layers)), run the closure, swap it in, record history
  unless the `"coalesce"` key matches the previous step.
- Enablement for menus: `disabled_reason_with` (`lib.rs:427`) → the spec's `enabled` or a mask
  target override, then `job_conflict`.
- Errors: `EngineError` (`lib.rs:111-127`): `UnknownCommand`, `Disabled(id, why)`,
  `BadParams{cmd,msg}`, `NoDocument`, `NoLayer`, `Other(String)`, `Cancelled`.

Two worked examples:

- **A filter with numeric params.** `filters.rs:194-263` (`run_filter`): parse params
  (`params_for`, e.g. `radius: f(p,"radius",1.0).clamp(0.1,1000.0)`), resolve the target layer,
  then `jobs::edit_job(s, &label, |doc, _, ctx| { … algo::apply_in_with(surf, &fp, area, bounds,
  selection, extent, ctl) … }, |fp| json!({"layer", "filter"}))`. The closure honours the
  selection (`doc.selection`, line 222), the layer locks (235–239), and smart objects (records a
  `SmartFilter { command, params, blend, opacity, visible }` instead of touching pixels, 244–249).
- **A command that creates a raster layer.** Layer via Copy (`edit_cmds.rs:239-274`): lift the
  selected pixels into a `Surface` (alpha × selection coverage, `edit_cmds.rs:67-105`), then
  inside `s.edit`: `let mut layer = Layer::raster(doc.next_layer_name("Layer"), fmt); *layer.surface_mut()… = surface; let nid = doc.insert_above(Some(id), layer); *active = Some(nid);`.

Adding a command: `docs/contributing.md` › "Adding a command" (find the id in
`crates/ui-egui/src/menu_catalog.rs`; algorithm in the lowest crate; module with `specs()`;
tests; dialog if it has params; `cargo xtask parity`).

## 2. Documents, pixels, selections, masks

- `Document` (`crates/doc/src/lib.rs:~640-670`): `size`, `mode: ColorMode`, `depth: SampleType`,
  `icc_profile`, `layers: Vec<Layer>` (bottom to top), `channels`, `guides`, **`selection:
  Option<Surface>`** ("grayscale coverage surface", `:657`), `metadata`, `global_light`, …
- `Layer` holds `content: LayerContent` (`Raster(Surface)`, `Group`, `Adjustment`, `Fill`,
  `Text`, `Shape`, `Smart(SmartObject)`), `mask: Option<LayerMask>`, `vector_mask`, `effects`,
  blend/opacity/fill/locks. `Layer::surface()` returns pixels for raster layers and cached
  text/shape/smart layers; `surface_mut()` only for raster layers (`doc/src/lib.rs:550-565`).
- `LayerMask { surface: Surface /* GRAY, 0 hide .. 1 reveal */, enabled, linked, density, feather }`
  (`doc/src/lib.rs:115-123`); `LayerMask::reveal_all()` / `hide_all()` build GRAY8 defaults (`:125-131`).
- `SmartObject { source: SmartSource, smart_filters: Vec<SmartFilter>, cache, filters_enabled,
  filter_mask, … }` (`doc/src/lib.rs:354-381`); `SmartFilter { command: String, params: Value, … }`
  (`:407`). `smart_cmds::add_smart_filter` (`engine/src/smart_cmds.rs:414-436`) appends one and
  can turn the selection into the filter mask. A generative edit on a smart object could be
  recorded the same way (re-runnable by command id).
- `Surface` (`crates/raster/src/lib.rs`): sparse 256² COW tiles, any `PixelFormat`
  (`crates/color/src/lib.rs:84`: `RGBA8`, `RGBA16`, `RGBA32F`, `GRAY8`, …). Reads:
  `read_region(rect) -> Vec<f32>` (normalised native channels, `:353`), `read_rgba_into(rect,
  &mut Vec<[f32;4]>)` (`:260`), `read_rgba8_into` (`:309`), `to_interleaved(rect) -> Vec<u8>`
  (encoded bytes, `:490`). Writes: `write_region` (floats), `from_interleaved(format, rect,
  bytes)` / `write_interleaved` (`:466-487`). **`write_interleaved` asserts the byte length**
  (`:476`): validate lengths before calling it, or the never-panic rule is broken.
- Flattened composite: `photocraft_compose::render(doc, rect) -> Buffer { rect, px: Vec<[f32;4]> }`
  (`crates/compose/src/lib.rs:37-42`, `:83`), straight alpha, parallel 256² tiles;
  `Buffer::to_rgba8()` (`:55`) quantises; `flatten_to_surface(doc, PixelFormat, background)`
  (`:227-248`) renders in bands into a surface of any depth. This is what exports and the PSD
  oracle use, so it is the right source for a model's input image.
- Selection → mask: `photocraft_algo::selection::mask_from_surface` (`crates/algo/src/selection.rs:36`)
  rasterises a selection surface over a rect into `Vec<f32>` (GRAY8 fast path);
  `mask_to_surface` (`:119`) is the inverse. There is no separate document-level feather value;
  feathering is baked into the selection surface (layer masks carry their own `feather`).
- PNG in memory: `photocraft_codecs::encode(&Image, Format, &EncodeOptions) -> Vec<u8>`
  (`crates/codecs/src/lib.rs:134`); PNG accepts Gray/GrayA/RGB/RGBA at **U8 or U16 only**
  (`codecs/src/codecs/png.rs:104-118`), so 32-bit documents must be converted before upload and
  the model's 8-bit result converted back. `photocraft_io::flat::export_flat`
  (`crates/io/src/flat.rs:282-324`) is the full flatten-and-encode path with colour handling.
- Colour: never assume sRGB or 8-bit in engine code (`AGENTS.md` rule 2). Convert at the
  boundary with `photocraft_cms::transform::cached(src, dst, opts)`; CMYK/Lab documents must be
  rendered to RGB for a model and the result placed as an RGB-converted layer.

## 3. Background jobs

`crates/engine/src/jobs.rs` (module docs at lines 1–19). Types: `JobId`, `JobCtx` (progress +
cancel, cloneable), `JobOutcome { Done(Value) | Failed(String) | Cancelled }`, `JobEvent`,
`JobInfo` (what `jobs.list`, the UI and MCP see), `Started { Done(Value) | Job(JobId) }`.

- `JobCtx::progress(fraction, message)` (`:65`, monotonic), `cancelled()` (`:78`), `check()`
  (`:88`, `Err(Cancelled)` for `?`), `stage(lo, hi, msg, |interrupt| …)` (`:104`) maps an
  algorithm's `photocraft_raster::Interrupt` onto a sub-range of the job.
- `jobs::run(s, label, lock_document, work, apply)` (`:278`): `work` runs on a worker thread
  (named `photocraft-job-N`, panics caught → `Failed`), `apply` on the UI thread; a document
  edited meanwhile makes the apply fail ("the document changed while it ran").
- `jobs::edit_job(s, label, |doc, active, ctx| …, finish)` (`:347`): the common case, a
  `Session::edit` whose body runs on a copy of the document in the worker. Inline (and on wasm)
  it is exactly `Session::edit`.
- `Session::start` (`:398`), `start_job` (`:440`, for non-command jobs such as opening files),
  `poll_jobs` (`:473`, called every frame), `wait_job` (`:497`, what the CLI and MCP use),
  `cancel_job` (`:513`, cancels at the next check and unlocks the document immediately),
  `job_conflict` (`:573`: edits of a locked document are refused with "“Label” is still running
  on this document; wait for it to finish or cancel it (Esc)").
- Commands `jobs.list` and `jobs.cancel` (`:739-773`).
- UI: `crates/ui-egui/src/jobs_ui.rs`: `run` (`:57`) calls `session.start` when
  `app.background_jobs` is on and returns `{"job", "pending": true}`; `tick` (`:93`) polls every
  frame, repaints at 60 Hz while jobs run, and lets Esc cancel the job in view; the status bar and
  a modal dialog show `JobInfo.progress/message`. The desktop app enables background jobs unless
  `PHOTOCRAFT_INLINE_JOBS` is set (`apps/photocraft/src/main.rs:~319`).
- Existing job-capable commands to copy from: every filter (`filters.rs:213`), Content-Aware Fill
  (`edit_menu_cmds.rs:288-414`), Content-Aware Scale (`edit_menu_cmds.rs:523`), Content-Aware
  Move (`retouch_cmds/content_aware_move.rs`), Photomerge (`photo_cmds.rs:408`), preset import.
- Tests to copy from: `jobs_tests.rs` (background vs inline equality, one undo step, cancel
  leaves the document unchanged, panics become errors, progress monotonic, non-object params
  rejected).

## 4. Networking today

None at runtime except local automation. `Cargo.lock` has `rmcp`, `tokio`, `wasm-bindgen-futures`,
`web-sys`, `js-sys`; **no** `reqwest`, `ureq`, `hyper`, `tungstenite`, `ewebsock`, `ehttp`.
`tokio` is a dependency of `crates/automation` (`Cargo.toml:24`, MCP over stdio and the loopback
bridge) and `apps/photocraft-cli` (`Cargo.toml:32`). The web app enables only DOM/Blob/URL/
Storage/event `web-sys` features (`apps/photocraft-web/Cargo.toml:37`), not `fetch` or
`WebSocket`. The control server binds `127.0.0.1` only (`apps/photocraft/src/control_server.rs`);
the MCP bridge and headless server refuse non-loopback addresses (`crates/automation/src/bridge.rs:28-51`,
`rpc.rs:273-280`). The only external fetch in the repo is `xtask` running `curl` for pinned
corpora (`xtask/src/pinned.rs:160-177`). A ComfyUI client therefore brings the first HTTP and
WebSocket dependencies into the tree; keep them pure Rust (rustls), native-only for now.

## 5. Preferences

`crates/engine/src/prefs.rs`: typed sections (serde structs, `camelCase`, `#[serde(default)]`),
collected in `pub struct Preferences` (`:744-784`: `general`, `interface`, `workspace`, `tools`,
`history_log`, `file_handling`, `export`, `performance`, `scratch_disks`, `cursors`,
`transparency_and_gamut`, `units_and_rulers`, `guides_grid_and_slices`, `plug_ins`, `type_`,
`enhanced_controls`, `raw_defaults`, **`integrations`**, plus shortcuts, menus, toolbar,
workspaces, dialogs…). `SECTIONS` (`:787`) lists the dialog pages in Photoshop's order.
Path-based access `Preferences::get/set/reset/get_path/set_path/check_value` (`:981-1118`)
validates choices, ranges, colours and shortcut syntax; commands `prefs.get`, `prefs.set`,
`prefs.reset` and `edit.preferences.<section>` (`:1367-1438`, `:1577-1617`) expose them to the
UI, CLI and MCP. Frontends persist JSON via `prefs_to_json` / `load_prefs_json` (the desktop app
under `%APPDATA%\Photocraft\`, the web build in `localStorage`).

Upstream already has **Edit › Preferences › AI Integrations…** → `edit.preferences.integrations`
(`crates/ui-egui/src/menu_catalog.rs:151`) backed by the `Integrations` section (currently
`allow_agent_control`, `control_port`). The ComfyUI settings belong there.

The scorecard's "settings that do nothing" audit (`xtask/src/scorecard.rs:256-389`,
`crates/engine/tests/prefs_usage.rs`) parses `pub struct Preferences` and checks that every leaf is
read somewhere outside the prefs model and its UI. A new preference must be read by code in the
same change, or `cargo xtask scorecard --check` reports it.

## 6. UI hooks

- **Menus**: `crates/ui-egui/src/menu_catalog.rs` is Photoshop's menu tree as data
  (`(path, label, shortcut, command id)` rows, e.g. `:97` `Content-Aware Fill… → edit.contentAwareFill`).
  A row is live when the id exists in the engine registry (`menus::is_live`). `menus.rs` holds
  the shell-only `UI_COMMANDS` (zoom, panels…), invocation routing and special handlers; document
  edits must be engine commands. Photoshop's **Generative Fill** rows are deliberately absent
  (`canvas_tool_menu.rs:37-39`), so adding them is a fork addition, not parity wiring.
- **Dialogs**: `filter_dialog.rs` parses the `params` doc string into controls (`parse_spec`,
  `:14-107`: ranges, choices, bools, ints, text, document pickers, grids, raw JSON) and previews
  the command on a proxy session (`preview_document`, `:402`); `dialogs.rs:322-375` confirms a
  dialog by extracting params and calling `app.run`.
- **Panels and state**: `state.rs` (`Panels` `:302-339`, `UiState` `:709-816`, serde) so the
  control channel can read and drive the UI (`ui.inspect`, `ui.set`); the options bar switches on
  the active tool in `panels.rs:465-550`; the status bar shows job progress (`panels.rs:971-1025`).
- **Theme**: colours and radii only from `theme::Tokens`; widgets from `widgets::*`
  (`docs/ui-design.md`).
- **Localisation**: every UI string through `tl!` (`crates/ui-egui/src/lib.rs:10-15` → `i18n::t`);
  `cargo xtask i18n-coverage` derives the English key set from `tl!` literals, the menu catalogue,
  the UI command table and engine command labels, and tests enforce catalogue coverage
  (`docs/localization.md`). New menu labels and bar strings need catalogue rows.
- **Visual verification**: `cargo run --release -p photocraft-ui-egui --example snapshot` renders
  the UI offscreen; the control channel's `ui.screenshot` captures the live app.

## 7. What already exists that is "smart" (all classical, no ML)

- `crates/algo/src/segment/` (max-flow, GMM, SLIC, GrabCut, saliency, focus heuristics) behind
  `smartselect_cmds.rs` (Quick Selection, Object Selection, Select Subject, Refine Edge, Focus Area).
- `cutout_cmds.rs:38-71`: **Remove Background** as a Quick Action: Select Subject → optional edge
  refinement → Background becomes a normal layer with a non-destructive mask. The generative
  version should reuse its layer/mask plumbing and only swap the matte source.
- `edit_menu_cmds.rs:288-414` + `crates/algo/src/content_aware.rs`: Content-Aware Fill
  (PatchMatch-style, rotations/scales/mirror, colour adaptation) as a background job in one undo
  step. This is the shape `generate.fill` takes, with the model replacing PatchMatch.
- `retouch_cmds.rs` (+ `patch.rs`, `content_aware_move.rs`): heal, patch, content-aware move.
- No `ml`, `genai` or `comfy` crate, no `onnx`/`ort`/`candle` in the lock file; `docs/architecture.md:127`
  lists `ml` as "not started"; `docs/roadmap.md` grades AI at ~0 %, deferred by decision #41.

## 8. Automation surface

- MCP (`crates/automation/src/server.rs`): `session_list`, `doc_open/new/save/export/inspect/
  render_preview/select/close`, `command_list`, `command_run`, `command_batch`, `jobs_list`,
  `jobs_cancel`, and bridge-only `ui_inspect/screenshot/pointer/menu_invoke/set`, `control_call`.
  `command_run { id, params, wait }` with `wait: false` starts a background job
  (`headless.rs:202-241`).
- Control protocol (`crates/ui-egui/src/control.rs`, `docs/control-protocol.md`): JSON lines on
  loopback with a bearer token: `engine.execute`, `jobs.list`, `jobs.cancel`, dialog operations,
  `ui.*` inspect/screenshot/pointer/menu/set.
- CLI (`apps/photocraft-cli/src/lib.rs:257-290`): `run` parses repeated `--cmd/--params`,
  dispatches through the headless automation layer, prints one JSON line per command, saves
  `--out`. `photocraft-cli commands --filter generate` will list the new commands from their
  `params` doc strings.
- `crates/automation/tests/agent_tasks.rs`: the ten-task MCP acceptance test to extend.

## 9. Testing conventions

- Command tests sit in the module (or `*_tests.rs` beside it): build a `Session`, open a synthetic
  document, `execute` with `json!`, assert document state, undo/redo, disabled states and bad
  params, at several depths.
- `crates/engine/tests/panic_hunt.rs` enumerates `command_specs()` and runs each command with
  adversarial params under `catch_unwind` and a timeout; ignored by default, run with
  `cargo test -p photocraft-engine --test panic_hunt -- --ignored`. New commands are covered
  automatically and must not panic.
- External services: no precedent in the repo. Existing opt-in tests use `#[ignore]` (timing,
  benchmarks, user-supplied files). Plan: a fake ComfyUI (`std::net::TcpListener`) for CI and one
  ignored live test behind `PHOTOCRAFT_COMFY_URL`.
- Lints: workspace `unsafe_code = "forbid"`, clippy `dbg_macro`/`todo` warn (`Cargo.toml:56-61`);
  clean crates add `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic,
  clippy::unimplemented, clippy::todo, clippy::unreachable)]` (e.g. `crates/raster/src/lib.rs:7`).
- CI (`.github/workflows/ci.yml`): fmt, clippy `-D warnings`, tests, `xtask layers`, `xtask wasm`,
  `scorecard --check`, the corpus job; nightly perf on macOS.

## 9.5 What the fork has added so far (2026-10-08)

- `crates/genai` (`photocraft-genai`, L4): `GenerativeBackend` trait, `Request`/`Response`
  (`Rgba8`, `Gray8` pixels), `template` (API-format graphs with typed `{{placeholders}}`,
  built-in `qwen-edit-2511/fill`), `comfy` (`ComfyClient` over ureq + tungstenite, `ComfyBackend`
  with the upload → prompt → socket/poll → view loop, cancellation via `/interrupt`), `png`, and
  `fake` (feature `fake-server`: an in-process ComfyUI stand-in for tests).
- `crates/engine/src/generate_cmds.rs`: `generate.fill` (background job via `jobs::edit_job`,
  result on a new masked layer), `generate.health`, `generate.models`; tests in
  `generate_cmds_tests.rs`. Registered with one line in `commands.rs`; `menu: &[]` until Phase 2
  adds the catalogue rows and translations.
- `crates/engine/src/prefs.rs` › `Integrations`: `comfy_server`, `default_edit_model`,
  `generative_timeout_secs`, `allow_research_models` (+ validation in `check_value`/`range`).
- `xtask/src/layers.rs`: `("genai", Class::Layer(4))`.

## 10. Gotchas checklist for the new crate

1. Register `genai` in `xtask/src/layers.rs TABLE` at L4; depend only on L0–L3 and standalone crates.
2. `crates/*` is a workspace glob (`Cargo.toml:3-4`): a half-written manifest breaks every build.
   Write manifests atomically (`AGENTS.md` §6).
3. `cargo xtask wasm` must pass: network code behind `cfg(not(target_arch = "wasm32"))` with an
   "unsupported on the web" error path; jobs already run inline on wasm.
4. No `unwrap`/`expect`/`panic`/indexing on input-derived values; `write_interleaved` asserts
   lengths, so check them first.
5. A new `doc` field (generative metadata) makes `photocraft-format` fail to compile until
   `manifest.rs`/`convert.rs` carry it with `#[serde(default)]` (`AGENTS.md` §8); map it in `io`
   or keep it as an unknown block for PSD.
6. New menu rows: `menu_catalog.rs` + `cargo xtask parity` (commit `docs/parity.md`); new UI
   strings: `tl!` + catalogue rows; new prefs: read them in code and run `cargo xtask scorecard`.
7. No JavaScript, no webview, no Python in the build. The plug-in sandbox has no network imports
   (`docs/plugins.md:39-45`), so ComfyUI cannot be a wasm plug-in.
8. Before a PR upstream: `contributors/people.toml` rules (`AGENTS.md` › Contributor credits).
