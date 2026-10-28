# Plan: the ArtCraft experience in PhotoshopEX

A hand-off plan for an agent (or a person) to build, in this checkout
(`C:\Users\5090\Documents\PhotoshopEX`, branch `photoshopex`), the generation experience described
in [`artcraft-study.md`](artcraft-study.md) §6: a **Generate panel**, a **Generations panel** with
a persistent library of results, a **Variations strip**, then a Generate hub, alerts and
preferences. Local compute throughout: where ArtCraft prints a credit cost, we print a time
estimate measured on this machine. Written 2026-10-11 against commit `ce5c692`; line numbers
below are from that commit and will drift, the symbol names will not.

---

## 0. How to work this plan

**Read first, in order:** the local `CLAUDE.md` (this checkout, its rules; not in the repository),
[`../../AGENTS.md`](../../AGENTS.md) (upstream's rules: never crash, everything is a command, thin
data-driven UI, never break wasm, tests are the gate), [`codebase-orientation.md`](codebase-orientation.md),
[`architecture.md`](architecture.md), [`artcraft-study.md`](artcraft-study.md) §3 and §6, the top
entries of [`devlog.md`](devlog.md), and the local `dev-environment-windows.md`.

**Rules that bind every phase**

- No `unwrap`/`expect`/`panic`/`unsafe` in our code (`panic_hunt`, clippy config). Errors become
  `EngineError` or status text.
- Every feature is an engine command with a params doc string (`CommandSpec.params`), an
  `enabled` predicate, tests against the in-process fake ComfyUI server
  (`photocraft_genai::fake::FakeComfy`), and a row in `docs/control-protocol.md`. The UI only
  calls commands (`app.run(id, params)`) and reads state.
- Shared upstream files (`commands.rs`, `menus.rs`, `menu_catalog.rs`, `state.rs`, `panels.rs`,
  `dock.rs`, `control.rs`, `jobs.rs`) get surgical edits; new behaviour lives in new modules.
- Every user-visible string goes through `tl!("…")` and gets 13 translations appended with
  `docs/comfy/bench/scripts/append-translations.ps1` (it reads `pref-translations.tsv` next to it:
  rows `lang<TAB>Source<TAB>Translation` for cs, de, es, fr, id, it, ja, ko, pl, pt-br, ru,
  zh-hans, zh-hant). The tests `every_tl_literal_is_translated`,
  `complete_languages_translate_every_menu_string` and `preference_labels_are_translated`
  (`crates/ui-egui/src/i18n/mod.rs`, `prefs_ui.rs`) fail otherwise.
- Every preference added is read by code in the same change (`tests/prefs_usage.rs` enforces).
- Native-only things (files, sounds, threads) are gated with `web_unavailable()` /
  `cfg(not(target_arch = "wasm32"))`; `cargo build --target wasm32-unknown-unknown -p photocraft-engine`
  must stay green.
- Finish each phase with: `cargo test -p photocraft-engine -p photocraft-genai`,
  `cargo test -p photocraft-ui-egui` (use `CARGO_TARGET_DIR=target-ui` for the UI crate to keep
  the release build's lock free; the one known baseline failure is
  `input_tests::eyedropper_and_alt_sampling_show_a_pipette`), `cargo xtask layers`,
  `cargo xtask parity`, `cargo xtask scorecard`, the wasm engine build, a release build of the
  GUI (`cargo build --release -p photocraft`; the exe cannot be overwritten while it runs), a
  dev-log entry, roadmap markers, and one commit per phase on `photoshopex`.
- Live checks run against the local ComfyUI (`127.0.0.1:8188`, started with
  `powershell -File C:\Users\5090\ComfyUI\start-comfyui-fast.ps1`), through the CLI
  (`target\release\photocraft-cli.exe run <image> --cmd <id> --params "<json>" --out <png>`;
  every quote inside the JSON escaped as `\"`; `--new "{\"width\":512,\"height\":512}"` instead
  of a file when no document is needed). Scripts from earlier sessions are in
  `docs/comfy/bench/scripts/` (read `enhance-live.ps1` for the pattern).

**Decisions** (answered by the owner on 2026-10-11):

1. Panels on by default in the Essentials workspace: **on**.
2. Library cap: **500 results / 5 GB**, oldest unpinned completed results dropped first.
3. Sounds: behind a cargo feature `alerts` (`rodio`), on in the desktop release build, off in CI
   and wasm; the preference defaults to off.

**Correction to §3.1** found while building: job ids restart at 1 in every process, so a record's
id cannot be the `JobId`. The store assigns its own ids (unique across processes: unix
microseconds, bumped on collision) and maps the running `JobId` to it.

---

## 1. What exists today (the ground)

| Piece | Where | Notes |
|---|---|---|
| Generative commands `generate.fill / image / expand / edit / removeBackground / similar / splitLayers / info / variation / free / health / models / enhancePrompt` and `select.byText / byPoint / subjectML` | `crates/engine/src/generate_cmds.rs`, `select_ml_cmds.rs`; registered at `commands.rs:1099` | Each is a background job through `crate::jobs::{run, edit_job}`; results are layers with a `GenerativeInfo` block (`generate_cmds.rs:233`, `set_generative_info` 258, `generative_info` 266: command, prompt, enhanced, template, seed, steps, guidance, edge, rect, name, width, height, transparent) |
| Templates (ComfyUI API graphs + `meta`) | `crates/genai/workflows/*.json`, parsed by `crates/genai/src/template.rs` (`Meta` at 99, `ModelSlot` at 70 with `nvfp4`) | `generate.models` lists them with installed marks, `autoFill/Expand/Edit/Matte`, `device`, `blackwell` |
| Backend | `crates/genai/src/comfy.rs` (`run` 503, `prepare` for the NVFP4 policy), `lib.rs` (`Request` 145, `Response`, `Health` 236, `GenerativeBackend` 250), `fake.rs` (`Options` 41: color, segments, missing_files, matte, vram_free, enhanced_text, device) | `run_switching` in `generate_cmds.rs` purges on a model-set change |
| Job system | `crates/engine/src/jobs.rs`: `JobCtx` 39, `JobEvent` 125, `JobInfo` 136, `Jobs` 210 (`running`, `events`, `recent`), `poll_jobs` 473, `cancel_job` 513, `job/jobs/job_on` 539–556 | Nothing persists; `recent` keeps a few ended jobs for `jobs.list` |
| UI: the task bar | `crates/ui-egui/src/generative_bar.rs` (+ `_tests.rs`) | Fill \| Edit mode, prompt, template picker with installed badges, variations 1–4, Enhance wand, progress, 1/n switcher, `ui.set` fields `generativeBar/Mode/Prompt/Template/Variations`, `ui.inspect.generativeBar` |
| UI: jobs | `jobs_ui.rs`: `tick` 93 polls and dispatches `on_event` (the bar hooks at 151), `status_progress` 268, the modal `dialog` 293 (skipped when `generative_bar::shows_job`) | |
| UI: dock | `dock.rs`: `Group` enum 27 (`ALL` 44, `key` 46, `default_height` 59, `tabs` 100, `tab_mut` 118, `shown` 134), `DockLayout` 160; `panels.rs`: `right_dock` 1031 (rail entries 1046), `dock_panels` 1082, `dock_body` 1111, `properties_body` 2259; `state.rs`: `Panels` 303, `DockTabs` 698; `menus.rs`: `UI_COMMANDS` window toggles 41–48, `panel_alias` 66, `window.toggle.*` handling 437; `menu_catalog.rs` Window rows 751–777; `control.rs`: `UI_SET_FIELDS` 73, `inspect` 614, `dockTabs` 270 | Adding a dock group touches exactly these |
| Thumbnails | `lib.rs`: `layer_thumb` 1241, `store_thumb` 1277, `thumb_image` 1305, `surface_fingerprint` 1295; `brand.rs` for a PNG → texture pattern | |
| Home screen | `canvas.rs`: `start_screen` (called at 1086), recent files list 1623; `chrome_ui.rs::home_button` 163 | |
| Preferences | `crates/engine/src/prefs.rs`: `Integrations` 711 (comfyServer, defaultEdit/GenerateModel, defaultFill/ImageTemplate, generativeTimeoutSecs, allowResearchModels, showGenerativeBar, enhancePrompts, modelPrecision), `choices` 918 (`MODEL_PRECISIONS`), `prefs_ui.rs` `humanize` 762 / `choice_label` 791; file `preferences.json` in `apps/photocraft/src/app_dirs.rs::config_dir()` (`services.rs::prefs_file` 57) | |
| Measured times (for ETAs) | `benchmarks.md` | 2511 Lightning 8 fill/edit ≈ 16 s at 0.75 MP (14 s at 1024² edit), 2511 40-step ≈ 145 s, Krea 2 Turbo 1024² ≈ 7 s fp8 / 5.8 s NVFP4, Qwen-Image-2.1 fill ≈ 25 s, SAM 3.1 1.8–4.6 s, enhancer 1–2 s, Split into Layers ≈ 72 s |

---

## 2. The target, in one picture

```
┌ title bar ─────────────────────────────────────────────── [⟳ 2] ───┐
│ options bar  [Home] …                                               │
├────────────────────────────────────────────┬───────────┬───────────┤
│                                            │ Variations│ ▸ rail    │
│   canvas                                   │ ┌───┐     │ Generate  │
│                                            │ │ ● │ 1/3 │ Generations│
│      ┌ task bar (compact Generate) ┐       │ ├───┤     │ Properties│
│      │ Fill|Edit  prompt  ✨ Generate ~16 s│ │   │     │ Layers    │
│      └───────────────────────────────┘     │ ├───┤     │           │
│                                            │ │38%│     │ ┌ Generate ┐
│                                            │ └───┘     │ │deck prompt│
│                                            │           │ │pill 3:2 1K│
│                                            │           │ │4 ✕ ✨ Gen │
│                                            │           │ ├Generations┤
│                                            │           │ │▒▒ ✓ ✓ ✓  │
│                                            │           │ │✗ failed… │
├────────────────────────────────────────────┴───────────┴───────────┤
│ status bar   Generative Edit ▬▬▬▭▭ 38 %  ×                         │
└────────────────────────────────────────────────────────────────────┘
```

Three surfaces: the **Generate** dock group (the prompt box; the task bar under a selection is
its compact form), the **Generations** dock group (feed + queue + library), the **Variations
strip** at the canvas edge. Everything they do is a command.

---

## 3. Phase 1: the Generations library and panel

**Goal.** Every generative job is recorded, survives a restart, shows up in a panel as pending /
failed / completed, and a completed result can be placed, recreated, saved and copied from, even
after its document was closed. This is the backbone ("jobs you can walk away from; nothing is
lost").

### 3.1 Engine: `crates/engine/src/generate_jobs.rs` (new module)

**Record.** One per generative job:

```rust
/// A generative job as the Generations panel and `generate.jobs` see it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerationRecord {
    pub id: u64,                 // the JobId's number; stable in the store
    pub created_ms: u64,         // unix ms (SystemTime; 0 on wasm)
    pub finished_ms: u64,
    pub command: String,         // "generate.edit" …
    pub label: String,           // the job label ("Generative Edit: …")
    pub prompt: String,          // as typed
    pub enhanced: String,        // the rewritten prompt, if any
    pub template: String,        // resolved template id (after `auto`)
    pub seed: u64,
    pub count: u32,              // variations requested
    pub params: Value,           // the command's params, for Recreate
    pub document: Option<String>,// the document's title at the time
    pub status: GenStatus,       // Pending | Running | Done | Failed | Cancelled
    pub progress: f32,
    pub message: String,         // last progress message
    pub failure: Option<GenFailure>,
    pub result: Option<GenResult>,
    pub dismissed: bool,
    pub pinned: bool,            // never evicted
}
pub struct GenFailure { pub kind: GenFailureKind, pub message: String }
pub enum GenFailureKind { ServerUnreachable, ModelFilesMissing, OutOfMemory, Cancelled, ServerError, BadInput, Other }
pub struct GenResult {
    pub layers: Vec<u64>, pub width: u32, pub height: u32, pub request: (u32, u32),
    pub ms: u64, pub timings: Vec<Value>,
    pub files: Vec<String>,      // stored result PNGs, one per variation, relative to the store dir
    pub thumbnail: String,       // 128-px PNG, relative to the store dir
}
```

`GenFailureKind` is derived from the error text and type in one place (`classify_failure(&EngineError) -> GenFailureKind`): `Unavailable` / "cannot reach" → `ServerUnreachable`; "missing" + "models" → `ModelFilesMissing`; "out of memory" / "OOM" / "CUDA" → `OutOfMemory`; `EngineError::Cancelled` → `Cancelled`; `BadParams` → `BadInput`; a ComfyUI node error → `ServerError`.

**Store.** `pub struct GenerationStore { dir: Option<PathBuf>, records: Vec<GenerationRecord>, dirty: bool }` on `Session` as `session.generations` (add the field next to `jobs`; `Session::new` makes an in-memory store; the app calls `session.set_generations_dir(config_dir.join("generations"))` at start-up, next to where it loads `preferences.json`). Files: `<dir>/index.json` (the records, written whole on change, debounced to once a second from `poll_jobs`), `<dir>/thumbs/<id>.png`, `<dir>/results/<id>-<i>.png`. Caps: 200 completed records or 2 GB of results, evicting the oldest unpinned; failed and cancelled records are kept for 30 days; `dismissed` hides, it does not delete. Writes are atomic (write `index.json.tmp`, rename). Corrupt or missing files never fail a command: a record whose files are gone shows "files missing" and offers Recreate.

**Hooks, surgical, in `jobs.rs`:** when `Session::start` registers a job whose command id starts with `generate.` or `select.by` / `select.subjectML`, call `self.generations.began(id, command, label, params)`; in `end_job` call `self.generations.ended(id, &outcome)`; in `poll_jobs` copy progress for running generative jobs (`ctx.fraction()`, `ctx.message()`) into the records and flush when dirty. The generate commands fill what only they know: after a successful apply, `generate_cmds` calls `s.generations.attach_result(job_id, GenResult{…})` with the layer ids, the template, the seed, the enhanced prompt and the PNG bytes of each result (encode with `photocraft_genai::png::encode_rgba8`, already used for uploads; the 128-px thumbnail from the first result through a Lanczos downscale: reuse `resize_rgba8` in `generate_cmds.rs`). The job id is available in the apply closures through `JobCtx` (add `pub fn id(&self) -> JobId` if it is not exposed).

**Commands** (add to `generate_cmds::specs()` or a new `generate_jobs::specs()` registered the same way at `commands.rs:1099`):

| Id | Params → result | Notes |
|---|---|---|
| `generate.jobs` | `{"limit":int=100,"includeDismissed":bool=false,"status":{"pending|running|done|failed|cancelled"?}} → {"jobs":[GenerationRecord…],"running":n,"unread":n}` | Newest first; `journal: false`; `enabled: always` |
| `generate.jobs.dismiss` | `{"id":{u64?},"which":{"completed|failed|stale|all"?}} → {"dismissed":n}` | `stale` = running for over 10 minutes with no progress |
| `generate.jobs.pin` | `{"id":u64,"pinned":bool=true}` | |
| `generate.jobs.clear` | `{"which":"completed|failed|all"} → {"removed":n}` | Deletes records and files; the UI confirms first |
| `generate.recreate` | `{"id":u64,"seed":{u64?=random},"variations":{int?}} → as the original command` | Re-runs `command` with `params` (seed replaced). If the original document is still open and the command needs it, run there; if it needed a selection or a document that is gone, fall back: fill/edit become `generate.image` with the stored prompt into a new document, with a status note |
| `generate.place` | `{"id":u64,"index":int=0,"target":"layer|document"} → {"layer"|"document"}` | Decodes `results/<id>-<i>.png` into a new raster layer above the active one (or a new document of the image's size), with the stored `GenerativeInfo` re-attached |
| `generate.export` | `{"id":u64,"index":{int?=all},"dir":{path?=Preferences},"name":{str?=Preferences}} → {"files":[…]}` | Native only; naming from Phase 5's preference, default `{model}_{date}_{seed}` |
| `generate.jobs.setDir` | internal (not a command): `Session::set_generations_dir` | |

Also: `generate.info` already returns `GenerativeInfo`; add `"job": id` to it when the layer came from a recorded job (store the job id in `GenerativeInfo.job: u64`, serde default).

### 3.2 UI: the Generations dock group

- `dock.rs`: add `Group::Generations` (key `"generations"`, tabs `&["Generations"]`, default height 320, compact 160, min 120) to `Group::ALL` **below Properties and above Character**, and `Group::Generate` (Phase 2) above it. Update `key`, `default_height`, `compact_height`, `tabs`, `tab_mut`, `shown`, `shown_mut`, `scrolls_itself` (Generations scrolls itself). `state.rs`: `Panels.generations: bool` (`#[serde(default = "default_true")]`), `DockTabs.generations`.
- `panels.rs`: rail entry `("grid-2x2", tl!("Generations"), Group::Generations)` with a running-count badge drawn over the icon (small accent disc with the number; 0 = no badge); `dock_body` arm `(Group::Generations, _) => crate::generations_panel::panel(app, ui)`.
- `menus.rs`: `("window.toggle.generations", "Generations", &["Window"], None)` in `UI_COMMANDS`; `panel_alias`: `"generations" => "window.toggle.generations"`. `menu_catalog.rs`: `(&["Window"], "Generations", None, "window.panel.generations")` in alphabetical position. Translations for "Generations".
- `control.rs`: nothing new for visibility (`panels.generations` and `dockTabs.generations` flow through serde); add `"generations": crate::generations_panel::inspect(app)` to `inspect` (the view mode, the counts, the ids shown) and `generationsView` (`grid|list`) to `UI_SET_FIELDS` (bump the array length).
- `generations_panel.rs` (new) + `generations_panel_tests.rs`:
  - State in `UiState`: `pub generations: GenerationsPanel { view: Grid|List, last_viewed: Option<u64>, unread: BTreeSet<u64>, selecting: bool, selected: BTreeSet<u64> }` (serde; `#[serde(skip)]` for transient).
  - Header row: title, grid/list toggle (`icons::button` with "grid-2x2" / "align-justify"), a select toggle, an overflow menu (Clear completed, Clear failed, Remove all… each with `widgets::dialog_buttons` confirmation in a small modal).
  - Body: three sections with `widgets::section_label`: **In progress** (cards: shimmer = `t.hover` fill, real `progress` from `app.session.job(id)`, "38 %", ETA = `max(0, estimated − elapsed)` where estimated comes from the template's `estimatedSeconds` (Phase 2 meta; until then from a small table in code keyed by template family), the prompt clamped to three lines, the template name, "Generating 4 · results appear one at a time" when count > 1, Cancel ×), **Failed** (a red status chip with the `GenFailureKind` label, the message in a tooltip, the prompt, the time, Dismiss, Recreate), **Completed** (cards with the thumbnail texture, a "Last viewed" badge, hover actions **Place** (`generate.place`, target layer when a document is open else document), **Similar** (`generate.similar` when the layer still exists, else `generate.recreate`), **Save** (`generate.export`), **Copy prompt** (clipboard via `ui.ctx().copy_text`); click = select the result layer when its document is open (`layer.select`), else `generate.place {target: document}`; the card shows the template's family glyph and name).
  - Grid: cards 3 per row at the default dock width, square thumbnails letterboxed; list: 48-px thumbnail rows with the prompt and status. Virtualise: draw only rows in the visible rect (`ui.available_rect` and `ScrollArea::show_rows`), never more than 200 records at once; `generate.jobs {limit: 200}` once per second at most (cache the result in the panel state, refresh on `JobEvent`s and every 1 s while anything runs).
  - Thumbnails: a `HashMap<u64, TextureHandle>` in the panel state; decode `thumbs/<id>.png` lazily with `photocraft_codecs::decode` as `brand.rs` does; drop textures of records no longer listed.
  - Events: in `jobs_ui::on_event` right after the bar's hook (line 151) call `crate::generations_panel::on_event(app, &e)`: refresh, mark unread when the panel is not visible, and (Phase 5) play a sound.
  - The status bar keeps its progress readout; add the running count to the rail badge only.
- Tests (`generations_panel_tests.rs`, pattern of `generative_bar_tests.rs` with `FakeComfy` and `PhotocraftApp::new(Session, Services::default())`): a fill through `app.run` appears as completed with a thumbnail texture and the layer id; a failed run (fake with `missing_files` for every file → `ModelFilesMissing`) appears in Failed with the label; Place after closing the document creates a document of the result's size; Recreate re-runs with a new seed; dismiss hides; `ui.inspect` exposes `generations`; `ui.set {generationsView: "list"}` switches. Engine tests (`generate_jobs_tests.rs`): records lifecycle, classification of failures, caps and eviction with a temp dir, atomic index write, `generate.export` naming, wasm-safe no-dir mode.

### 3.3 Docs and done

- `control-protocol.md`: the new commands and `ui.set generationsView`, `ui.inspect.generations`.
- `architecture.md`: a "Generations library" paragraph (files, caps) and command rows.
- `roadmap.md`: Phase 4/5 markers; `devlog.md` entry with numbers (panel draw time with 200 records under 1 ms; index write time).
- Done when: a fill, an edit and an image generated from the GUI show as pending then completed with thumbnails; closing the document and clicking Place brings the result back; a restart keeps the list; a failed run (server stopped) shows "Server unreachable" with Recreate; all tests green.

---

## 4. Phase 2: the Generate panel (the prompt box) and references

**Goal.** One prompt box, docked, shaped by the chosen template's capabilities, with a reference
deck, aspect/size/count pickers, the Enhance wand and a Generate button that shows the ETA. The
task bar under a selection becomes its compact form.

### 4.1 Template capability flags (genai `Meta`, `template.rs:99`)

Add, all `#[serde(default)]`:

```rust
pub creator: String,            // "Alibaba Qwen", "Krea", "Black Forest Labs", "Meta"
pub description: String,        // one line for the picker ("the fast local edit model")
pub badges: Vec<String>,        // free text; the picker adds the ETA badge itself
pub references: References { max: u32, placeholders: Vec<String> },  // e.g. max 2, ["image2","image3"]
pub sizes: Sizes { aspects: Vec<String>, min: u32, max: u32, step: u32 },  // image templates
pub counts: Counts { max: u32, default: u32 },                        // default max 4 / 1
pub estimated_seconds: Estimate { base: f32, per_megapixel: f32 },    // from benchmarks.md
pub pages: Vec<String>,         // "image" | "fill" | "edit" | "expand" | "matte" | "split" | "select"
```

Fill them for every shipped template (values from `benchmarks.md`: 2511 Lightning 8 base 6 s + 13 s/MP; 2511 40-step base 20 s + 160 s/MP; Krea 2 Turbo base 2 s + 5 s/MP (NVFP4 4 s/MP); Qwen-Image-2.1 fill base 8 s + 22 s/MP; SAM 3.1 base 1.5 s; enhancers base 1 s; layered split base 60 s). `generate.models` returns them verbatim per template, plus `estimatedSecondsFor(width, height, count)` computed by the engine so the UI and scripts share one formula: `(base + per_megapixel × MP) × count`, MP from the request size after `request_size()` (the fill pipeline's 0.75 MP cap applies).

**References as a request feature.** `Request.references: Vec<Rgba8>` (genai `lib.rs:145`); `comfy.rs::run` uploads each and binds the template's `references.placeholders` in order (`{{image2}}`, `{{image3}}`); the fake records them. Engine params on `generate.edit` and `generate.image`: `"references": [ {"layer": id} | {"path": "…"} | {"job": id, "index": i} ]`, capped by the template's `references.max`, each rendered to RGBA (a layer through `photocraft_compose::render` of the layer alone; a path through `photocraft_codecs::decode`; a job through the Generations store). Templates to extend, each verified against the live server's `/object_info` before shipping: `qwen-edit-2511/edit` and `edit-lightning-8` (`TextEncodeQwenImageEditPlus` takes `image1..image3`: references bind `image2`, `image3`, `max: 2`); Qwen-Image-2.1's multi-reference graph as a new `qwen-2.1/edit-ref` template if ComfyUI's official 2.1 template has one (`max` up to 10; research licence); Krea 2's style-reference LoRA as `krea2-turbo/image-style` (`max: 1`) if the official template exists. Ship the deck UI with whatever verified; the panel hides the deck when `references.max == 0`.

### 4.2 UI: the Generate dock group

- `dock.rs`: `Group::Generate` (key `"generate"`, tabs `&["Generate"]`, default height 260, compact 180, min 150) placed **above Generations**. `Panels.generate: bool` (default true), `DockTabs.generate`, rail entry `("sparkles", tl!("Generate"), Group::Generate)`, `dock_body` arm → `crate::generate_panel::panel(app, ui)`, Window menu row and alias, translation "Generate".
- `generate_panel.rs` (new) + tests. State `UiState.generate: GeneratePanel`:

```rust
pub struct GeneratePanel {
    pub mode: String,            // "image" | "fill" | "edit" | "expand"
    pub into: String,            // "document" | "layer"   (image mode only)
    pub prompt: String,
    pub negative: String,        // behind a disclosure, off by default
    pub references: Vec<Reference>,  // {"layer":id} | {"path":…} | {"job":id,"index":i}
    pub template: BTreeMap<String, String>,  // per mode; "" = auto / preference
    pub aspect: String,          // "1:1" | "3:2" | "2:3" | "16:9" | "9:16" | "4:3" | "3:4" | "custom"
    pub width: u32, pub height: u32,
    pub count: u8,               // 1..=4
    pub enhance: Option<bool>,   // None = preference
    pub expand: [u32; 4],        // left, top, right, bottom for expand mode
    #[serde(skip)] pub job: Option<u64>, #[serde(skip)] pub focus: bool,
}
```

  Layout, top to bottom (see the study's wireframe): the **reference deck** at the left of the prompt (a fanned stack of up to three 36-px thumbnails with "+N"; hover expands to a row with × on each, drag to reorder with `egui_dnd`-free manual reordering (swap on drop), a "+" menu: *Layer…* (a submenu of the document's raster layers by name), *File…* (`rfd` picker through the existing file services), *From generations…* (the last 20 completed results); drop of an image file onto the panel adds a reference; the deck is hidden when the template's `references.max` is 0); the **prompt** (`TextEdit::multiline` growing from 2 to 6 rows, hint per mode: "Describe what you want in the image…", "Describe what to generate in the selection…", "Describe the change…", "Describe what continues the scene…"; a counter `n / 4000` right-aligned in `t.text_faint`, red past `MAX_PROMPT_CHARS`; a ⤢ button opening the prompt in a modal with the deck always expanded; Ctrl+Enter generates always, Enter generates when the preference says so); the **options row**: the **model pill** (family glyph + name; opens the picker of §5) · **aspect** (only in image mode: the template's `sizes.aspects`, a popover with little rectangle icons; "custom" reveals width/height `value_field`s on the 16-px grid) · **size** (1K / 1.5K / 2K presets within `sizes`) · **count** (1–4 as a dropdown) · **✨** (the enhancer, as the bar has it; shows "Enhancing…") · **Clear all** (asks when references are attached) · **Generate · ~24 s** (`widgets::primary_button`; the ETA from `generate.models`'s formula, recomputed when anything changes; while a job runs the button becomes **Cancel** with the percent); a **mode strip** below (`pill_tab`s Image · Fill · Edit · Expand; Fill needs a selection and says so; Expand shows four pad fields with the Crop tool's semantics) and an **into** toggle (new document / layer) in image mode.
  Generate builds the command from the state: `generate.image {prompt, negative, width, height, variations: count, template, references, enhance, target: into}`; `generate.fill {prompt, variations, template, enhance}`; `generate.edit {prompt, variations, template, references, enhance}`; `generate.expand {prompt, left, top, right, bottom, template}`. The result ids go to the Generations store automatically (Phase 1) and to the Variations strip (Phase 4); the panel itself only keeps `job`.
- **The task bar as the compact form** (`generative_bar.rs`): it reads and writes the same `GeneratePanel` state for prompt, template and count (drop `GenerativeBar.prompt/template/variations` in favour of the shared struct, keep `ui.set generativePrompt/…` as aliases writing the shared fields, so existing control-protocol tests keep passing), shows only mode, prompt, ✨, count and **Generate · ~16 s**, and hides itself while the Generate group is expanded in the dock (a preference `integrations.showGenerativeBar` already exists; add `"auto"`: show only when the panel is collapsed or hidden).
- `control.rs`: `generatePanel` in `UI_SET_FIELDS` (`{mode, into, prompt, negative, template, aspect, width, height, count, enhance, references}` validated as the bar's setter does), `ui.inspect.generatePanel`.
- Preferences (`Integrations`): `enter_to_generate: bool` (false), `group_models_by_family: bool` (true), `show_generative_bar: String` ("auto" | "always" | "never"; migrate the old bool through `#[serde(deserialize_with)]` or accept both). Labels + choices translated; read by the panel and the bar.
- Tests: the panel generates an image with two layer references and the fake receives three uploads with the right placeholders; mode switching changes the command; the ETA text matches the formula for a given template and size; Enter honours the preference; the bar and the panel share the prompt (`ui.set generativePrompt` shows in `ui.inspect.generatePanel.prompt`); hidden deck for a template without references; `references.max` enforced with a status message.

### 4.3 Docs and done

`control-protocol.md` (`generatePanel`, `references` params, `generate.models` additions), `architecture.md` (the template meta table), `models.md` (which templates take references, verified dates), roadmap, dev log with live timings of a two-reference 2511 edit. Done when: from the panel, an image with a layer reference generates and lands as a new document, a fill from a selection generates with the ETA shown and roughly right (within 50 %), the bar and the panel never disagree.

---

## 5. Phase 3: the model picker

**Goal.** ArtCraft's picker: grouped by family, newest first, each row with a face, a one-line
character, an ETA badge, an installed mark and the licence; one component used by the panel, the
bar and the generated dialogs' `template` field.

- `model_picker.rs` (new): `pub fn pill(ui, app, page: &str, current: &mut String) -> bool` (the trigger: family glyph + name; a glyph is a 14-px rounded square in the family's colour with its initial, colours from a fixed map: Qwen Image Edit `#7c3aed`, Qwen Image 2.1 `#9333ea`, Krea 2 `#f97316`, Flux `#0ea5e9`, SAM `#10b981`, Layered `#eab308`, other `t.text_dim`); the popover lists `generate.models` templates whose `pages` contain `page`, grouped by `family` when there are eight or more and the preference `groupModelsByFamily` is on (one row per family with the selected child's name as the description and a submenu, like `group-model-items.ts`), sorted newest first inside a family by a numeric-aware compare of the name; a row shows the glyph, the name, the description, badges (`~16 s` from the formula at the panel's current size and count; `research` in amber when `license == Research`, greyed and non-selectable unless the preference allows; `not installed` when `installed == Some(false)`, still selectable, the engine's error says what is missing), and a check on the current one; an **Auto** row first ("the fastest permissive tier the server has").
- Replace the bar's `widgets::dropdown` template picker and the dialogs' `template` combo with it; keep `generate.models` as the only data source (probe in the background as the bar does today, `apply_installed`).
- Tests: grouping threshold, newest-first order, research gating, the ETA badge text, the glyph map covers every shipped family (a test iterates `template::builtin()`).

Done when the picker reads like ArtCraft's and `generative_bar_tests` still pass with the new component.

---

## 6. Phase 4: the Variations strip

**Goal.** Candidates beside the canvas as thumbnails you click, with pending placeholders, instead
of the 1/3 arrows.

- `variations_strip.rs` (new): drawn in `canvas.rs` after `canvas_view` (right edge by default; preference `integrations.variationsStrip: "right" | "bottom" | "off"`), only when the active document has a **bundle**: the layers of the last generative job on this document (`generative_bar.results` today; generalise to `app.ui.bundles: BTreeMap<DocId, Vec<Bundle{job: u64, layers: Vec<u64>, shown: usize}>>`, newest bundle first, at most the last 5 bundles per document), plus pending placeholders for running jobs on this document (from `app.session.jobs()` filtered by `document`, with their percent).
- A tile is a 56-px square: the layer thumbnail (`app.layer_thumb`), a highlight ring on the shown one, a × on hover (`generate.dropVariation {layers, index}`: deletes that layer and re-indexes; new command, journaled), a pin/keep button on the shown one (`generate.keepVariation {layers, index}`: deletes the others, new command). Click = `generate.variation {layers, index}` (exists). A header with "1/3" and a **Similar** button (`generate.similar` on the shown layer, which appends a new bundle). Pending tiles show the shimmer and the percent; a failed job shows a small red tile with the reason on hover and a Retry (`generate.recreate`).
- `generative_bar.rs`: the 1/n switcher goes; `valid_results`/`show_variation` move to the strip module; `on_event` records bundles for any generative command on the document.
- Tests: three variations show three tiles with the first highlighted; click switches visibility in one undo step; × removes a layer; keep flattens; pending placeholder appears while a fake job is "running" (use `Session::start` with `jobs.spawn` as `jobs_ui_tests` does); `ui.inspect.variations` lists bundles.

Done when a Fill with 4 variations lets you compare them by clicking tiles, and Keep leaves one layer.

---

## 7. Phase 5: the finish

### 7.1 Generate hub

- On the Home screen (`canvas.rs::start_screen`): a "Generate" section of cards under the recent files, two rows: **Create** (Image → opens the Generate panel in image mode with a new document dialog; needs nothing) and **Edit** (Fill, Edit, Expand, Remove Background, Split into Layers, Select by Text; each opens a document picker when none is open, then the command's dialog or the panel in that mode). Card = tinted icon tile (lucide names: image, square-dashed, wand-sparkles, scaling, scissors, layers, text-cursor), uppercase mono label, one sentence, a badge (NEW for the first release, SOON for Angles). Also `Help › Generative Features…` opening the same cards in a modal, and `Window › Generate` focusing the panel.
- The cards call `menus::invoke` with the command ids; nothing new in the engine.

### 7.2 Alerts (sounds and toasts)

- Preference `integrations.alerts: "off" | "done" | "doneAndFailed"` (default `off`), label "Alerts", choices translated.
- Implementation behind a cargo feature `alerts` of `photocraft-ui-egui` (on in `apps/photocraft` by default, off in CI's `--no-default-features` lane and wasm): `alerts.rs` with two synthesised PCM blips (done: two rising sines 880/1320 Hz, 90 ms each; failed: one 220 Hz, 160 ms) played through `rodio` on a background thread; no asset files. Without the feature the module is a no-op. `generations_panel::on_event` plays them. The status toast we have stays.

### 7.3 Save to disk

- Preferences `integrations.save_results_dir: String` (empty = ask), `integrations.save_results_naming: String` (default `{model}_{date}_{seed}`; tokens `{model}`, `{template}`, `{date}` as `YYYY-MM-DD_HHMMSS`, `{seed}`, `{prompt}` truncated to 40 safe characters, `{n}`), validated in `prefs.rs::validate` (no path separators, at least one token). `generate.export` (Phase 1) reads them; the Generations panel's Save and a "Save all selected" in select mode use them; `generate.export` with `dir` absent and the preference empty returns an error the UI turns into a folder picker.

### 7.4 Angles (optional, last)

- Research step first, written into `models.md`: is there a permissive, Comfy-supported "angles" LoRA for Qwen-Image-Edit-2511 (ArtCraft's catalogue names `qwen_edit_2511_angles`)? Verify the file, the licence, and a graph against `/object_info`. If yes: template `qwen-edit-2511/angles` with placeholders `rotation`, `tilt`, `zoom` folded into the prompt the way the LoRA's card prescribes; command `generate.angles {"rotation":0..315 step 45,"tilt":-30|0|30|60,"zoom":0|5|10,"variations"}` producing layers; a dialog with an orbit widget (egui painter: a sphere with a camera marker dragged around it, snapping to the allowed values) and sliders, results into the Variations strip. If no: skip and say so in the dev log.

### 7.5 Done

A dev-log entry per phase; `roadmap.md` Phase 4 ("Models and workflows as data") and Phase 5 ("Quality and depth") markers; `architecture.md` and `control-protocol.md` current; a short "How to use" section in `docs/comfy/README.md` for the three surfaces; a release GUI built; a recording or screenshots of the walkthrough in §9 for the owner.

---

## 8. Cross-cutting specifications

### 8.1 Data on disk

```
<config_dir>/generations/
  index.json            { "version": 1, "records": [GenerationRecord…] }
  thumbs/<id>.png       128 px, square letterboxed, from the first result
  results/<id>-<i>.png  full results, RGBA PNG (the engine's own encoder), i = variation index
```

`config_dir` is `apps/photocraft/src/app_dirs.rs::config_dir()` (overridable with
`PHOTOCRAFT_CONFIG_DIR`, which the tests use with a temp directory). The CLI shares the store, so
`photocraft-cli run … --cmd generate.jobs` lists what the GUI made (one process at a time writes;
the index is re-read when its mtime changed before a write).

### 8.2 Failure taxonomy (`GenFailureKind` → label)

| Kind | Label (translated) | Typical source |
|---|---|---|
| `serverUnreachable` | Server unreachable | `Error::Unavailable` from `comfy.rs` |
| `modelFilesMissing` | Model files missing | `generate.models` installed false / server "value not in list" |
| `outOfMemory` | Out of memory on the server | ComfyUI node error containing "out of memory" |
| `cancelled` | Cancelled | Esc, Cancel |
| `serverError` | Server error | any other node error (message in the tooltip) |
| `badInput` | Invalid request | `EngineError::BadParams` |
| `other` | Failed | everything else |

### 8.3 ETA formula

`seconds = (base + per_megapixel × megapixels) × count`, megapixels from the request size the
command will send (`request_size()` for fill/edit/expand; `width × height` for image), rounded to
a whole second, shown as `~16 s`, `~2 min 25 s` above 90 s. The Generations panel's remaining time
is `max(0, seconds − elapsed)` while progress < 0.05, then
`elapsed × (1 − progress) / progress` once the server reports steps (real progress wins).

### 8.4 Control protocol additions (summary)

Commands: `generate.jobs`, `generate.jobs.dismiss`, `generate.jobs.pin`, `generate.jobs.clear`,
`generate.recreate`, `generate.place`, `generate.export`, `generate.dropVariation`,
`generate.keepVariation`, optional `generate.angles`. `ui.set`: `generationsView`,
`generatePanel {…}`, the existing `generativeBar/Mode/Prompt/Template/Variations` as aliases.
`ui.inspect`: `generations`, `generatePanel`, `variations`. Menu ids: `window.toggle.generate`,
`window.toggle.generations`, `help.generativeFeatures`.

### 8.5 Performance budgets

Panel draw under 1 ms with 200 records (virtualised rows, textures cached, no PNG decode on the
UI thread except the first time a thumbnail is seen, at most two decodes per frame); index write
under 20 ms (write on a 1 s debounce from `poll_jobs`, never per frame); result PNG encode on the
job thread, not the UI thread; thumbnails 128 px.

### 8.6 Tests to add, by file

| File | Covers |
|---|---|
| `crates/engine/src/generate_jobs_tests.rs` | record lifecycle through fill/edit/image/select against the fake; failure classification; eviction; atomic index; `generate.place` after close; `generate.recreate` fallbacks; `generate.export` naming |
| `crates/engine/src/generate_refs_tests.rs` | references from a layer, a file, a job; placeholder binding order; `references.max`; the fake's recorded uploads |
| `crates/genai/src/template.rs` tests | every built-in template has `estimated_seconds`, `pages`, `creator`; `references.placeholders` exist in the graph |
| `crates/ui-egui/src/generations_panel_tests.rs` | pending → completed cards; failed row; hover actions run commands; view toggle; inspect/set |
| `crates/ui-egui/src/generate_panel_tests.rs` | command built per mode; ETA text; Enter preference; deck caps; shared state with the bar |
| `crates/ui-egui/src/model_picker_tests.rs` | grouping, order, gating, badges |
| `crates/ui-egui/src/variations_strip_tests.rs` | tiles, click, drop, keep, pending |
| existing `generative_bar_tests.rs`, `dock_tests.rs`, `jobs_ui_tests.rs`, i18n and prefs tests | keep green |

### 8.7 Live verification scripts (write them into `docs/comfy/bench/scripts/`)

`generations-live.ps1` (three commands through the CLI, then `generate.jobs` shows them with
files on disk; `generate.place` into a new document; `generate.export` names), `panel-live.ps1`
(the GUI through the control channel: `ui.set generatePanel {…}`, `ui.menu.invoke generate.fill`,
poll `generate.jobs`, screenshot with `ui.screenshot` if available, as `snapshots-bar.ps1` does).

---

## 9. Acceptance walkthrough (the owner's test)

1. Launch PhotoshopEX with ComfyUI running. The Essentials workspace shows Generate and
   Generations in the dock.
2. In the Generate panel, mode Image, type "a lighthouse at dawn, photo", pick Krea 2 Turbo in
   the picker (grouped by family, "~6 s" badge), count 4, Generate. Four pending tiles appear in
   Generations with percent and "~6 s"; the rail badge shows 1; a new document opens with four
   layers; the Variations strip shows four tiles; clicking one shows it; Keep leaves one.
3. Select a rectangle, mode Fill, "a red kite", Generate · ~16 s. The task bar is hidden because
   the panel is open (or shows the compact form when the group is collapsed).
4. Mode Edit, drag the kite layer into the deck as a reference, "make the sky stormy, keep the
   kite", Generate. The 2511 edit runs with two images; the result is a layer masked to nothing
   (whole picture).
5. Stop ComfyUI, Generate again: a Failed row "Server unreachable" with Recreate; start ComfyUI,
   Recreate succeeds.
6. Close the document. In Generations, hover the lighthouse result: Place opens it as a document;
   Save writes `krea2-turbo-image_2026-10-11_201501_12345.png` to the chosen folder; Copy prompt
   copies the prompt.
7. Quit and relaunch: the list is still there with thumbnails.
8. Home screen: the Generate cards open the panel in the right mode.
9. Preferences › AI Integrations: Enter to generate on; Alerts "done and failed" plays a blip on
   completion; Variations strip "bottom" moves it.
10. `photocraft-cli run --new "{\"width\":64,\"height\":64}" --cmd generate.jobs --params {}` lists
    the same records.

---

## 10. File map

**New files**: `crates/engine/src/generate_jobs.rs` (+ `_tests.rs`), `crates/engine/src/generate_refs_tests.rs`,
`crates/ui-egui/src/generations_panel.rs` (+ `_tests.rs`), `generate_panel.rs` (+ `_tests.rs`),
`model_picker.rs` (+ `_tests.rs`), `variations_strip.rs` (+ `_tests.rs`), `alerts.rs`,
`crates/genai/workflows/qwen-edit-2511-edit*.json` (reference slots), possibly
`qwen-2.1-edit-ref.json`, `krea2-turbo-image-style.json`, `qwen-edit-2511-angles.json`,
`docs/comfy/bench/scripts/generations-live.ps1`, `panel-live.ps1`.

**Touched files (surgical)**: `crates/engine/src/jobs.rs` (two hooks + progress copy + `JobCtx::id`),
`lib.rs` (the `generations` field and `set_generations_dir`), `generate_cmds.rs` (`attach_result`,
`references`, `GenerativeInfo.job`, new specs), `prefs.rs` (preferences, choices, validation),
`crates/genai/src/{template.rs, lib.rs, comfy.rs, fake.rs}` (meta fields, `Request.references`,
uploads, fake recording), `crates/ui-egui/src/{dock.rs, state.rs, panels.rs, menus.rs,
menu_catalog.rs, control.rs, jobs_ui.rs, generative_bar.rs, canvas.rs, chrome_ui.rs, prefs_ui.rs,
lib.rs}`, `crates/ui-egui/src/i18n/*.tsv`, `apps/photocraft/src/main.rs` (the store dir),
`apps/photocraft-cli` (the store dir), `xtask` nothing, docs as listed per phase.

---

## 11. Glossary

- **Bundle**: the layers one generative job produced on one document (its variations).
- **Record**: one job in the Generations store.
- **Library**: the store's files on disk (results and thumbnails).
- **Compact form**: the task bar under a selection, showing a subset of the Generate panel.
- **ETA**: the time estimate from `estimated_seconds` and the request size; never a price.
