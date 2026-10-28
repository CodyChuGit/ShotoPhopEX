# PhotoshopEX dev log (continues the Comfy PhotoCraft log)

Newest entry first. Terse: what landed, numbers, what is still open. Upstream keeps its log in
the gitignored `log/devlog.md`; this one is tracked so the next session (any machine, any agent)
can pick up.

## 2026-10-27: App Icon: any image, one click (replaces the Ps tile)

The owner: the Ps tile should not live in the repository; the app should take any image, download
one, and switch between options in one click. So the tile is gone (`assets/app-icon/ps/`, and
`build.rs`, `app_icon.rs`, `brand.rs` point at upstream's icons again) and a feature replaces it.

**Landed**

- **Edit › Preferences › App Icon…** (`edit.appIcon`, `app_icon_ui.rs`), also one click on the
  title bar's mark: a gallery of tiles (Built-in plus the user's own), click one to use it,
  right-click to remove; **Choose Image…** (any format the codecs read), **Paste Image** (the
  clipboard), and an address field with **Download**. Each image is fitted into a transparent
  square of at most 256 px (box filter, alpha-weighted, never enlarged) and kept as a base64 PNG
  in the preferences (`appIcon: {active, icons: [{name, png}]}`, top level so the generic dialog
  doesn't show it; at most 12), so it works on the web build too and nothing ships.
- Applied live: the title-bar mark (a custom texture `brand.rs` prefers) and the window/taskbar
  icon (`ViewportCommand::Icon`). The executable's own file icon (Explorer, pinned shortcuts) is
  baked in at build time and stays the built-in one.
- Downloads: a new `Services::download` (a plain `fn`, run on a thread). The desktop app asks the
  system `curl` (Windows 10 1803+, macOS, Linux ship it; no console window, 30 s, 20 MB,
  http/https only) since the app builds no TLS stack. The web build has none, so the row hides.
- Automation: `edit.appIcon {use | remove | add: {name, data}}`, `ui.inspect.appIcon`.
- Tests: `app_icon_ui_tests.rs` (fitting, add/use/remove/cap/errors, the preferences round trip,
  a fake download through the window and one click back to Built-in). 13 catalogs got the 9 new
  strings. `input_tests`' harness pins `zoomWithScrollWheel: false`: three ⌥-wheel tests broke
  when the 2026-10-15 entry made the wheel zoom by default.

**Open**

- Not tried by hand in the running app yet (tests only). `input_tests::eyedropper_and_alt_sampling_show_a_pipette`
  is still the known baseline failure on this PC.

## 2026-10-16: the native-inference question, Quality mode, and the Object Selection tool's SAM 3.1 mode

- **Research** (five researchers, report in `docs/comfy/research/Native inference for PhotoCraft on RTX
  5090.md`, notes in `research/notes/`): no Rust runtime runs this model set; candle reaches ~0.65× ComfyUI's
  bf16 speed stock, burn's attention kernel is an open blocker, TensorRT has a FLUX path only, and
  in-process Python breaks the no-crash rule. Recommended: keep the ComfyUI protocol as the backend
  trait, ship an app-managed sidecar for zero setup, go native per model only after same-machine
  benchmarks. **NVFP4** is a GEMM and VRAM lever (1.6–2.6× per DiT step, 1.2–1.8× per image on a
  5090, with a quality cost under post-training quantisation), not a general speed-up; Lightning
  distillation dwarfs it. The owner chose **highest quality first, then optimise**.
- **Quality** (`integrations.quality`: `best` default, `balanced`, `fast`): what `auto` optimises
  for. `generate_cmds::auto_order(command, quality)` replaces the fixed orders (`best` = the
  full-step base templates; `balanced` = the 8-step Lightning tier first, the old behaviour;
  `fast` = the 4-step fill tier first), `likely_auto` takes the quality, and `backend()` keeps
  `modelPrecision: auto` off the NVFP4 files under `best`. A Best / Balanced / Fast menu sits beside
  the model in the Generate panel (writes the preference); the picker's Auto row follows.
  `generate.models` reports `quality`. Ten engine tests that pin the Lightning tier now set
  `balanced` in their session helpers.
- **Object Selection with SAM 3.1** (`object_select.rs`, `select_ml_cmds.rs` reworked):
  - **Object finder**: the pointer resting a quarter second over the picture runs `select.probe`
    (new: a `jobs::run` without a document lock, not journaled, no Generations row, quiet in the
    status bar; the request goes out at ≤ 0.6 MP) and the answer tints the object in the accent
    colour; the click applies the kept probe (`select.byPoint {probe}`) with no second detection
    while the picture and its history are unchanged (probes live in `Session::select_probes`, eight
    kept). The health check behind it is cached ten seconds **per server URL** (a first version
    cached per app and sent clicks to a dead server after the preference changed).
  - **Box prompt**: a drag is `select.byPoint {box}` through the new `sam3.1/segment-box` template
    (stock `PrimitiveBoundingBox` → `SAM3_Detect.bboxes`); `sam3.1/segment-point` gained
    `negative_coords`. Checked live before writing them: an empty `[]` list is accepted, a
    negative point trims the mask, a box around the whole fox finds it as the point does.
  - **Refine mode** (the options bar's New | Refine): a click adds a spot the last object must
    include, ⌥-click one it must not; the last query runs again with every point so far after
    `edit.undo` takes its result back, so one undo step stays. Ctrl was tried first and is the Move
    tool's override, hence a mode. **Refine edge** (on by default) runs the detector's hard mask
    through the classical edge refinement, as Remove Background does; Select › Subject and the
    Properties' Select Subject go through the detector when it answers.
  - Object Selection no longer snaps its presses (`snap_ui`): a prompt snapped to a selection edge
    landed off the object (a test caught it as a 20,20 click arriving as 23,13).
  - Automation: `select.byPoint {points, negativePoints, box, refine, probe}`, `select.probe`,
    `ui.inspect.objectSelect`, `ui.set {toolOptions}` (any field, spelled as inspect shows it),
    `ui.click` / `ui.move` modifiers, `ui.inspect.canvasRect` + `activeDocument` for document →
    screen mapping in scripts.
- **Live** (`docs/comfy/bench/scripts/object-select-live.ps1`, release GUI, SAM 3.1 warm): the
  finder's probe for the fox arrived 0.8 s after the pointer rested; the click applied it with no
  new detection (selection 75,123 742×878); Refine's ⌥-click on the tail re-ran with the negative
  point (344k → 289k pixels in a CLI check: the tail region went); a box around the head found a
  patchy face region (39 % of its box filled at any threshold; `refine_iterations` 0 gives 51 %,
  5 gives 13 %); Select › Subject ran `select.subjectML`. **Model limits seen**: SAM 3.1's box
  prompt is good at framing a whole object and poor at cutting a part out of one (the head of a
  fox), and a negative point on a whole limb trims rather than removes it; the Refine mode's
  islands in the screenshot are SAM's own low-confidence pixels, not the edge refinement.
- **Tests**: engine `select_ml_cmds_tests` (+2: box / negatives / refine; probe kept, applied,
  stale, unknown, nothing-found, downscaled) and the quality test (auto orders, Blackwell NVFP4
  under best vs balanced, `generate.models.quality`); UI `object_select_tests.rs` (3: box prompt,
  Refine mode and its undo depth, mode switching; the finder in a kittest harness with the inline
  fake; no server and the menu's Select Subject) and `retouch_ui`'s updated test. Gates: clippy,
  fmt, layers, wasm, parity, scorecard 57 of 156 unread (the new preference is read).
- **Select Subject, after the owner said it "sucks"**: a sweep of seven phrasings over seven
  pictures (through the CLI) showed "the main subject" finding the fox, a sideways face, a
  balloon and a lighthouse tower, while "subject" alone failed on three pictures with the
  server's raw `SaveImage: index 0 is out of bounds` (an empty mask batch), and "person" on a
  seascape returned an 83-pixel speck. So `select.subjectML` now asks in turn: "the main
  subject", "the foreground object", "person", "animal", "object" (one detector call each,
  ~0.5 s warm, stopping at the first hit); instances under 2 % of the largest are dropped unless
  one is asked for by number; the server's empty-batch error reads as "nothing matching"; and
  every Select Subject entry point (the Select menu, the Properties quick action, both options
  bars) goes through the detector with Refine edge. Before this build the menu and the Quick
  Selection bar still ran the classical `select.subject`, which on the lighthouse picture selects
  the whole lower half: most likely what the owner saw. The fake server gained `segments_for`
  (per-prompt answers) to test the chain.
- **Quick Selection through SAM 3.1** (owner: "make quick select use sam also"): a stroke of the
  Quick Selection tool, when the server answers, is one `select.byPoint` with at most 4 spots
  spread evenly along the stroke's samples (`retouch_ui::stroke_prompt`: whole pixels inside the
  canvas, no repeats), `mode` add (⌥: subtract, so the object under the stroke comes out) and
  `refine` = the bar's Enhance Edge; the classical `select.quick` stays for no server, a server
  that is down, or a stroke off the canvas. The options bar says "Paint over an object to select
  it" when a server is configured (×13). Tests: `quick_selection_strokes_ask_the_detector_when_a_server_answers`
  (fake server; no server / down server → `select.quick`) and `stroke_prompt_spreads_a_few_whole_pixel_spots_inside_the_canvas`.
  The cap was 16 first; the live script (`bench/scripts/quick-select-live.ps1`) showed a hatched,
  fragmented fox, and a CLI sweep of the same stroke with 1, 2, 4, 8, 16 spots gave 344 k, 335 k,
  337 k, 292 k and 164 k pixels: SAM 3.1's mask degrades as positive points pile up, so 4 it is
  (whole fox, 0.9 s warm; a second stroke adds, an ⌥ stroke over the tail subtracts it).
- **Open**: the sidecar (report step 2) is not started; SAM's box-on-a-part weakness could be
  helped by sending the box crop at full resolution; the finder could pre-warm the checkpoint on
  tool selection (the first probe after a cold server takes ~19 s, the model load).

## 2026-10-12 (night): plan Phase 5, the finish (hub, alerts, saving, Angles) and the walkthrough

Built [`plan-artcraft-experience.md`](plan-artcraft-experience.md) §7 and ran its §9 walkthrough.

- **Generate hub** (`generate_hub.rs`): eight cards (Create: Image; Edit: Fill, Edit, Expand,
  Remove Background, Split into Layers, Select by Text, Angles) under the Home screen's recent
  files and in **Help › Generative Features…** (`help.generativeFeatures`, first in Help). A card
  opens the Generate panel in its mode or runs its menu command; an Edit card with nothing open
  shows the open dialog and carries on once the picture is open (a cancel leaves nothing waiting).
  The Home card scrolls on a short window. Window › Generate now also focuses the prompt.
  `ui.inspect.hub` lists the cards with their drawn rects.
- **Alerts** (`alerts.rs`): `integrations.alerts` (`off` default, `done`, `doneAndFailed`) picks a
  synthesized blip in `generations_panel::on_event` (880 → 1320 Hz done, 220 Hz failed, no sound
  files). Playback is the `alerts` cargo feature (rodio 0.20.1, a new dependency; one blip at a time
  on its own thread). It is not a default feature, because CI builds the workspace on Linux without
  the ALSA headers, so "on in release" means the Windows and macOS packaging scripts build with
  `--features heif,photocraft/alerts`; the Linux tarball, Flatpak (offline vendored crates) and
  FreeBSD packages leave it off for now (ALSA would be a new build and runtime dependency there).
- **Saving results**: `integrations.saveResultsDir` (empty = ask) and `saveResultsNaming`
  (`{model}_{date}_{seed}`, validated: at least one token, no folders) are what `generate.export`
  uses without `dir` / `name`; automation may not set the folder (`workspace.rs`, however the path
  is written). Window › Generations gained a select mode (✓ in the header, or Select in a card's
  menu) with **Save selected**.
- **Generative Angles**: the research step found fal's Multiple Angles LoRA for
  Qwen-Image-Edit-2511, Apache-2.0, with the authors' own ComfyUI graph (stock nodes only, checked
  against our server's `/object_info`). Template `qwen-edit-2511/angles` (angles LoRA, then the
  4-step Lightning LoRA, ModelSamplingAuraFlow 3.1, CFGNorm, 4 steps CFG 1; page `angles` so no
  other picker lists it); `generate.angles {rotation, tilt, zoom, seed, variations}` builds the
  LoRA's `<sks> [azimuth] [elevation] [distance]` prompt and runs Generative Edit with the new
  `wholePicture` option (a selection does not mask a camera move); Edit › Generative Angles… opens
  `angles_ui.rs`: an orbit widget (the subject, the ground ring with 8 stops, the camera dragged
  around and up, snapping to the LoRA's 96 cameras), Direction / Height / Distance / Variations,
  Generate with its estimate, and a warning naming the LoRA when the server lacks it. **Not run
  live**: the LoRA is not on the dev server and the app downloads no weights; tested against the
  fake (the graph, the words, the layers in the Variations strip).
- **Live walkthrough** (`docs/comfy/bench/scripts/walkthrough-live.ps1`, release GUI with
  `alerts`, ComfyUI up), all ten steps of §9: Essentials shows both panels; Image × 4 with Krea 2
  Turbo into a new document in 17.5 s (estimate ~22 s), four tiles, a tile click and Keep leave one
  layer; Fill "a red kite" 15.7 s with the task bar away; Edit "make the sky stormy, keep the kite"
  with the kite layer as reference 18.5 s; a dead server URL gives a `serverUnreachable` row and
  Recreate succeeds; Place after close; the CLI's export writes
  `krea2-turbo-image_2026-10-12_061608_<seed>.png`; after a relaunch 4 completed and 1 failed rows
  with 4 thumbnails; the Home Image card and Help's Expand card set the panel's mode; Enter to
  generate, `lastAlert = done`, the strip at the bottom; CLI and GUI both list 6 records.
  Screenshots (and the Phase 3/4 ones) in `C:\Users\5090\Documents\PhotoshopEX-walkthrough-2026-10-12`.
- **Fixes on the way**: hub card sentences were clipped mid-line (now two lines with an ellipsis,
  cards 64 pt); "CREATE"/"EDIT" and Russian "Direction"/"Distance" were missing from catalogues
  (`every_tl_literal_is_translated` caught them).
- **Tests**: `generate_hub_tests.rs` (4), `angles_ui_tests.rs` (3), `alerts.rs` (2), the
  Generations select mode and alert choice (2), engine export preferences and Angles (graph,
  words, validation, 96 cameras), automation's save-folder refusal. Gates: clippy (also with
  `alerts`), fmt, layers, wasm, parity (the two menu items), scorecard 57 of 155 unread.
- **Open**: Angles needs a live run once the LoRA is installed (and its estimate a measurement);
  Linux packages without alert sounds; the hub's Edit cards cannot be driven by scripts with
  nothing open (they show the native open dialog).

## 2026-10-12 (evening): plan Phase 4, the Variations strip

Built [`plan-artcraft-experience.md`](plan-artcraft-experience.md) §6.

- **`variations_strip.rs`** (new): the active document's last five generative runs as 56-pt tiles
  at the canvas's right edge (`integrations.variationsStrip`: `right`, `bottom`, `off`; new
  preference, read by the strip). The plan suggested a new `app.ui.bundles` map fed by job
  events; the runs are instead read from the Generations library each frame (records whose `doc`
  is the active document, fill / edit / expand / image / similar, with the result layers that
  still exist). Runs started by a dialog, a script or the CLI-shared library show too, and undo,
  delete and close need no bookkeeping. A tile is the layer's pixels over the run's area (cached
  by pixel fingerprint, four builds per frame). Click shows it (`generate.variation`), × drops it,
  ✓ on the shown one keeps it, ✦ in the header runs Generate Similar for as many again; a running
  job shows placeholders with a sheen and its percent; a failed one a red tile with the reason,
  Retry (`generate.recreate`) and Dismiss.
- **Engine**: `generate.dropVariation {layers, index}` (deletes one; when it was on screen and the
  others are hidden the next one shows and becomes active) and `generate.keepVariation {layers,
  index}` (keeps one, deletes the others), one undo step each; the variation commands share their
  parsing and now refuse a layer listed twice.
- **Task bar**: the 1/n switcher, its `results` / `shown` state and `show_variation` are gone
  (the strip replaces them; `ui.inspect.generativeBar` no longer reports `results`).
- **No modal over the canvas**: a fill started from a dialog or a script showed the modal progress
  dialog in the middle of the canvas. It now stays away while the strip draws the job's
  placeholders, the task bar shows the job or the Generate panel shows its Cancel (that last case
  was a Phase 2 gap).
- **Automation**: `ui.inspect.variations` (placement, visibility, runs with status, layers,
  shown, failure, and the drawn rects of tiles, ×, ✓, ✦, placeholders, Retry, Dismiss).
- **Live** (`docs/comfy/bench/scripts/variations-live.ps1`, release GUI, ComfyUI up): a fill with
  four variations over open water ran in the background in 33.8 s; four placeholders and no modal
  while it ran, then four tiles each with a different red sailboat; `ui.click` on the third tile
  showed it in one undo step ("Variation 3"); ✓ left one layer (5 → 2 layers with the
  background). One run showed the tool switched from Brush to Path Selection after the click; it
  did not reproduce (the history showed only "Variation 3"), the window was on the desktop, and a
  whole-app test now checks that a tile click reaches nothing but the strip.
- **Tests**: `variations_strip_tests.rs` (5: tiles, click / × / ✓ each one step and undo; running
  placeholders then tiles, the modal exemption, a failed run's Retry and Dismiss; bottom and off;
  Similar adds a run, a text selection stays out, five runs at most; the whole-app click), engine
  `dropping_and_keeping_variations_are_one_undo_step_each`, the bar tests read the strip. Gates:
  clippy, fmt, layers, wasm, parity unchanged, scorecard 57 of 152 unread.
- **Open**: a run's thumbnails show the layer unmasked over its padded area (the request
  rectangle), not the masked composite; pending tiles of `generate.image` into a new document sit
  on the document it started from.

## 2026-10-12 (later): plan Phase 3, the model picker

Built [`plan-artcraft-experience.md`](plan-artcraft-experience.md) §5.

- **`model_picker.rs`** (new): one pill + popover for the Generate panel (`generate-panel`), the
  task bar (`generative-bar`) and the generated dialogs' `template` field (`flt-<command>`, rows
  carried as `__models` in the dialog's fields). Data-driven: callers pass `model_picker::Item`s
  built by `generate_panel::picker_items` from `generate.models`, so the plan's
  `pill(ui, app, page, …)` became `pill(ui, id, items, current, auto, group, width)`. Family glyphs
  from the plan's colour map; newest first by a numeric-aware compare; family submenus from eight
  rows when the new preference `integrations.groupModelsByFamily` (default on) allows; badges for
  the estimate, `research` (greyed, not selectable until `allowResearchModels`) and
  `not installed` (selectable). The Auto row names what it resolves to ("Uses: Qwen-Image-Edit-2511,
  Lightning 8 steps"; the preference's default first, as the engine does). Rows whose short names
  collide show full names (both Krea text-to-image templates shortened to "Krea 2 Turbo").
- **The probe moved**: Phase 3 first made loading the catalogue probe the server. The panel is on
  screen from startup, so every launch contacted ComfyUI and left "Checking generative models" in
  the status line; three UI tests caught it (a job they did not start). Now the catalogue loads
  offline, the bar probes when it opens (as before Phase 3), a picker probes when its popover
  first opens (`generate_panel::probe_once`), and a finished `generate.models` job stays out of the
  status line (`jobs_ui.rs`, one arm).
- **One-row form**: at 1440×900 the panel stays in its one-row form even with Generations hidden,
  which hid the picker, so that form now carries the pill as the glyph alone (30 pt).
- **Automation**: `ui.inspect.modelPickers` lists the pickers drawn with their rect, label and
  whether the popover is open; the pill and rows have accessibility labels ("Model: Auto").
- **Live** (`docs/comfy/bench/scripts/model-picker-live.ps1`, release GUI, ComfyUI up): each
  popover opened through `ui.click` on its inspected rect and screenshotted. These were checked:
  Fill lists the five fill templates with estimates (~6 s to ~54 s) and Qwen-Image-2.1 greyed
  `research`; Image marks the Krea style template `not installed` (its LoRA is absent); the task
  bar and the Generate Image dialog show the same component.
- **Tests**: `model_picker_tests.rs` (6: grouping threshold and preference, numeric newest-first,
  research gating and badge text, display names, every shipped family has a glyph, the panel's
  rows and the probe against the fake, clicking through the popover with kittest); the one-row
  panel test checks the glyph pill; `jobs_ui_tests` checks the quiet background check. Gates:
  clippy, fmt, layers, wasm, parity unchanged, scorecard 57 of 151 unread (the new preference is
  read). Workspace: the nine known environment failures plus the scorecard staleness test, which
  passes after regenerating `docs/scorecard.md`.
- **Open**: the dialogs show no estimates (their size lives in the dialog's own fields); the
  popover grows to the longest name (about 600 pt for the mask-guided fill).

## 2026-10-12: plan Phase 2, the Generate panel and reference images

Built [`plan-artcraft-experience.md`](plan-artcraft-experience.md) §4.

- **Template capabilities** (genai `template.rs`): `creator`, `description`, `badges`, `pages`,
  `references {max, min, placeholders, nodes}`, `sizes`, `counts`, `estimatedSeconds {base,
  perMegapixel}` on every shipped template (values from `benchmarks.md`), `Meta::estimate_seconds`.
  Reference placeholders a request leaves unbound are pruned with their nodes and the inputs that
  linked to them, so one graph serves zero to N references.
- **References**: `Request.references` (checked against min/max before any network call,
  uploaded, bound in order). The 2511 edit templates take two (`image2`/`image3`, wired as the
  official graph does). New templates, each checked against the live `/object_info`:
  `qwen-edit-2511/image-ref` (permissive: 1–3 references into an empty latent of the asked size),
  `krea2-turbo/image-style` (the official style-reference graph with Ostris's LoRA, Krea 2
  Community licence per its Hugging Face metadata; not run, the LoRA is not installed),
  `qwen-2.1/edit` (research opt-in, up to nine references). Engine: `references` on
  `generate.edit` / `generate.image` as `{"layer"}` / `{"path"}` / `{"job","index"}`
  (`generate_refs.rs`, ≤ 1 MP each), remembered on the layer for Generate Similar; `generate.image`
  switches Krea 2 to the reference template when references come with no template named, and
  gained `variations`. `generate.models {width, height, count}` reports capabilities and `eta`;
  `generate_cmds::estimate_seconds` / `likely_auto` are the shared formula. Automation refuses
  `{"path"}` references.
- **Window › Generate** (`generate_panel.rs`): mode strip, reference deck (+ menu: active layer,
  a layer, a file, a stored generation; drops onto the panel become references through a new
  `DropTarget::References`), the growing prompt with counter / larger editor / wand / negative
  disclosure, model / aspect / size / count / destination / expand pads, Clear all (confirmed with
  references), and **Generate · ~16 s** (Cancel with the percent while running). Ctrl+Enter
  generates; Enter too with the new preference `enterToGenerate`. One-row form under 150 pt.
- **The task bar is the panel's compact form**: prompt, per-mode template and count live in
  `UiState::generate`; `ui.set generative*` stay as aliases and `ui.inspect.generativeBar` still
  reports them. `showGenerativeBar` became `auto` / `always` / `never` (a saved boolean reads as
  `auto` / `never`); in `auto` the bar steps aside while the panel is on screen and Edit ›
  Generative Fill… focuses the panel's Fill mode.
- **Dock**: Generate sits above Generations. With both open Layers asks for 320 pt instead of 500,
  the other groups give way first, and the generative groups fold to their tab strips when Layers
  would drop under 260 pt. Measured: full prompt box on a 1250-pt window; one-row forms and five
  Layers rows at 900 pt (ten without them); folded and three rows at 720 pt (four without).
- **Live** (`docs/comfy/bench/scripts/generate-panel-live.ps1`, release GUI on the RTX 5090,
  through the panel's own Generate path): a 2511 edit with two layer references ("put the fox
  from image 2 on the rocks … holding the compass from image 3") placed both, 39.5 s including
  the model load; an image from one layer reference into a new document 12.1 s against 14.9 s
  estimated; a fill 9.3 s cold and 8.7 s warm against 8.7 s estimated (the plan asked for
  within 50 %). Then warm CLI runs measured what a reference costs (`benchmarks.md` §5g): 11.8 /
  19.9 / 30.2 s with 0 / 1 / 2 references, so `estimatedSeconds` gained `perReference` (9 s on
  the Lightning tier) and the Lightning-8 family was refitted to `7.5 s + 5.5 s/MP`: the edits now
  estimate 11.7 / 20.7 / 29.7 s, the reference image 11.3 s, the fill 8.6 s.
- **Tests**: genai 19 + 12 client (pruning, estimate, bad reference metadata, references bound in
  order and limits checked first); engine `generate_refs_tests.rs` (5: edit references and
  Similar, limits and malformed specs, the image template switch and document variations, file
  and stored-generation references, the one formula); UI `generate_panel_tests.rs` (6) and the
  updated bar tests; dock tests for the new budget. Workspace gates: clippy, fmt, layers, wasm,
  `panic_hunt`; parity 637 / 637; scorecard 57 of 150 unread (the new preference is read).
- **Open**: `qwen-2.1/edit` and the Krea style template are not run live (their per-reference
  costs are guesses); the model picker is still a dropdown (plan Phase 3); the first run after a
  model switch takes 20–30 s longer than the estimate, which knows only warm servers.

## 2026-10-11 (night): plan Phase 1, the Generations library and panel

Built [`plan-artcraft-experience.md`](plan-artcraft-experience.md) §3 after the owner answered
§0: panels on by default, a **500-result / 5 GB** cap, sounds behind a cargo feature (Phase 5).

- **Engine** (`crates/engine/src/generate_jobs.rs`, new): `GenerationStore` on `Session`
  (`session.generations`). The dispatcher records every generative command (fill, image, expand,
  edit, similar, split, remove background, the three SAM selections, recreate): a background job
  gets a running row at start and its outcome in `end_job`; an inline run (CLI, tests, MCP with
  `wait`) is recorded when it returns; refused requests leave no row. Result layers are rendered
  over their request rectangle with the mask as alpha into `results/<id>-<i>.png` plus a 128-px
  thumbnail, on a writer thread for background jobs. Failures get a kind from the message
  (server unreachable, model files missing, out of memory, server error, timeout, bad input,
  discarded, interrupted). One correction to the plan: job ids restart per process, so record
  ids are unix µs + pid, bumped past known ids (plan §0 notes it).
- **Persistence**: `<config dir>/generations/index.json`, atomic, one-second debounce from
  `poll_jobs`, flushed on drop; another process's rows are merged (ours win, removed rows stay
  removed), so `photocraft-cli run … --cmd generate.fill` shows up in the app's panel. The CLI
  attaches the folder only for runs with a generative command. Rows left running by a closed
  process become "interrupted"; a corrupt index is ignored.
- **Commands**: `generate.jobs`, `generate.jobs.dismiss` / `.pin` / `.clear`,
  `generate.recreate` (Similar while the layer exists, else the original command on its
  document, else a fill/edit becomes Generate Image of its prompt with `fallback: "image"`),
  `generate.place` (a stored result back as a layer at its place, or a new document),
  `generate.export` (name tokens `{model} {template} {date} {seed} {prompt} {n}`, never
  overwrites; denied to automation as an ambient path). `generate.info` gains `job`.
- **UI** (`generations_panel.rs`, new): `Group::Generations` below Properties, on in Essentials
  (a saved layout without the field gets it too), rail button `grid-2x2` with a running/unread
  badge, Window › Generations, `ui.inspect.generations`, `ui.set {generationsView}`. In progress
  rows (real progress, Cancel), Failed rows (kind label, message tooltip, Recreate, Dismiss),
  Completed grid or list (hover Place, Generate Similar, Save…, Copy prompt; context menu with
  Place as Layer, Open as Document, Pin/Unpin, Dismiss; click selects the layer or opens the
  result as a document); overflow menu with confirmed Clear completed / failed / all.
  Only visible lines are laid out; thumbnails decode lazily, six per frame. When the dock
  squeezes the group (any column where Layers needs the room, which on this 1250-pt window is
  already the case), it draws a **compact strip**: one row of square tiles (progress, a warning
  sign with the reason in the tooltip, thumbnails) with context menus and the overflow menu.
  36 strings ×13 languages.
- **Layout cost** (#147): Generations sits between Properties and Layers at a compact 80 pt, so
  the default workspace shows **8 Layers rows at 900 pt** (10 without it) and 2 at 720 pt (4).
  `dock_tests` checks upstream's guarantee with the group hidden and PhotoshopEX's with it.
- **Tests**: 15 engine tests (`generate_jobs_tests.rs`: lifecycle inline and background, failure
  kinds against the fake (down, OOM, validation error), restart, interrupted rows, caps and pins,
  dismiss/clear, place after close, recreate fallbacks, export naming, two-process merge,
  no-folder mode, bad params); 6 UI tests plus an ignored bench (`generations_panel_tests.rs`);
  the automation deny test. `panic_hunt` now clears the server URL in its fresh session: with
  ComfyUI running, `generate.splitLayers {}` and `select.subjectML {}` ran real jobs (>4 s).
  Workspace: 4347 passed, the same 9 environment failures as the baseline (GPU parity rounding,
  7 egui-wgpu canvas tests, the eyedropper test). Clippy, fmt, layers, wasm green; parity 636 /
  636 (the Window › Generations row); scorecard moved only by that row. rustfmt also reformatted
  a few lines in the earlier generative files (`generate_cmds.rs`, its tests, `fake.rs`,
  `generative_bar.rs`) and one clippy fix in `generate_enhance_tests.rs`.
- **Numbers** (dev profile, `bench_generations_panel_with_200_rows`): `panel()` with 200
  completed rows **0.13 ms** in the grid, **0.08 ms** in the list (1.45 ms before laying out
  only the visible lines; budget 1 ms); the index for 200 rows is 278 KB and writes in 2.5–3.0 ms.
  Live, release GUI through the control channel (`docs/comfy/bench/scripts/generations-live.ps1`,
  RTX 5090, ComfyUI warm): fill 13.7–14.7 s, edit 10.7–13.3 s, a 768×512 Krea 2 image
  5.6–7.7 s, each a background job that showed as running then completed with its thumbnail;
  Place after File › Close All opened the 464×416 fill result as a document; a dead server URL
  gave a `serverUnreachable` row; a restart kept all four rows. A process killed mid-job had
  already flushed its running row, which the next start marks interrupted.
- **Open**: an ETA on running rows waits for the template `estimatedSeconds` of Phase 2; the
  Save… button uses the platform save dialog until Phase 5's export preference; record ids are
  looked up by layer only within the session that made them (`generate.info.job`); a process
  that still holds a row another process cleared writes it back on its next flush (the CLI is
  short-lived, so in practice only two GUI instances could do this).

## 2026-10-11 (later): the build plan

[`plan-artcraft-experience.md`](plan-artcraft-experience.md): the hand-off plan the owner asked
for so another agent can build the study's §6. Five phases, each with the engine commands (params
doc strings), the data shapes (`GenerationRecord`, the library on disk under
`<config_dir>/generations/`, template capability flags, `Request.references`), the UI modules
and the exact upstream files to touch (dock groups, rail, menus, control protocol, with line
anchors at `ce5c692`), the tests per file, the docs per phase, performance budgets, a failure
taxonomy, the ETA formula, live scripts, an acceptance walkthrough and a file map. Open
decisions listed in §0 (panels on by default, a 200-result / 2 GB library cap, sounds behind a
cargo feature). Nothing built yet.

## 2026-10-11: the ArtCraft study, refocused on the experience

The owner clarified: not the business model, the **UI and experience**, with the credits removed
(local compute). [`artcraft-study.md`](artcraft-study.md) was rewritten from the frontend
source: the one-window shell with Home + app tabs and the Task Queue button; the Create pages'
editorial empty state, feed and floating prompt box; the prompt box piece by piece (reference
deck, growing prompt with counter and fullscreen, model pill and family-grouped picker with wait
badges, aspect/resolution/quality/count pickers that follow the model's capabilities, Generate);
pending tiles, failed rows and completed cards with hover actions (Recreate, Make Video, Share,
Download); the Task Queue popover; the lightbox's prompt panel; the Library's folders, tags and
drag-to-canvas; the 2D editor's side toolbar, inpaint toolbar, History Stack of candidate bundles
and prompt editor; Angles' orbit sphere; settings (Enter to generate, sounds, themes, keybinds,
download naming). Then the PhotoshopEX translation: a **Generate panel**, a **Generations
panel** (feed + queue with real progress and time estimates), a **Variations strip** over the
candidate layers, a Generate hub, sounds and preferences, with wireframes and a five-step build
order. Write-up only.

## 2026-10-10: the ArtCraft study (write-up only)

The owner asked to look into ArtCraft's paid features and to make the implementation "more
like how they do things overall", write-up first. [`artcraft-study.md`](artcraft-study.md) is
that write-up, from getartcraft.com, the team's posts and a local read of the public
`storytold/artcraft` repository (a shallow clone in the temp directory, not kept in the tree).

Short version: ArtCraft's only paid feature is generation through its own hosted provider,
priced in credits (1 credit = $0.01; Basic $8 for 1,000, Pro $28 for 3,750, Max $48 for
6,600; refunded on failure); everything else is free, and users can bypass credits with their
own Fal/Replicate/Google keys or Midjourney/Grok logins. The patterns worth taking: a model
catalogue as data with capability flags that drive the UI, one request shape with the provider
chosen late and a priority/fallback list, a persistent local task queue with typed statuses and
failure reasons plus a Task Queue popover, a prompt box built from capabilities with a reference
deck and the cost (for us: the time) on the Generate button, and a history stack of candidates.
The study ends with an ordered adoption list (catalogue → task queue → prompt panel →
providers with BYO keys, the ArtCraft Omni API among them → failure labels → a gallery) and
three questions for the owner (cloud providers at all; queue or providers first; asking the
ArtCraft team for API access). Nothing in the app changed.

## 2026-10-09 (late night): the PhotoshopEX branch

The owner branched the fork into their own version: this checkout,
`C:\Users\5090\Documents\PhotoshopEX`, branch `photoshopex`, from `comfy-photocraft` at `9cd7165`
(the enhancer, the NVFP4 policy and the Object Selection tool's SAM mode included). Landed here:

- **The app icon**: a Photoshop-style tile (navy `#001E36`, the letters "Px" in `#31A8FF`, Segoe
  UI Bold) replaces the PhotoCraft kitsune everywhere the app and its packaging read an icon
  (`assets/app-icon/`: the hicolor PNGs 16–512, the 1024 macOS render, the `.ico` packed with
  `cargo xtask ico`, the `.icns` written by Pillow, the SVG masters). `packaging/icon-px.py`
  regenerates them (Pillow from the ComfyUI portable Python; the Rust toolchain packs the .ico).
  Adobe's own "Ps" mark is a trademark and is not copied; the letters are this version's.
  `app_icon.rs`'s tests (every shell size in the .ico, the window icon decodes) pass.
- **The session's dev files**: the PowerShell bench and verification scripts in
  `docs/comfy/bench/scripts/` (with a README), the bench images in `docs/comfy/bench/images/`
  (git-ignored, 171 MB).
- `CLAUDE.md` says what this checkout is; `ATTRIBUTION.md` and `brand.rs` name the new icon.

Open here: the product name (the window title and `--help` still say PhotoCraft; "Photoshop" is
Adobe's trademark, so the published name is the owner's call), and everything in the entries
below.

## 2026-10-15: local-build look and feel (wheel zoom, no Discord button, Ps tile)

The user's three asks for the local build. Hotkeys were already Photoshop's (`shortcuts.rs`,
no overrides in `preferences.json`); the one that felt off was the wheel: Photoshop's default
is wheel = scroll, Alt + wheel = zoom, which upstream copies, and the user wants wheel = zoom.

**Landed**

- `General::zoom_with_scroll_wheel` defaults to `true` in the fork (`prefs.rs`); the saved
  `preferences.json` on this PC was flipped by hand too, since serde writes every field and a
  stored `false` would have won over the new default. Alt + wheel now scrolls (`wheel_nav.rs`
  swaps the two; untouched).
- The title bar's Discord button is gone (`panels.rs`: the block and `DISCORD_ROOM`); Help ›
  Discord stays, so `links.rs` and its tests are untouched. The `titlebar.rs` width test now
  asserts the button is absent at every width; `docs/ui-design.md` says the same.
- A Photoshop-style "Ps" tile (`assets/app-icon/ps/`, README there) in place of the kitsune in
  the title bar (`brand.rs`), the window/taskbar icon (`app_icon.rs`, non-macOS) and the exe
  resource (`build.rs`). New files, three path edits; the kitsune assets are untouched for
  merges. Personal build only: it must never ship (`upstream.md` brand checklist).

**Open**

- The fork is 1,728 commits behind upstream (2026-10-15), which closes ~1,200 issues a week,
  many of them parity fixes. A merge is the next big lever on "feels unlike Photoshop".

## 2026-10-09 (night): the prompt enhancer

The user asked for "some kind of prompt optimization" after the doodle-to-food session, and
whether the settings are the best ones. Research first (`benchmarks.md` §5e): the 2511 and
Krea 2 graphs already match ComfyUI's shipped templates node for node; what the Qwen authors
recommend is a VLM rewrite of the instruction with the picture in view (their
`polish_edit_prompt`), which ComfyUI's own Qwen-Image-2.1 template does with `TextGenerate`.

**Landed**

- `generate.enhancePrompt {prompt, task: fill|edit|image, useImage, seed}` → `{prompt, enhanced}`:
  the Krea 2 text encoder (Qwen3-VL 4B, Apache-2.0, already installed for Krea 2) run as a
  vision-language model through `TextGenerate` (templates `enhance/edit` with the picture,
  `enhance/image` without; `Task::Text`, `Response.texts`, `PreviewAny` text outputs in
  `comfy.rs`). The engine's rules are the chat's system turn, the user's words the user turn;
  a fill is told where its selection sits ("the upper-left part"). Two tries with consecutive
  seeds; the answer is cleaned of a stray "assistant", quotes and markdown; nothing usable
  leaves the prompt as typed. `enhance: true` on `generate.fill`, `edit` and `image` does it
  on the way (an expand keeps its wording); Preferences › AI Integrations › **Enhance prompts**
  makes it the default; the layer remembers both prompts (`GenerativeInfo.enhanced`,
  `generate.info`), and Generate Similar rewrites anew. The task bar has a wand button by the
  prompt that replaces the text with the rewrite (job, "Enhancing…", the modal dialog stays
  away). Translated ×13 (4 strings). Tests: 6 engine (`generate_enhance_tests.rs`, the fake
  answers `Enhanced: ` + the user turn or `Options.enhanced_text`), the bar test.
- **The chat template.** Half the first live rewrites came back empty: `TextGenerate`'s
  `thinking: false` appends an empty `<think>` block that the non-thinking 4B Instruct model
  answers with an immediate end of turn (or a restarted "assistant" turn). With `thinking:
  true` the template ends at the plain assistant turn: 26 of 26 clean. Kept in the template
  notes so nobody "fixes" it back.
- **The doodle.** With the rules' first wording the rewriter asked to keep the doodle's outline
  and 2511 drew bread again; three hand-written sentences showed the model needs "a photograph
  of a real human face … replacing the black outline strokes with real skin, hair and facial
  features", which now is the rule for a change of medium, and the rewriter writes exactly that
  (`bench/doodle-sentence-A*.png`: a clean photographic face).

**Numbers.** A rewrite is 0.8–1.3 s (sentence) / 1.8–2 s (paragraph) after a 3 s first load.
End to end through the CLI, both of the user's doodle prompts give a photographic human face
(18.8 s and 17.2 s with the rewrite, `bench/enhance-doodle-*-final.png`). The 4B encoder stays
resident (5 GB); with 2511 alone the steady state is 13.5–14.6 s an edit. Answer to the
user's VRAM question in `models.md` §4: Qwen-Image-2.1 int8 + its 8B encoder + SAM 3.1 ≈ 18
GB, they fit together.

**NVFP4 (same session).** Comfy-Org's official repacks of the 2511 encoder (5.7 GB) and Krea 2
Turbo (7.2 GB) measured against fp8 (`benchmarks.md` §5f): the encoder's precision does not
change an edit's time (14.1 s either way) but saves 3 GB; Krea 2 is 18 % faster (5.7–6.0 s
against 7.0–7.1 s a 1024² image) and leaves 5 GB more free; the pictures are alike. Shipped
as a policy: template slots carry an `nvfp4` alternative, Preferences › AI Integrations ›
**Model precision** (`auto` | `default` | `nvfp4`, `auto` by default) loads it on a Blackwell
GPU (`is_blackwell` on the device name) when the server has it; `run_switching` computes the
model set from the prepared request so the purge sees a precision change; `generate.models`
reports `device`, `blackwell`, `nvfp4`, `nvfp4Installed`; tests in
`generate_precision_tests.rs` (the fake takes `Options.device`). Live: the server loaded both
NVFP4 files with nothing but the defaults. Lesson from the first benchmark attempt: ComfyUI
caches a repeated graph; vary the seed or the "warm" run takes 290 ms.

**The SAM 3.1 UI (same session).** The Object Selection tool: with a generative server
configured, a click (no drag) runs `select.byPoint` on the point under the pointer (⇧ adds, ⌥
subtracts, as the rectangle does), the options bar has a **Select by text** field (Enter →
`select.byText`, replacing the selection; `toolOptions.selectText`) and says "Click an object,
or drag a rectangle around it", and **Select Subject** goes through `select.subjectML`, falling
back to the classical `select.subject` when no server is configured or the command cannot
start. A drag stays the classical `select.object`. Test in `retouch_ui.rs` against the fake.
Translated ×13 (4 strings, the two job labels included).

**Open.** The GUI is the 8:56 PM build until the user closes it (the enhancer, the NVFP4
policy, the SAM UI and the temp-file model memory are in the CLI build only); SAM box prompts
(a drag through the detector) and hover highlighting are the next steps of the tool.

## 2026-10-09 (evening): first user session, two fixes

The user tried the app on a line drawing of a face: Generative Fill over a rectangle with
"make this hyper realistic", then "turn this drawing into a hyper realistic portrait that looks
similar". Both results were a plate of food; the second run took 229 s.

**What happened** (from the server's history, `/history`, and the uploads): the fills were
correct inpaints (the output outside the mask matched the input to within 1/765), but Fill
regenerates the selection from scratch with the picture as a reference, and "make this hyper
realistic" over a white rectangle with a doodle gave the model nothing to hold on to: it drew
a dish. The instruction was an *edit* of what is there, which is Generative Edit's job (the
whole picture by instruction, no noise mask). The 229 s: the server still held the 19 GB
layered model from the Split tests in another process, so the app's first run (which never
purged: the model-set memory is per process) loaded 2511 partially and streamed weights.

**Landed**

- `run_switching` purges on a process's first run too when the server reports less than a
  quarter of its VRAM free (one `/system_stats` call); the fake takes `Options.vram_free` and
  a test checks the purge precedes the first prompt and happens once.
- The task bar has a **Fill | Edit** mode (`generativeMode` over `ui.set`): Edit runs
  `generate.edit` with the same prompt, template picker (edit templates) and variations, the
  selection masking the result; and a Fill prompt that reads like an instruction to change
  what is there ("make this…", "turn it into…") shows "Changing what is there? Switch to Edit".

**Follow-up, the same evening.** Generative Edit with the user's own prompts still gave food
("turn this drawing into a hyper realistic portrait…", "make this hyper realistic", on
Lightning and on the 40-step base): the model does not read the doodle as a face, "this" is a
plate to it. Prompts that **name what the drawing shows** work: "turn this sketch of a face
into a realistic photo of a human face" and "redraw this simple line drawing of a smiling face
as a photorealistic portrait photo of a person with the same round face, two eyes, a nose and
a smile" both gave a face (`bench/user-drawing-*.png`). Noted for the UI copy: describe the
subject, not "this".

The slow runs: a fresh load of 2511 at 1088×704 ran three times at 14–17 s under logging
(ComfyUI 0.39 stages models for dynamic VRAM loading; no partial loads in the log), so the
209 s and 229 s runs were contention: the Split tests' 19 GB layered model in one case, and
in the other the server being used from its own web UI at the same time (its log shows Krea 2
attempts with the text encoder loaded as `qwen_image`, which fails: Krea 2 needs CLIPLoader
type `krea2`, which PhotoCraft's template uses; our Krea 2 run took 9.5 s). Landed on top:
the model-set memory behind `run_switching` is mirrored to a file in the temp directory, so
the app, the CLI and MCP agree on what a server holds across processes and only a real
change purges; the low-VRAM rule now applies only to a server no PhotoCraft process has
recorded; `generate.free` forgets the record.

## 2026-10-09 (late): Split into Layers (Qwen-Image-Layered), and a softer detector matte

**Landed**

- `generate.splitLayers` (Edit › Split into Layers (Generative)…, translated ×13, generated
  dialog): the composite (or the active layer) goes to **Qwen-Image-Layered** through
  `qwen-layered/split`, ComfyUI's official "Image to Layers" subgraph as a template (the
  picture as a `ReferenceLatent` on both conditionings, `EmptyQwenImageLayeredLatentImage` with
  `layers` slices, `LatentCut` dropping slice 0 and `LatentCutToBatch` making a batch, the
  layered VAE decoding RGBA). Each result becomes a new layer above the active one, background
  first, with the alpha the model gave it; `layers` 1..8 (default 3); the picture goes out at
  640 px on its longer side (the official size) and comes back resampled. A new `Task::Layers`;
  the fake answers such graphs with one banded RGBA image per layer. The model is Apache-2.0
  (`models.md`); the 19 GB fp8 repack and its VAE were downloaded here for testing only.
- Remove Background's detector route now softens its hard mask with the classical edge
  refinement (`refine: true` by default, the Quick Action's parameters), so hair and soft
  edges get partial coverage from the pixels themselves.
- Tests: two for the split (layer order, alpha bands, names, graph, sizes, validation, no
  "similar"), the refinement in the detector test.

**Live** (the 1024² lighthouse, `bench/split-layer-*.png`): three layers in **72 s** (model
load included) at 640²: layer 0 a near-white base carrying the boat's shadow, layer 1 the sky
and sea with the boat (soft where the rocks were), layer 2 the rocks and lighthouse cut out
with real soft alpha. The model chooses the decomposition; the composite is the picture. A
second run reuses the loaded model.

**Open**: the layered model's 640 px working size (a 1024 run is worth a try), transparent
fills (the model's alpha times the selection), preview frames.

## 2026-10-09 (late): Remove Background without the opt-in, and Select by Point

**Landed**

- `generate.removeBackground` works out of the box: `template` defaults to `auto`, which is
  the Qwen-Image-2.1 matte (soft alpha, research licence) when the preference allows it and
  the server has the files, else **SAM 3.1** (`sam3.1/segment`, SAM License, no opt-in): the
  prompt or "the main subject" (the phrase `select.subjectML` uses) goes to the detector and
  the union of its instance masks becomes the layer mask or the selection, hard-edged. The
  route is decided in the job (`resolve_auto` over the matte template with the detector as
  the fallback); a named template of another task is refused; `generate.models` reports
  `autoMatte`. The classical Quick Action `layer.removeBackground` remains the no-server path.
- `select.byPoint {"x","y"}` (Phase 3.5): a new template `sam3.1/segment-point` prompts
  `SAM3_Detect` with `positive_coords` (a literal JSON list in the API format, probed live:
  the boat under one point in 2 s) and no text; the engine scales the canvas point to the
  request's pixels. No menu row yet (it needs a click); scripts and MCP can use it, a tool
  modifier later.
- Tests: the detector route (prompt, no prompt, as selection, the fallback when the 2.1 files
  are missing, `autoMatte` either way), by point (coordinates sent, validation).

**Open**: a soft-matte refinement for the detector route (its masks are hard); box prompts;
preview frames; Qwen-Image-Layered.

## 2026-10-09 (late): the leftovers of Phases 2 and 3

**Landed**

- Generative layer metadata (Phase 2 item 5): every layer a generative command makes carries a
  `GenerativeInfo` (command, prompt as typed, resolved template, seed, steps, guidance, edge,
  request rect, name; images add size and transparent) as JSON in a PSD additional-layer-info
  block keyed `cpGn`, so upstream's `Layer` is untouched and the record survives PSD round
  trips and `Layer::duplicate`. `generate.info` reads it.
- `generate.similar` (Edit › Generate Similar, translated ×13, enabled on a generative layer):
  runs what made the layer again with a new seed in the same place, the layer's mask as the
  area (no selection needed), the result a layer above it. Fills and edits re-run as they were
  (the same resolved template); an expanded area is filled again with the expand's prompt
  through the fill templates (its canvas already exists); a generated image is generated
  again as a layer. Research gate as usual.
- The picker's installed badges: `generate.models` takes `async: true` and probes the server
  on a worker (`jobs::run`, result = the usual answer); the bar starts it when it loads its
  templates, keeps the job id, and marks templates whose files the server lacks "(not
  installed)" (translated). The bar never waits for the server.
- The Crop tool's expand state (Phase 3): a Generative Expand checkbox in the Crop options bar
  (`ToolOptions::crop_generative`); on ↵ with a frame beyond the canvas, `commit_crop` crops to
  the part inside first (when that is not the whole canvas) and then runs `generate.expand`
  with the overhang as pads, so the model paints the new area; an empty prompt continues the
  scene. Off, the canvas just grows as before.
- The control-protocol script test of the Phase 2 DoD already existed as
  `generative_bar_tests::the_menu_opens_the_bar_and_generate_runs_the_fill_with_variations`
  (select → `ui.set` → generate → `ui.inspect` against the fake); the roadmap now says so.
- Tests: two engine tests (metadata + similar for fills, duplication; images and expands), the
  crop test (beyond, partly inside, option off), the bar test checks the badges.

**Open**: preview frames in the bar; a permissive matting default; Qwen-Image-Layered.

## 2026-10-09 (late, Phase 3): Generative Edit, and the automatic purge on a model switch

**Landed**

- `generate.edit` (Edit › Generative Edit…, translated ×13, generated dialog): the whole
  composite goes to the official 2511 edit graph as the model is meant to be used, no noise
  mask, the instruction decides what changes (`qwen-edit-2511/edit-lightning-8` and `edit`,
  `auto` between them, `AUTO_EDIT_ORDER`, `generate.models` reports `autoEdit`). The result
  is a layer above the active one, masked to the selection when there is one (soft dithered
  edge or hard), so "turn the red boat blue" with a marquee on the boat changes the boat and
  nothing else on the canvas. Descriptions are wrapped as "Change this image so that it shows
  {prompt}, keeping everything else exactly as it is."; instructions get ". Keep everything
  else exactly as it is." Shares `fill_region` (variations, sizing, Lanczos, timings); the
  request carries no mask when the template takes none.
- `run_switching`: every generative run (fill, expand, edit, image, matte, select by text)
  goes through it. The engine remembers the model files (`folder/file`) of the last run per
  server URL (a process-wide static) and asks the server to unload its models before a run
  whose files differ: the Lightning run after the 40-step base went from 121 s to 20.2 s
  (`benchmarks.md` §5c). Never on a server's first run; a server that cannot purge runs as
  before. A test drives Lightning → base → base → Lightning against the fake and counts the
  `POST /free`s (1, then 2) and their position.
- Corrected a diagnosis: the 40-step base's 130–148 s is its own cost (two passes a step at
  CFG 4), not thrash; the thrash was the Lightning tier after it.
- Four engine tests for edit (whole canvas, no mask upload, wrapping; the selection confines
  the result; `auto` falls back to the base, validation, `autoEdit`), one for the purge.

**Live** (`bench/edit-*.png`): a dark storm with lightning and the lamp lit at 16 s; the boat
turned blue inside its selection and nothing else; a sunny day at 20 s after a switch.

**Open**: the Crop tool's expand state; a permissive matting default; Qwen-Image-Layered;
a 20-step middle tier for the base.

## 2026-10-09 (late, Phase 3): Remove Background on Qwen-Image-2.1's alpha channel

The user: "qwen2.1 supports alpha channels". It does, natively: the 2.1 VAE decodes RGBA.
Probed live before writing code (`bench/probe-rgba-s7.png`, `probe-t2i-alpha-s3.png`):
ComfyUI's official background-removal recipe returned a colour-type-6 PNG with a soft matte in
18 s, and a plain text-to-image prompt ending in "isolated on a transparent background, output
a PNG image" returned a boat with a clean matte in 10 s.

**Landed**

- `generate.removeBackground` (Edit › Remove Background (Generative)…, translated ×13, generated
  dialog): the layer (or the composite with `sampleAllLayers`) goes to `qwen-2.1/matte`, the
  official recipe as a template (the 2.1 edit graph, the picture as image_1 at its own size,
  "Remove the background, and output a PNG image"; `prompt` names what to keep and becomes
  "Remove the background, keeping only {prompt}, …"). The result's **alpha** comes back as the
  layer's mask (the Background becomes a normal layer, like the Quick Action) or, with
  `asSelection`, as the selection combined by `mode`. The pixels the model re-rendered are
  never used; the layer keeps its own. Requests go out at ≤ 1 MP with sides in multiples of 32
  (what `TextEncodeQwenImage21` rounds to), the matte comes back through the tent filter, the
  layer's own transparency goes to the model over mid-grey and multiplies the matte. One undo
  step; "kept nothing" is an error that changes nothing. Research licence gate as for the other
  2.1 templates.
- `generate.image` takes `"transparent": true`: the template's new `promptFormatTransparent`
  wraps the prompt (only `qwen-2.1/image` has one; others refuse with a clear error) and the
  layer keeps the alpha the PNG comes back with.
- Fake server: `Options.matte` gives generated images an alpha rectangle. Five engine tests
  (mask from alpha with pixels untouched and undo, subject wrapping + Background → layer,
  as-selection + add mode, validation + licence gate + "kept nothing", transparent image +
  refusal). Docs: models.md, architecture.md, roadmap; parity/scorecard regenerated.

**Live** (the 1024² lighthouse with the boat, `bench/matte-*.png`, `transparent-compass.png`):
16.8 s per matte (25 steps, 2.1 int8). The model's own reading kept the boat solid and the
lighthouse as a ghost; "the red boat" gave a perfect boat cutout (147×46 px); "the lighthouse"
kept the lighthouse with its rocks, the white tower semi-transparent against the bright sky
(a model quirk to note in the UI later: name the subject, re-roll, or refine the mask). A
transparent "vintage brass compass, product photo" came back in 10.8 s as a clean cutout with a
real soft alpha.

**Open**: a permissive matting model (BiRefNet) as the default so the command works without
the research opt-in; Qwen-Image-Layered ("image to layers": ComfyUI 0.39 ships
`EmptyQwenImageLayeredLatentImage` and official templates; needs `qwen_image_layered_bf16` +
its VAE, not installed) as a Layers › "Split into Layers" feature; transparent fills (the
model's alpha times the selection) for adding cut-out objects.

## 2026-10-09 (late): soft, dithered edges for fills and expands

The user: "the borders are harsh, is there a way to do like a noise opacity in the edges so
that there aren't hard cutoffs?" Yes; it took three changes, each found by measuring the
right-side expand's sky profile (`benchmarks.md` §3, "Edges").

**Landed**

- `generate.fill` and `generate.expand` take `"edge":"soft|hard"` (default soft; a choice in
  the generated dialogs, labels already translated). Soft: the result layer's mask is the
  feathered request band itself, its ramp **dithered** with per-pixel noise (`dither_edge`: a
  splitmix64 hash of pixel index and seed, amplitude `0.6 × (1 − |2m − 1|)`, so nothing changes
  inside or outside the band and the grain peaks mid-ramp). The result fades into its
  surroundings as grain, not a gradient or a line. Hard: exactly the selection (or the added
  canvas). The feather is now 4 % of the longer side for both (fill 6–48 px, was 2 %; expand
  8–80 px, was 8 %: at 8 % the model moved the horizon inside the band at one seed, a ghosted
  double horizon; at 2 % the old edge still read as a line).
- `feather_outward` starts its ramp at **full** coverage on the edge and falls to nothing at
  twice the radius (the GrowMask + FeatherMask shape). The old `max(selection, blur)` ramp
  started at half coverage: a 0.5 step at the edge in both the noise mask and the layer mask,
  which is half the tone difference left standing as a line.
- Expand's padding is pre-filled by replicating the picture's edge pixels instead of mid-grey:
  the grey never reached the model as content (cropped reference, full noise mask) but the VAE
  carried its tone into the latents of the edge band, a dark line right on the old edge
  (luminance 213 between 225 and 228). `PREFILL` grey remains only for an empty picture.
- Tests: the soft-edge test (full on the edge pixel, high just outside, falling off, grain in
  the row, hard = selection, `edge: fuzzy` rejected), `dither_edge` unit test, the prefill test
  now checks edge continuity, expectations for the new radii and ramp.

**Live** (`bench/edge-s11-*.png`, `bench/expand-right-{soft,hard}.png`, `xab*-*.png`): the
expand's sky goes 229 → 226 across the band with no step or dip, nothing visible in a
contrast-stretched crop; the hard variant keeps its step. The soft and hard variants of one
seed are the same request, so the second returns from ComfyUI's cache in 0.1 s. The three
placement cases (boat ×2, balloon) are unchanged by the wider feather. Back-to-back expands
run 13–20 s (13 s with the reference's encoding cached, 19 s after another reference evicted
it).

**Found on the way** (`benchmarks.md` §3): what no edge can hide is the new side being a
different photograph. With a described prompt the Lightning model painted a crisp, sharp-
horizon sea next to the hazy original at two seeds; the empty prompt (the default instruction)
continued the haze; the 40-step base (130 s) and several wordings copied the lighthouse into
the new area at one seed, and so did the empty prompt at the same seed when re-run later. The
server is bit-deterministic in a steady state and not across model-load states (pixel diffs
0.00 on repeats, up to 21/765 between states: a second lighthouse against none), so two-seed
wording comparisons prove little; the wrappers stay as they are and the dialog's note that an
empty prompt continues the scene is the advice. Also: the first run after a graph switch
(base ↔ Lightning, or purge → fill → expand) took 31–131 s with the server at 2.4 GB free
(partial model load), then 13–20 s; the engine should purge before a run that changes the
model set (roadmap, next).

**Open**: the automatic purge on a model-set change; a prompt study for Expand over more seeds;
the Crop tool's expand state; `generate.edit`, `generate.removeBackground`.

## 2026-10-09 (night, Phase 3): Generative Expand

**Landed**

- `generate.expand` (Edit › Generative Expand…, translated ×13, generated dialog): adds canvas
  (`left/top/right/bottom` px, or a larger `width/height` with Canvas Size's `anchor`) and has
  the model paint it, as **one undo step**: `translate_doc` + the new size inside the job, the
  picture moves by the left/top pads, the selection and vector geometry move along. The request
  is the bounding box of the added area (an L or a frame when several sides grow) plus the fill
  margin of the picture. An empty prompt becomes "extend the scene beyond its original edges,
  continuing it naturally". Result adds `canvas` and `offset`.
- Its own templates, `qwen-edit-2511/expand-lightning-8` and `expand` (task `expand`,
  `AUTO_EXPAND_ORDER`, `generate.models` reports `autoExpand`): the fill graph plus an
  `ImageCrop` so the text encoder's reference is **the picture alone** while the padded canvas is
  the sampling latent under the noise mask. Found the hard way: shown the padded canvas as its
  reference, the edit model reproduced the padding as content (replicated edge pixels came back
  as streaks; mid-grey came back as a grey frame at one seed). The empty canvas is still painted
  mid-grey before upload (`prefill_outside`), and the result layer's mask is the soft request
  mask with an 8 % feather into the picture (`outpaint_feather_radius`), so the re-rendered band
  carries the new area's tone across the old edge instead of meeting it at a line.
- The fill job body is now `fill_region` (shared by fill and expand): Lightning tiers, `auto`,
  grid alignment, the request cap and 512 px floor, the feathered request mask, variations. The
  cap went from 1 MP to **0.75 MP**: with 2511 fp8 and its 7.9 GB text encoder resident on the
  32 GB card, a 1 MP latent plus the 1 MP reference pushed ComfyUI into offloading part of the
  model (a 45 s run instead of 18 s).
- Tests: two engine tests (one-step L-shaped expand with pixel/mask/undo checks, the grey
  pre-fill and the crop rectangle; target size + anchor, selection follows, validation, disabled
  without a document).

**Live** (the 1024² lighthouse, Lightning 8, `bench/expand-*.png`): widened to the right by
384 px with "more open sea and evening sky" in 20.2 s (a 736×1024 request), and framed by 192 px
on three sides with the default prompt in 14.8 s (a 1408×1216 canvas sent at 944×816). Both
continue rocks, sea and the pink evening sky coherently; the old edges are faint at most. The
earlier variants (padded reference: streaks, a grey frame, a blue sky) are kept next to them with
suffixes for comparison.

## 2026-10-09 (performance pass): Lightning tiers, `auto`, request sizing, server flags

**Landed**

- `qwen-edit-2511/fill` now matches ComfyUI's official 2511 graph exactly (CFGNorm after
  ModelSamplingAuraFlow, `FluxKontextMultiReferenceLatentMethod index_timestep_zero` on both
  conditionings). Two Lightning tiers, `fill-lightning-8` and `fill-lightning-4` (Apache-2.0
  LoRAs by lightx2v, `LoraLoaderModelOnly` after CFGNorm, CFG 1), and the preference default
  `defaultFillTemplate = auto`: the job asks the server which files it has (`/models/<folder>`)
  and takes the first of `AUTO_FILL_ORDER` (`fill-lightning-8`, then the 40-step base) whose
  files are all installed. `generate.models` reports `autoFill`; the task bar's picker starts
  with Auto. BYOB stays: the 850 MB LoRA is a download the user makes.
- Request sizing: a fill request over 1 MP is sent downscaled (Lanczos-3 pixels, tent-filtered
  mask, multiples of 16) and the result is resampled back under the full-resolution layer mask;
  `resize_rgba8` is Lanczos-3 everywhere (was bilinear); segmentation requests are capped at 2 MP
  and their masks come back through the tent filter. `requestWidth/Height` and per-variation
  `timings` (`encodeMs`, `uploadMs`, `queueMs`, `runMs`, `downloadMs`) are in the result.
- Uploads: PNG at the fastest deflate level (loopback, read once) and **content-addressed
  names** (FNV-1a of the bytes), so ComfyUI's node cache keeps the loader, the 7 B vision text
  encoder and the VAE encode across variations and re-rolls of one selection.
- `docs/comfy/bench/bench-fill.ps1` (the live benchmark), `docs/comfy/benchmarks.md` (numbers),
  `start-comfyui-fast.ps1` on the dev PC with `--fast fp8_matrix_mult --highvram`, the flag table
  and a 24 GB / RTX 3090 section in `comfyui-setup.md`, Lightning facts in `models.md`.
- Tests: auto resolution with and without the LoRA on the fake (`Options.missing_files`),
  the 1 MP cap end to end (upload size, placement, timings), resampler behaviour, the 2 MP
  segmentation cap; 33 engine generate/select tests, genai 29.

- Prompt wrapping for the 2511 fills: a description ("a red boat") becomes "Add {prompt} to
  this image, fitting the scene's perspective, lighting and surroundings naturally. Change
  nothing else."; a prompt that already opens with an imperative verb (`template::is_imperative`,
  "remove the car") becomes "{prompt}. Fit the result…" (`promptFormatImperative`, also on the
  2.1 fill). The request mask is **feathered outward** (2 % of the longer side, 4–24 px, two box
  blurs then `max` with the original) so the latent noise mask blends the edge; the layer mask
  stays the selection. Opt-in `qwen-edit-2511/fill-guided` passes the mask as a second
  reference image for the best placement at 1.5–3× the time.

- **Native-size sampling** (second pass, same day): the 2511 templates VAE-encode the uploaded
  image instead of the Kontext-scaled one, so the sampling latent has the request's own size
  while the text encoder keeps its 1 MP reference. The engine grows the request rectangle to
  the 16-px grid (`align_to_grid`), caps it at 1 MP and floors it at 512 px on the longer side
  (`request_size`). 35–45 % faster and better placed (benchmarks §4). `generate.free`
  (Edit › Purge › Generative Models) asks the server to unload its models after a measured
  memory-pressure state made fills three times slower (benchmarks §5).

**Measured** (all tables and the method in `benchmarks.md`; the lighthouse, a 576×512 request):

| Fill, warm | default flags, unique uploads | `fp8_matrix_mult`, cached encoder (re-roll) |
|---|---|---|
| 2511 base, 40 steps CFG 4 | 162 s | 153 s |
| 2511 Lightning 8 | 18.6 s | 16.0 s |
| 2511 Lightning 4 | 12–16 s | 8.1 s |
| Qwen-Image-2.1, 25 steps | 4.5 s | 2.6 s |

A 2511 fill is sampling-bound (~2 s per Lightning step at 1 MP, ~4 s per base step with CFG 4);
the client's share is 2 ms encode, 5 ms upload, under 0.4 s download. With native-size sampling
the three A/B cases went from 16.2–16.5 s to 9.2–11.2 s, and a 120×90 selection fills in 10 s.

**Findings**

- Quality, three cases at fixed seeds (benchmarks §3): the plain prompt with a hard mask gave a
  cut-off boat with a rectangular seam at one seed; the shipped wrapper + feather gave no seam
  in any case and the boat on the water at seed 11; placement at seed 7 stays the model's
  choice (rocks) unless `fill-guided` or Qwen-Image-2.1 is used, both of which put it on the water.
- `--highvram` is harmful with 2511 fp8 on 32 GB (2 GB free, LoRA reload thrashed); dynamic
  VRAM already keeps a family resident. lightx2v's `fp8_e4m3fn_scaled` 2511 file ran 35 % faster
  but returned noise in the masked area with our graph; removed. `fp16_accumulation` does
  nothing for bf16/fp8/int8 models. Shipped flag: `--fast fp8_matrix_mult` (~5 %).
- Two benchmark runs overlapped on the GPU once (a background task kept running after I thought
  it had died); the numbers above come from clean, sequential foreground runs.
- PowerShell: variable names are case-insensitive (`$seed` is `$Seed`), `.NET`'s working
  directory is not PowerShell's (`[IO.File]` needs absolute paths), and `"$f:"` is a drive
  reference; all three bit the benchmark scripts once.

## 2026-10-09 (late night): Phase 2, the generative task bar and variations

**Landed**

- `generate.fill` takes `variations` (1..4): one backend run per variation with consecutive
  seeds, each its own masked layer, only the first visible, all in one undo step; the result
  lists `layers` and `seeds`, and the job's progress spans the runs ("Variation 2/3").
  `generate.variation {layers, index}` shows one of them, hides the others and activates it, as
  one undo step. `generate.models {probe: false}` lists the templates without touching the server.
- `crates/ui-egui/src/generative_bar.rs`: the task bar under the selection. Prompt (Enter
  generates), template picker (research templates only with the preference on), variations,
  Generate; a progress bar with Cancel while a fill runs on the document (its own or one a script
  started); a ‹ 1/3 › switcher after a multi-variation run; × hides it until the next selection.
  It sits centred below the selection, above it when there is no room, always on the canvas.
  Edit › Generative Fill… and the selection context menu open it; the generated dialog remains
  the fallback when the new `integrations.showGenerativeBar` preference is off or for scripts
  that pass params. `ui.set` gained `generativeBar`, `generativePrompt`, `generativeTemplate`,
  `generativeVariations`; `ui.inspect` reports `generativeBar`; the modal progress dialog stays
  away while the bar shows the job (`docs/control-protocol.md` updated).
- Tests: two engine tests (variations, the variation command and its validation), four UI tests
  (menu → bar → generate → switch against the fake server; the preference's dialog fallback; the
  bar's placement in a kittest harness; the placement arithmetic); six strings and one
  preference label translated in the 13 languages.
- Live: a two-variation Qwen-Image-2.1 fill from the bar on the lighthouse (a boat on the water),
  captured as `ui-genbar-idle/running/results.png` in `C:\Users\5090\ComfyUI\photocraft-tests`.

**Findings for the performance pass (next)**

- ComfyUI's official 2511 template chains UNETLoader → ModelSamplingAuraFlow 3.1 → CFGNorm 1.0 →
  (Lightning LoRA, 4 steps, CFG 1 | nothing, 40 steps, CFG 4) → KSampler euler/simple, with
  `FluxKontextMultiReferenceLatentMethod index_timestep_zero` on both conditionings. Our 2511
  fill lacks CFGNorm and the reference-method nodes. The 2511 Lightning LoRAs (lightx2v) are
  Apache-2.0: 850 MB bf16 files for 4 and 8 steps.
- Qwen-Image-2.1's few-step LoRA (Viggle turbo, 6 steps) carries the same research licence as
  the base model and needs the author's custom nodes (a sigma schedule and an unmerged LoRA
  loader); not a plain `LoraLoaderModelOnly`. `QwenImage21Cache` (already in our templates) is
  the other 2.1 speed lever.
- ComfyUI flags worth measuring on the 5090: `--fast fp16_accumulation` (ships in Comfy's own
  bat file; changes outputs slightly), `--highvram` (keeps models resident). SageAttention needs
  a wheel matching torch 2.14 / cu130 / Python 3.13 and is reported to break Qwen when enabled
  globally; left out.
- Photoshop's Generative Fill renders at most 1024 px on the longer side and upsamples; our
  2511 path does the same through `FluxKontextImageScale`, the 2.1 path keeps the input size.

**Still open (Phase 2)**: generative layer metadata, a committed control-protocol script test,
installed-model badges in the picker.

## 2026-10-09 (night): Phase 2, first slice: the generative commands are clickable

**Landed**

- Menu rows (fork additions in `menu_catalog.rs`, not parity rows): **Edit › Generative Fill…**
  and **Edit › Generate Image…** right after Content-Aware Fill, **Select › Select by Text…** (the
  command's own `menu`), and **Generative Fill…** first in the selection context menu
  (`canvas_tool_menu.rs`), where Photoshop puts it. Translated in the 13 complete-menu languages;
  `docs/parity.md` now counts 629 live items (the two Edit rows).
- Generated dialogs: the three commands are on `filter_dialog::has_dialog`'s allow-list (no
  preview). Their params docs were rewritten in the dialog notation, with 0 meaning "the
  template's default" for steps/guidance and "the document's size" for width/height, and
  `target: "auto|layer|document"`; the engine accepts those values from the CLI too (tests
  updated). OK runs the usual background job (`jobs_ui`: status-bar progress, Esc cancels), so
  the dialog itself needed no new UI code.
- Two fixes in upstream's `filter_dialog.rs`, both candidates for an upstream PR: `parse_spec`
  parses only the parameter object (a `:` in the notes after `→` leaked in as an `Int` field,
  visible as a stray `Ms"} (prompt` row), and ranges spanning ≤ 10 are stored with the two
  decimals the field shows (margin 0.25 was snapped to 0.3). Unit test added.
- Preferences › AI Integrations had been failing two ui-egui tests since Phase 1: the generated
  labels of the seven settings had no translations, and a test assumed the section was entirely
  unimplemented. Labels in 13 languages, `humanize` overrides ("ComfyUI server", "Generative
  timeout (seconds)"), the test now asserts the section is visible while agent control stays
  hidden.
- Visual check with the offscreen snapshot example, captures in
  `C:\Users\5090\ComfyUI\photocraft-tests\ui-*.png`: the three dialogs, the Edit and Select menus,
  the selection context menu, the Preferences page in English and German.

**Findings**

- The snapshot example panics on this PC's DX12 adapter inside egui-wgpu's staging belt
  (`renderer.rs:984`, the same failure as the 7 baseline canvas tests). `$env:WGPU_BACKEND =
  "vulkan"` works, and so does `WGPU_FORCE_FALLBACK_ADAPTER=1`; written into
  `dev-environment-windows.md` with the PowerShell quoting for `--script`.
- `input_tests::eyedropper_and_alt_sampling_show_a_pipette` fails the same way with
  `PHOTOCRAFT_CONFIG_DIR` pointing at an empty folder, so it is not reading machine preferences.
  Still unexplained, still in the baseline.
- The generated dialog is a stopgap: "0" for width/height/steps cannot explain itself, and there
  is no model picker. The task bar (Phase 2 item 2) replaces it.

**Numbers**: ui-egui lib 916 passed / 1 baseline failure; engine 859 + the generate/select
command tests; genai 29; i18n 1467/1467 in every language; parity 629/629; scorecard unchanged
(57 unread settings of 146); clippy, layers, wasm green.

**Still open (Phase 2)**: the task bar anchored to the selection, variations, progress preview,
a model picker fed by `generate.models`, generative layer metadata, and a committed
control-protocol script that selects, fills and screenshots.

## 2026-10-09 (later): Phase 3.5, Select by Text with SAM 3.1, live

**Landed**

- Template `sam3.1/segment` (community licence: the SAM License) from ComfyUI's official
  "Image Segment (SAM3)" template: `CheckpointLoaderSimple` (MODEL + CLIP from the 1.75 GB
  `sam3.1_multiplex_fp16.safetensors`), `CLIPTextEncode` on that CLIP, `SAM3_Detect`
  (threshold, 2 refinement passes, `individual_masks` on), `MaskToImage` → `SaveImage`, so the
  server answers with one PNG per instance. The server's `object_info` confirmed the node set.
- `Request.params` (template-specific placeholder values such as `threshold`) and
  `Error::NoOutput` (a detector that finds nothing) in the genai crate; the fake server answers
  `SAM3_Detect` graphs with one mask per configured rectangle.
- `crates/engine/src/select_ml_cmds.rs`: `select.byText` (phrase → instances → coverage →
  `sel::combine` with `replace|add|subtract|intersect`, `instance` picks one, `threshold`,
  `sampleAllLayers`; a background job and one undo step; reports each instance's bounds and
  pixel count) and `select.subjectML` (`what`: subject, person, face, hair, sky, animal, vehicle,
  text → a fixed phrase). Shared plumbing with `generate_cmds` (`plan_common`, backend, job
  progress, resampling), now `pub(crate)`.
- 7 command tests and 3 crate tests against the fake; clippy, wasm gate green.

**Live, on the 1024² lighthouse from the morning** (`photocraft-cli run … --cmd select.byText`):

| Phrase | Found | Bounds `[x, y, w, h]` | Pixels | Time |
|---|---|---|---|---|
| "the lighthouse" | 1 | `[695, 322, 111, 287]` (the tower, exactly) | 17 555 | 4.6 s, cold checkpoint load |
| "sky" | 1 | `[0, 0, 1024, 612]` (down to the horizon) | 601 550 | 1.8 s, warm |

Then `generate.fill` with `qwen-2.1/fill` on the lighthouse selection ("bold red and white
horizontal stripes", margin 0.4 → a 341×517 request) took 8.2 s and repainted only the tower
(`sam-lighthouse.psd/.png` in `C:\Users\5090\ComfyUI\photocraft-tests`). Select → describe →
generate, with no lasso, is now a two-command script.

**Still open**

- Point and box prompts (`select.byPoint`, the Object Selection tool's ML mode): `SAM3_Detect`
  takes `positive_coords`/`negative_coords` strings and `bboxes`; the coordinate string format
  needs reading from the node source before wiring.
- `panic_hunt` on a machine with ComfyUI running at the default URL will make real calls for
  `select.subjectML {}` (every other command fails validation first); run the hunt with the
  server stopped, as CI effectively does.
- Soft mattes for hair/glass (BiRefNet refinement) remain Phase 3.5's second half.

## 2026-10-09: first live runs against a real ComfyUI (Phase 1 DoD met)

**Setup on the dev PC** (nothing of this is in the repo; the app ships no weights, BYO models):
ComfyUI v0.39.0 Windows portable (NVIDIA build, 2.0 GB) at `C:\Users\5090\ComfyUI\ComfyUI_windows_portable`,
embedded Python 3.13.14, torch 2.14.0+cu130, RTX 5090 recognised. Start script
`C:\Users\5090\ComfyUI\start-comfyui.ps1` (loopback :8188, `--preview-method auto`). Model files
(Hugging Face, ~100 MB/s): Qwen-Image-2.1 int8 set 16.1 GB, Krea 2 Turbo fp8 set 17.4 GB,
Qwen-Image-Edit-2511 fp8mixed 19.1 GB + its 8.7 GB encoder (VAE shared with Krea 2). The server
listed the files without a restart.

**Findings that changed code**

- `TextEncodeQwenImage21`'s `latent` output is an **empty** latent sized to image 1, not an
  encoding of it (checked in `comfy_extras/nodes_qwen.py`): 2.1 edits in context and regenerates
  the whole picture, so `SetLatentNoiseMask` cannot inpaint with it. `qwen-2.1/fill` now sends
  the selection mask as a second reference image and wraps the prompt in a local-edit instruction
  (`meta.promptFormat`, new); PhotoCraft's layer mask confines the visible change. Result: correct.
- Grouped ("autogrow") inputs are spelled `images.image_1` in the API format (the server prefixes
  the group id; the first attempt with `image_1` failed with "unexpected keyword argument").
- `/object_info` on 0.39.0 has every class the four templates use; `CLIPLoader` offers both
  `qwen_image` and `krea2`.

**Numbers** (1024² text-to-image; fill = 384×320 selection, 576×512 request; `ms` is the
command's own timing, which includes upload, queue, sampling and download):

| Run | Template | Time | Notes |
|---|---|---|---|
| Text to image, cold | `qwen-2.1/image` | 28.4 s | model + encoder load from disk included |
| Text to image, warm | `qwen-2.1/image` | 10.8 s | 25 steps |
| Generative Fill, warm | `qwen-2.1/fill` | 4.7 s | boat placed exactly in the selection, seamless |
| Text to image, cold | `krea2-turbo/image` | 9.8 s | 8 steps; includes the 12 GB load |
| Generative Fill, cold | `qwen-edit-2511/fill` | 124.2 s | 40 steps + 19 GB load; see result note below |

Result note: the selection covered water on its left and rocks on its right. 2.1 (mask as a
reference image + instruction) placed a small boat on the water, matching the scene's light and
scale, with no visible seam. 2511 (latent noise mask) painted a larger boat on the rocks, inside
the selection but less plausible; worth retrying with the prompt describing the whole scene and
with the official template's `CFGNorm` node. Both results stayed inside the mask, as designed.
Krea 2 Turbo's text-to-image (8 steps) produced a different, photographic composition at the
same seed; Qwen-Image-2.1's fox test image is near photo quality.

VRAM after the 2.1 runs: 20.6 GB in use of 32 GB (ComfyUI keeps models resident); after all
three families had run: 21 GB (it swaps between them).

Outputs in `C:\Users\5090\ComfyUI\photocraft-tests\` (`q21-image.png`, `q21-fill.psd/.png`,
`krea2-image.png`, `q2511-fill.psd/.png`, `q21-fox.png`); the PSDs carry the masked
"Generative Fill: …" layer above the Background. Headless invocation, from the repo root:

```powershell
.\target\release\photocraft-cli.exe --% run in.png --cmd prefs.set --params "{\"values\":{\"integrations.allowResearchModels\":true}}" --cmd select.rect --params "{\"x\":320,\"y\":560,\"width\":384,\"height\":320}" --cmd generate.fill --params "{\"prompt\":\"a small red wooden rowing boat floating on the water\",\"template\":\"qwen-2.1/fill\",\"seed\":7}" --out out.psd
```

(PowerShell: one `--%` command per line; everything after the token is passed verbatim.)

**Still open**

- Judge the 2511 fill quality against 2.1 on more images; consider the official template's
  `CFGNorm` node and the Lightning 4-step LoRA for 2511.
- Variations, the task bar and menu rows (Phase 2); `generate.edit` / `expand` /
  `removeBackground` (Phase 3); `select.byText` with SAM 3.1 (Phase 3.5).
- The ComfyUI server was left running after the session (`Stop-Process -Name python` or close it
  from the start script's window to free 21 GB of VRAM).

## 2026-10-08 (later): Phase 1, the generative backend and `generate.fill`

**Landed**

- `crates/genai` (`photocraft-genai`, L4, registered in `xtask/src/layers.rs`): the
  `GenerativeBackend` trait (`health`, `run`, `model_files`), `Request`/`Response` over RGBA8 +
  Gray8 pixels, `Progress` for job progress and cancellation, `template` (API-format graphs with
  typed `{{placeholders}}`; built-in `qwen-edit-2511/fill` from ComfyUI's official Qwen-Image-Edit
  2511 template plus `ImageToMask` → `SetLatentNoiseMask` for inpainting), `comfy` (a blocking
  client on ureq 3.4 + tungstenite 0.30, no TLS: `/system_stats` version check → `/upload/image`
  ×2 → `/ws` → `/prompt` → socket progress + `/history` polling → `/view`; cancel = `/interrupt` +
  queue delete; deadline from the preference), `png` (via photocraft-codecs), and `fake` (feature
  `fake-server`: an in-process ComfyUI stand-in with HTTP + WebSocket, knobs for delay, failure,
  rejection, missing nodes, no-socket).
- `crates/engine/src/generate_cmds.rs`: `generate.fill` (validated params → composite crop with a
  25 % context margin + selection coverage → backend → result resampled if needed → **new raster
  layer in the document's own depth, above the active layer, with a layer mask equal to the
  selection**; one undo step; background job with progress and Esc-cancel), `generate.health`,
  `generate.models`. Commands have `menu: &[]` until Phase 2 (so parity and i18n are untouched).
- `generate.image` (pulled forward from Phase 3 on request): text to image with the built-in
  `krea2-turbo/image` template (from ComfyUI's official `image_krea2_turbo_t2i.json`: UNETLoader,
  CLIPLoader type `krea2`, CLIPTextEncode, ConditioningZeroOut negative, EmptyLatentImage,
  KSampler euler/simple 8 steps CFG 1, VAEDecode). `target: "layer"` covers the whole canvas
  (size defaults to the document, rounded down to the 16-px grid, min 64, resampled back),
  `target: "document"` opens a new RGB 8-bit document of the result's size (default 1024²).
  Krea 2 is a community-licence model: the catalogue says so and nothing pre-selects it for fill.
- Qwen-Image-2.1 templates (`qwen-2.1/fill`, `qwen-2.1/image`), from ComfyUI's official 2.1
  image-edit and text-to-image templates: `UNETLoader → QwenImage21Cache`, `CLIPLoader` type
  `qwen_image`, `TextEncodeQwenImage21` (positive, negative and the encoded latent of `image_1`,
  resolution 0 keeps the input size), KSampler euler/simple 25 steps CFG 1, the 2.1-specific VAE
  `qwen_image_2.1_vae_bf16`. Fill masks the encoder's latent with `SetLatentNoiseMask`. Both are
  **research-licence** templates: refused with a message until `allowResearchModels` is on, and
  listed as `allowed: false` by `generate.models` meanwhile.
- Preferences › AI Integrations: `comfyServer`, `defaultEditModel`, `defaultGenerateModel`,
  `defaultFillTemplate` (`qwen-edit-2511/fill`), `defaultImageTemplate` (`krea2-turbo/image`),
  `generativeTimeoutSecs`, `allowResearchModels`, each read by the commands and validated. Setting
  `defaultFillTemplate = qwen-2.1/fill` with the research opt-in makes 2.1 the everyday editor.
- Tests: 25 in the genai crate (unit + the client against the fake server: upload/queue/wait/
  download, polling without a socket, overrides, server failure, rejected prompt, cancellation
  interrupts within 3 s, deadline, unreachable server, old server version, health/model files,
  request validation) and 18 engine command tests (fill: masked layer geometry and pixels, undo,
  the uploaded crop and mask, background job parity with inline, cancel leaves the document
  alone, server failure, unreachable server, health/models, parameter validation, enablement,
  model overrides, a 16-bit document, resampling; image: canvas layer, new document, size
  rounding and validation, background job, template/task checks, model preference).
  `cargo test -p photocraft-engine --test prefs_usage` green.
- Gates on 2026-10-08: `cargo fmt`, `cargo clippy --all-targets -- -D warnings` on both crates,
  `cargo xtask layers` (29 crates, no violations), `cargo xtask wasm` (genai and engine check for
  wasm32; the client is native-only and seeds fall back to a counter there), `cargo xtask parity`
  (unchanged: 627/627, the commands have no menu rows yet), `cargo xtask scorecard` (57 unread
  settings of 144; the five new preferences are read), `panic_hunt --ignored` green in 25 s with
  the new commands, release `photocraft-cli` built (1 min 43 s); `photocraft-cli commands --filter
  generate` lists the four commands. Headless `generate.health` / `generate.fill` without a server
  return "cannot reach the ComfyUI server at http://127.0.0.1:8188 … start ComfyUI or change the
  server URL in Preferences › AI Integrations".

**Still open**

- A live run against a real ComfyUI (not installed on this PC yet) with the 2511 files; the
  template's node names come from the official template but have not executed here. First live
  run should also decide whether to add back `CFGNorm`.
- Phase 2: menu rows + translations, the generative task bar, variations, previews.
- `docs/comfy/models.toml` catalogue and `generate.image` / `generate.edit` / `removeBackground`
  (Phase 3) reuse the same crate; `select.byText` (Phase 3.5) needs a SAM 3.1 template.

## 2026-10-08: Phase 0, foundation

**Landed**

- Cloned `storytold/photocraft` (`main` @ `d836e34`, 642 commits, 49 MB) to
  `C:\Users\5090\Projects\comfy-photocraft`; branch `comfy-photocraft` created; local git identity
  set to the owner's email.
- Toolchain on the Windows PC: `scoop install rustup gh` → rustup 1.29.1, Rust stable 1.99.0
  (MSVC, using the pre-installed VS 2019 Build Tools), gh 2.102.0. No Python/Node on the machine
  (not needed for Rust; ComfyUI brings its own Python).
- Release build: `cargo build --release -p photocraft` **2 min 39 s** cold; `photocraft.exe`
  58.5 MB; `--version` → `photocraft 0.5.0 (dev build)`. `cargo build --release -p photocraft-cli`
  2 min 10 s (incremental on the same deps).
- Test baseline, `cargo test --workspace --no-fail-fast` (debug profile, 138 s after the build):
  **171 test binaries, 4227 passed, 9 failed, 32 ignored.** All failures are environment-related
  on this RTX 5090 / driver 617.42 / wgpu DX12 machine, none in engine or format code:
  - `photocraft-gpu --test parity adjustment_layers`: "Hue/Saturation Normal: max diff 1.00/255
    at (18,23)", i.e. exactly at the tolerance edge (GPU rounding differs from CI's software
    adapter).
  - 7 egui-wgpu canvas tests (`canvas_16f` ×2, `color_managed_canvas` ×2, `drag_preview_canvas`
    ×2, `live_stroke_canvas` ×1) panic inside `egui-wgpu 0.36.2 renderer.rs:984` "Failed to create
    staging buffer for index data" (a wgpu staging-belt issue on this adapter, not PhotoCraft code).
  - `photocraft-ui-egui --lib input_tests::eyedropper_and_alt_sampling_show_a_pipette`:
    "Precise keeps the crosshair" (unexplained; may read the machine's real preferences; retest
    with `PHOTOCRAFT_CONFIG_DIR` pointing at an empty folder).
  Everything else (engine 831 tests, ui-egui lib 915, io, psd, codecs, compose, automation…) is green.
- Research (web, dated 2026-10-08) on Krea 2 (open weights 2026-06-22, Krea 2 Community License,
  ComfyUI ≥ 0.26), Qwen-Image-2.1 (2026-09-20, **research-only licence**), Qwen-Image-Edit-2511
  (Apache-2.0), Z-Image Turbo (Apache-2.0), FLUX.2 variants, the ComfyUI HTTP/WebSocket API,
  Photoshop 2026 generative feature list, Krita's ComfyUI plugin, Rust ComfyUI crates.
  Finding worth repeating: **"Krea 2 Edit" is a community LoRA, not an official Krea model.**
- Codebase analysis delegated to Codex (`codex exec`, read-only, gpt-5.5) and verified by hand;
  key facts: `CommandSpec` fn-pointer registry, `jobs::edit_job` background jobs with progress
  and cancel, `ml` reserved at L4 in `xtask/src/layers.rs`, no HTTP client in the tree, an
  existing `integrations` preference section with an "AI Integrations…" menu entry.
- Documentation set in `docs/comfy/` (README, codebase-orientation, architecture, models,
  comfyui-setup, roadmap, dev-environment-windows, upstream, this log) and a root `CLAUDE.md`;
  a fork notice in the top-level README.

- `ai-integration-proposal.md`: the full catalogue of ComfyUI + local-LLM integrations (17
  generative, 9 perception, 8 language items, 8 infrastructure items) with a recommended order;
  adds "Select by text" and the in-app Assistant as new phases 3.5 and 4.5.
- SAM 3.1 researched and written in: Meta, 2026-03-27, text/point/box prompts, native in ComfyUI
  (PR #13408, one 1.75 GB checkpoint), SAM License (commercial OK, military/weapons banned, licence
  passed on). It is the model behind Phase 3.5 (`select.byText`, `select.subjectML`,
  `select.byPoint`); design sketch in the proposal §2.1.

**Numbers to carry forward**

| Metric | Value | How |
|---|---|---|
| Release build, cold | 159 s | `cargo build --release -p photocraft` |
| Test suite | 4227 pass / 9 fail / 32 ignored, 171 binaries | `cargo test --workspace --no-fail-fast` |
| Upstream menu parity | 627/627 | `docs/parity.md` |
| Generative commands | 0 | — |

- Published: fork **https://github.com/CodyChuGit/comfy-photocraft** created with `gh repo fork`,
  `comfy-photocraft` pushed (`5aed6c2`), `upstream` → storytold/photocraft. The push first hung on
  scoop git's `helper-selector` credential GUI; the repo now uses gh as its only credential
  helper (`dev-environment-windows.md` › Remotes and pushing).

**Still open**

- Install ComfyUI (portable build) and the Phase 1 model files; record versions and VRAM here.
- Decide the published product name and the brand-asset removal before any public build
  (`docs/comfy/upstream.md`).
- Check whether the 7 egui-wgpu staging-buffer failures and the 1/255 parity edge are known
  upstream (search issues for "staging buffer" / "parity 1.00/255"); if not, file them with the
  adapter/driver details above.
- Phase 1 starts with `crates/genai` (see `docs/comfy/roadmap.md`).
