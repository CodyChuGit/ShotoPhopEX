# CLAUDE.md: PhotoshopEX (branch `photoshopex` of Comfy PhotoCraft)

**This checkout (`C:\Users\5090\Documents\PhotoshopEX`) is the owner's own version**, branched on
2026-10-09 from the Comfy PhotoCraft fork at commit `9cd7165`: "my version of Photoshop that I
wanted from the start". Branch `photoshopex`; `origin` is the same GitHub fork
(CodyChuGit/comfy-photocraft), `upstream` is storytold/photocraft. What differs from
`comfy-photocraft` so far: the app icon (a Photoshop-style navy "Px" tile, `assets/app-icon/`,
`packaging/icon-px.py`), the session's bench scripts and images under `docs/comfy/bench/`
(`scripts/` tracked, `images/` ignored). Everything below still applies; the fork's rules and
docs are the base this version builds on. Before publishing a build, note that "Photoshop" and
the "Ps" tile are Adobe's trademarks: the icon here uses its own letters, and the product name is
the owner's call.

The base: the **Comfy PhotoCraft** fork of [storytold/photocraft](https://github.com/storytold/photocraft):
PhotoCraft (a clean-room, pure-Rust Photoshop reimplementation) plus a fully local, open-source
generative toolset driven by a ComfyUI server (Krea 2, Qwen-Image-Edit, Qwen-Image-2.1, Z-Image…).
Integration branch: `comfy-photocraft` (in `C:\Users\5090\Projects\comfy-photocraft`). Upstream
remains the source of truth for everything that is not generative.

## Read first, in this order

1. [`AGENTS.md`](AGENTS.md): upstream's rules. They all apply here: never crash (no `unwrap`/
   `expect`/`panic`/`unsafe`), everything is a command, no format or colour assumptions,
   clean-room, tests are the gate, thin data-driven UI, never break wasm, performance is a feature.
2. [`docs/comfy/README.md`](docs/comfy/README.md): the fork's vision, principles and document index.
3. [`docs/comfy/codebase-orientation.md`](docs/comfy/codebase-orientation.md): the subsystems a
   generative feature touches (commands, pixels, jobs, prefs, UI, automation, tests), cited to code.
4. [`docs/comfy/architecture.md`](docs/comfy/architecture.md) and
   [`docs/comfy/roadmap.md`](docs/comfy/roadmap.md): what we are building and in which order.
5. [`docs/comfy/devlog.md`](docs/comfy/devlog.md): what the last session did and left open.
   **Append an entry when you finish a task** (upstream keeps its dev log in the gitignored
   `log/`; the fork keeps this one in the tree so it survives machines and agents).

Upstream's `docs/architecture.md`, `docs/development.md`, `docs/contributing.md`,
`docs/control-protocol.md`, `docs/plugins.md` and the `book/` are the detailed references.

## Working on this machine (Windows)

See [`docs/comfy/dev-environment-windows.md`](docs/comfy/dev-environment-windows.md). Short form:

```powershell
# a shell opened before 2026-10-08 needs cargo on PATH:
$env:PATH = "C:\Users\5090\scoop\persist\rustup\.cargo\bin;$env:PATH"
cargo build --release -p photocraft           # ~2.5 min cold; target\release\photocraft.exe
cargo build --release -p photocraft-cli
cargo test --workspace --no-fail-fast         # baseline in devlog.md: 9 GPU-driver-related failures on this PC
cargo xtask layers; cargo xtask wasm; cargo xtask parity; cargo xtask scorecard --check
```

`photocraft.exe --help` opens the GUI (not a CLI flag); stop it with `Stop-Process -Name photocraft`.
Use a separate `CARGO_TARGET_DIR` per parallel agent. ComfyUI is a separate process on
`127.0.0.1:8188`. **The app starts it itself** when Preferences › AI Integrations › ComfyUI folder
names the install (`C:\Users\5090\ComfyUI\ComfyUI_windows_portable` here; `generate.engine.*`,
`photocraft_genai::sidecar`, the status bar's engine dot; the native engine is being built to
[`docs/comfy/plan-native-engine.md`](docs/comfy/plan-native-engine.md), Windows and Linux first).
By hand: `powershell -File C:\Users\5090\ComfyUI\start-comfyui-fast.ps1`
(the flags measured in [`docs/comfy/benchmarks.md`](docs/comfy/benchmarks.md)); setup and API in
[`docs/comfy/comfyui-setup.md`](docs/comfy/comfyui-setup.md). Live timings come from
`docs/comfy/bench/bench-fill.ps1`.

The native engine lives in a second repository, **`C:\Users\5090\Documents\genai-native`**
(github.com/CodyChuGit/genai-native, private; harness, loader, worker): `crates/genai` depends on
its `gn-engine` crate by git rev, fetched with the system git (`.cargo/config.toml`); after
pushing there, bump the rev in `crates/genai/Cargo.toml`. `photocraft --engine` /
`photocraft-cli --engine` run this executable as the engine worker; Preferences › AI
Integrations › Engine = Native routes generative commands to it.

## Conventions specific to the fork

- Generative code lives in `crates/genai` (L4, register in `xtask/src/layers.rs`) and
  `crates/engine/src/generate_cmds.rs`; UI in `crates/ui-egui`. Shared upstream files
  (`commands.rs`, `menus.rs`, `state.rs`, `menu_catalog.rs`) get surgical one-line edits so
  upstream merges stay clean.
- Commands are `generate.<verb>` (pixels) and `select.<verb>` for model-backed selection
  (`crates/engine/src/select_ml_cmds.rs`); preferences live under the existing `integrations`
  section (Edit › Preferences › AI Integrations…); every preference is read by code in the same
  change. Workflow templates are JSON files in `crates/genai/workflows/`, taken from ComfyUI's
  official templates and verified against a live server's `/object_info` before they ship.
- Model and licence facts go in [`docs/comfy/models.md`](docs/comfy/models.md) with a source and a
  date; research-only models are never defaults in a build for others. PhotoshopEX (this
  owner's version) is the exception since 2026-10-28, on the owner's call: engine `native` and
  "Allow Research-Only Models" on by default, so Qwen-Image-2.1 and Krea 2 serve everything
  with no ComfyUI. Turn both back before publishing (the checklist in `upstream.md`).
- Keep the fork MIT OR Apache-2.0; ship no model weights. Before any published build, follow the
  brand checklist in [`docs/comfy/upstream.md`](docs/comfy/upstream.md).
- Tests for anything that talks to ComfyUI run against an in-process fake server; one `#[ignore]`
  live test may use `PHOTOCRAFT_COMFY_URL`.

## Before you finish a task

Upstream's list (`AGENTS.md` §5) plus: append to `docs/comfy/devlog.md`, update
`docs/comfy/roadmap.md` status markers, and keep every `Cargo.toml` valid at every step.

## Delegation

An orchestration skill routes grind work to Codex (read-only repo analysis, tests, git) and
agy/Gemini (research, visual QA) when they are available. On this PC `codex exec` works with the
prompt on stdin; `agy -p` headless needs a `permissions.allow` rule to run tools. Claude keeps
architecture, implementation and acceptance.
