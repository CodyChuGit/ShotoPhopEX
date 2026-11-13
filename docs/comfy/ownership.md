# Whose code is whose

ShotoPhopEX is a fork of [PhotoCraft](https://github.com/storytold/photocraft) by the ArtCraft
team. Most of this repository is PhotoCraft's code, used under its MIT OR Apache-2.0 licence; the
fork adds local generative AI features on top. This page says which parts belong to whom, so
upstream merges, licence notices and reviews can tell them apart. It is a map, not legal advice;
the licence texts govern (`LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`).

Measured on 2026-11-13 against the last upstream merge (PhotoCraft 0.22, merge base `885a6ddf2`).
To refresh the numbers, run `git diff --stat $(git merge-base HEAD upstream/main) HEAD` with
upstream fetched as the `upstream` remote.

## Upstream's (PhotoCraft, ArtCraft Team and contributors)

Everything not listed under "The fork's" below: the editor itself (about 300,000 lines of Rust in
`crates/` and `apps/`), its assets, docs (`docs/user/`, `book/`), packaging, CI and tests. Its
copyright stays with its authors; `NOTICE` carries their notice unchanged.

Not included: upstream's `docs/brand/` (the ArtCraft name, wordmark and logos). They are
trademarks, not open source, and the brand licence asks modified versions to remove them and not
to present themselves as ArtCraft products. They were removed in 0.1.2; do not add them back, and
do not use "ArtCraft" as the fork's name, publisher, folder, link label or logo. Saying in plain
text that ShotoPhopEX is based on PhotoCraft by the ArtCraft team is allowed.

## The fork's (ShotoPhopEX, Cody Chu and contributors)

Whole folders and files that exist only in the fork:

| Where | What |
|---|---|
| `genai-native/` | The native AI engine (its own workspace; `genai-native/NOTICE` lists the code it builds on) |
| `crates/genai/` | The generative backend, model catalogue and workflow templates |
| `crates/engine/src/generate_*.rs`, `select_ml_cmds*.rs`, `model_packs.rs`, `memory_profile.rs`, `model-packs.json` | Generate, Edit, Expand, Split, Swap, Upscale, Object Select and model download commands |
| `crates/ui-egui/src/generate_*.rs`, `generations_panel*.rs`, `generative_bar*.rs`, `model_picker*.rs`, `object_select*.rs`, `angles_ui*.rs`, `gen_preview.rs`, `variations_strip*.rs`, `get_models_ui.rs`, `memory_profile_ui.rs`, `engine_ui.rs`, `alerts.rs`, `app_icon_ui*.rs`, `dialogs/filter_dialog/generative.rs` | The generative panels, dialogs and on-canvas tools |
| `assets/app-icon/sp/` | The "Sp" app icon |
| `docs/comfy/`, `docs/compliance/`, `docs/images/genai/` | Fork docs and compliance notes |
| `packaging/third_party_notices.py`, `packaging/icon-tile.py`, `.github/workflows/shotophopex-release.yml`, `.github/RELEASE_NOTES.md` | Fork packaging and release |
| `vendor/craft-update/` | A copy of upstream's craft-update library (MIT OR Apache-2.0, ArtCraft Team) with two marked changes; see its README |

## Upstream files the fork changes

About 180 upstream files (plus the interface translation catalogues) carry fork changes, about
10,000 changed lines in all. They are the places where the fork's features plug into the editor,
because upstream has no extension point the generative features could use instead. Many fork
changes in them are marked with a `ShotoPhopEX` or `Fork:` comment. Grouped:

| Where | Files | Why |
|---|---|---|
| `crates/ui-egui/src/` | ~80 | Menus, the dock and panels (Generate, Generations), the Remove tool's Generative option, the filter dialog, preferences, the start screen, About, links, branding (`i18n.rs` shows "ShotoPhopEX" where upstream text says "PhotoCraft") |
| `crates/ui-egui/src/i18n/*.tsv` | 40 | Translations of the fork's interface strings, added at the end of each catalogue |
| `crates/engine/src/` | 18 | Command registration, background jobs, preferences, build info (app name and upstream base version) |
| `apps/photocraft`, `apps/photocraft-cli` | 14 | Wiring the native engine in, app identity |
| `crates/automation`, `crates/io`, `crates/gpu` | 8 | Small hooks for the generative commands |
| `packaging/` | 17 | The fork's own name, publisher, install folder, upgrade code and release file names |
| `assets/app-icon/` | 16 | The fork's icon in place of upstream's |
| `README.md`, `NOTICE`, `ATTRIBUTION.md`, `LICENSE-MIT`, `SECURITY.md`, `Cargo.toml`, a few docs | ~10 | Fork description, notices and settings |

Apache-2.0 section 4(b) asks that modified files carry a prominent notice that they were changed.
The fork's `NOTICE` and this page say so for the files above; Git history records each change.

## Installer identity (since 0.1.2)

| | Upstream PhotoCraft | ShotoPhopEX |
|---|---|---|
| Windows publisher (Manufacturer) | Learning Machines LLC | ShotoPhopEX |
| MSI UpgradeCode | `D9660F32-2D0A-4ABE-B73D-17BE25F1542C` | `7A30FAE6-70CF-4CB3-A7C7-384E37ECCBB2` |
| Install folder | `Programs\ArtCraft\PhotoCraft` | `Programs\ShotoPhopEX` |
| Remembered-folder key | `Software\PhotoCraft` | `Software\ShotoPhopEX` |
| Linux vendor / AppStream developer | Learning Machines LLC | ShotoPhopEX |

Still shared with upstream (renaming them needs updater and settings migration work):
the exe name `photocraft.exe` (and with it the Windows `App Paths` entry), the file-association
ProgIDs `Photocraft.Image` and `Photocraft.PhotoshopDocument`, the app id
`ai.storyteller.photocraft` (Linux desktop file, Flatpak), the settings folder and the crate names.
