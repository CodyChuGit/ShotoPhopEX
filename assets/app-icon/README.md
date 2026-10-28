# PhotoshopEX app icon

A **Photoshop-style tile**: a dark navy rounded square with the letters "Px" in light blue, the
look of the Adobe application tiles, with PhotoshopEX's own letters (Adobe's "Ps" mark and its
icon artwork are Adobe's trademarks; nothing of theirs is copied here).

## Palette

| Colour | Hex | Used for |
|---|---|---|
| Navy | `#001E36` | the tile |
| Light blue | `#31A8FF` | the letters |

## Geometry

A 512-unit tile (`viewBox="0 0 512 512"`), rounded square with `rx=112`, no border, the letters
set in Segoe UI Bold at 56 % of the tile, centred. The macOS renders pad it onto Apple's
824/1024 icon grid; the Windows and Linux renders crop 22 units off each side, as the previous
icon did, so the packaging scripts need no change.

## Provenance

Made for this branch on 2026-10-09 with a local Pillow script, `icon-px.py` (Segoe UI Bold; kept out of the repository), replacing
the PhotoCraft kitsune (upstream storytold/photocraft, `assets/app-icon/` there). License: see
`LICENSE.txt`.

## Files

- `photocraft.svg`, `photocraft-small.svg`: the editable master (a `<text>` element; the PNGs are
  what ships).
- `photocraft-1024.png`: 1024 px render on the macOS grid.
- `photocraft.icns`: macOS bundle icon (`CFBundleIconFile`).
- `photocraft.ico`: Windows icon, 16–256 px, embedded in the `.exe` by `apps/photocraft/build.rs`.
- `hicolor/<size>/apps/ai.storyteller.photocraft.png` (16–512) and `hicolor/scalable/...svg`:
  Linux icon theme.

The file names stay `photocraft.*` because `build.rs`, `app_icon.rs`, `brand.rs` and the
packaging scripts read them by name.

## Regenerate

```powershell
python -I <path to your icon-px.py> .
cargo run -q -p xtask -- ico assets\app-icon\photocraft.ico target\icon-px\ico-16.png target\icon-px\ico-20.png target\icon-px\ico-24.png target\icon-px\ico-32.png target\icon-px\ico-40.png target\icon-px\ico-48.png target\icon-px\ico-64.png target\icon-px\ico-128.png target\icon-px\ico-256.png
```

Change `LETTERS`, the colours or the font at the top of `icon-px.py` (a local file, not in the repository).
