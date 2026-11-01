# PhotoCraftEX app icon

A **"Pc" tile** (PhotoCraft): a dark navy rounded square with the letters "Pc" in light blue,
drawn with `packaging/icon-tile.py`, which also draws the sister apps' tiles. Until 2026-10-29 the
owner's builds showed a "Ps" tile here; it was redrawn as "Pc" so the repository carries no Adobe
mark. The upstream-style "Px" tile can still be regenerated with `python -I packaging\icon-px.py .`.

## Palette

| Colour | Hex | Used for |
|---|---|---|
| Navy | `#001E36` | the tile |
| Light blue | `#31A8FF` | the letters |

## Geometry

A 512-unit tile (`viewBox="0 0 512 512"`), rounded square with `rx=112`, no border, the letters
set in Segoe UI Bold at 56 % of the tile, centred. The macOS renders pad it onto Apple's
824/1024 icon grid; the Windows and Linux renders use the full tile with no crop, so
the rounded corners stay transparent (upstream #4478 made the same change to its icon).

## Provenance

Made for this branch on 2026-10-09 with `packaging/icon-px.py` (Pillow, Segoe UI Bold), replacing
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
python -I packaging\icon-tile.py . photocraft Pc "#001E36" "#31A8FF"
cargo run -q -p xtask -- ico assets\app-icon\photocraft.ico target\icon-px\ico-16.png target\icon-px\ico-20.png target\icon-px\ico-24.png target\icon-px\ico-32.png target\icon-px\ico-40.png target\icon-px\ico-48.png target\icon-px\ico-64.png target\icon-px\ico-128.png target\icon-px\ico-256.png
```

Pass other letters or colours to `icon-tile.py`; `icon-px.py` draws the upstream-style "Px" tile.
