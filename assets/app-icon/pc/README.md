# Fork app icon: the "Pc" tile

PhotoCraftEX shows its own "Pc" tile (PhotoCraft) instead of upstream's kitsune. Until
2026-10-29 this was a "Ps" tile; it was redrawn as "Pc" so the repository carries no Adobe mark.
The code points here through three one-line edits (`crates/ui-egui/src/brand.rs`,
`apps/photocraft/src/app_icon.rs`, `apps/photocraft/build.rs`). The tile is the same artwork as
the packaging icon set one folder up.

## Palette

| Colour | Hex | Used for |
|---|---|---|
| Navy | `#001E36` | the rounded tile |
| Light blue | `#31A8FF` | the "Pc" |

## Geometry

A rounded square with the same corner radius as the kitsune tile (112/512), "Pc" in a bold sans
(Segoe UI Bold on Windows, Arial or Liberation Sans Bold elsewhere) at 56% of the tile, centred,
cropped 22 units a side as the packaging icons are.

## Files

| File | Used by |
|---|---|
| `pc-128.png` | `brand.rs`: the title-bar mark (mipmapped texture) |
| `pc-256.png` | `app_icon.rs`: the window and taskbar icon on Windows and Linux |
| `pc.ico` | `build.rs`: the `photocraft.exe` resource icon (PNG entries at 16 to 256 px) |

## Regenerating

```powershell
python -I packaging\icon-tile.py . photocraft Pc "#001E36" "#31A8FF"
copy assets\app-icon\hicolor\128x128\apps\ai.storyteller.photocraft.png assets\app-icon\pc\pc-128.png
copy assets\app-icon\hicolor\256x256\apps\ai.storyteller.photocraft.png assets\app-icon\pc\pc-256.png
copy assets\app-icon\photocraft.ico assets\app-icon\pc\pc.ico
```

Delete the `photocraft-macos-512.png` the script also writes; it is not tracked.
`app_icon.rs`'s `exe_icon_has_every_shell_size` test checks the `.ico`.
