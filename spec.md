# Matched benchmark: RTX 5090, RTX 3090, M1 Air 16 GB

Goal (Cody, 2026-11-08): one apples-to-apples speed section in the README. Same task, same
input picture, same size, same seed on every device, each bar a different colour per device.

## Rules

- App: v0.1.1 release build (or newer main), driven through its control channel / commands, in
  its own instance with a COPY of the prefs. Never close windows or processes you didn't start.
- One GPU job at a time; nothing next to a cargo build.
- Input picture: `kitchen-1024.png` (1024×1024, sha256
  `0345bd94a6a883e788ae98b7017f00b3913dd089fc14f3a84df2bade6c4aa4ff`), in this folder.
- Size: **1024×1024** for everything. Seed 42. Variations 1. Image Quality: Fast.
- Model: **Auto** (whatever the app's memory profile picks on that device). Record which model ran.
- Per task: 1 cold run after Purge models, then 3 warm runs. Report cold, warm median, and the
  model. If a task doesn't fit or isn't on that device, say so (don't force another model).
- Report also: GPU/chip, memory, OS, driver, memory profile chosen, peak memory if easy.

## Tasks

| id | Task | How |
|---|---|---|
| t2i | Generate Image | `generate.image`, prompt "a cozy kitchen with a white fridge and a bowl of fruit, morning light", 1024×1024 |
| fill | Generative Fill | rect selection x 640–920, y 400–570 (the fruit bowl), `generate.fill` prompt "a vase of sunflowers" |
| remove | Generative Remove | same selection, Generative Remove |
| edit | Generative Edit | no selection, `generate.edit` "make the fridge red" |
| expand | Generative Expand | 1024×1024 → 1536×1024 (add 512 px on the right), empty prompt |
| split | Split into Layers | Medium, subjects "fridge, fruit bowl" |
| clown | Clown Pass | default |
| seltext | Select by text | `select.byText` "the fruit bowl" |
| rembg | Remove Background | `generate.removeBackground` |
| enhance | Prompt Enhance | `generate.enhancePrompt` on the t2i prompt |

Write results to `summary.md` (columns: id, model, cold s, warm median s, notes) and send the
table to the README thread ("Update README for the Mac version").
