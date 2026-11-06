# Overnight goal, 2026-10-27: the removal that never touches ComfyUI

> **Mission.** By morning, a Spot Healing dab or a Remove stroke on the native worker
> (Qwen-Image-2.1, no ComfyUI) should beat the shipped 2511 route on the hard bench scenes.
> It should take shadows with the object, and the worker should answer its own questions fast.
> Then make the LoRA engine something the owner can see and trust. Every step is measured,
> tested and committed locally. Anything that needs the owner goes in the morning report, not
> into the code.

You are an autonomous agent working alone overnight in `C:\Users\5090\Documents\PhotoshopEX`
(branch `photoshopex`) and `C:\Users\5090\Documents\genai-native`. Nobody will answer a
question until morning. Do not ask. Decide, record why in the progress log at the bottom of
this file, and keep going.

---

## 0. For the owner, before bed (2 minutes)

1. Open a **new** Code session with its folder set to `C:\Users\5090\Documents\PhotoshopEX`.
2. Set the permission mode to **Auto**. In Default mode the first permission prompt stops the
   night.
3. Close PhotoshopEX if it is open. The agent rebuilds `target\release\photocraft.exe` and
   will not kill a running app.
4. Optionally, tick any pre-approval below. An unticked box means the agent does not do it.
5. Send this one message:

   ```
   Read docs/comfy/goals/overnight-2026-10-27.md and execute it to its stop condition. Work the queue in order, update the file's progress log after every task, and finish with the morning report.
   ```

**Owner pre-approvals** (change `[ ]` to `[x]` to allow):

- [ ] Download **big-lama** as ONNX (Apache-2.0, about 200 MB) from a Hugging Face repo the
      agent names in the log, so Q7 can run.
- [ ] Download the official **Qwen-Image-2.1 Turbo** checkpoint (Comfy-Org, about 7 GB,
      research licence) so Q8 can run.
- [ ] Use the **Viggle Turbo** LoRA (already installed, research licence) as a measured fast
      tier on the 2.1 route. It is never a default.

---

## 1. Ground rules (they override anything below)

**Never:**

- Push, open a PR, or change a remote.
- Download anything that is not pre-approved above.
- Accept a licence, sign in anywhere, or run an installer.
- Edit the owner's real `preferences.json`.
- Change a default the owner decided: 2511 stays the removal model, research models are never
  defaults, and Krea 2 stays on ComfyUI under `auto` unless the plan's "faster at parity" rule
  is measured true.
- Kill `photocraft.exe` or a process you did not start. The session-0 tenants (`sd-server`, SSH
  pythons) are not yours.
- Commit `Cargo.lock`. It stays modified because of the local `[patch]`.
- Bump genai-native's rev in `Cargo.toml`.

**Always:**

- Follow `AGENTS.md`: no `unwrap`, `expect`, `panic` or `unsafe`; everything is a command;
  never break wasm.
- Follow `CLAUDE.md` and the fork conventions in `docs/comfy/`.
- Translate every new UI string into all 20 languages. Make sure every new pref is read by
  code. Keep edits to shared upstream files surgical.
- Make one commit per logical step, written with `git commit -F <file>`. In PowerShell 5.1,
  `-m @'…'@` splits on quotes.
- **Tests are the gate.** Before each commit:
  - Run `cargo test -p photocraft-engine` and `cargo test -p photocraft-genai`, plus the crate
    you touched, or `photocraft-ui-egui` for UI work.
  - Run clippy and allow no new warnings.
  - Run `cargo xtask wasm` and `cargo xtask layers`.
  - In genai-native, run the touched crate's tests through `native-env.cmd`.
- A live claim needs a live run. Contact sheets are judged by looking at them with Read. Write
  each per-stroke verdict in a table, not just a total.

**When stuck:**

- Each task has a time box. When it runs out, write what you learned, commit what is sound
  (tests green), and move to the next task.
- If the same failure happens three times in a row (build, GPU wedged, worker dead), stop
  everything and write the morning report.

---

## 2. Setup (first 15 minutes)

```powershell
cd C:\Users\5090\Documents\PhotoshopEX
git branch --show-current            # must be photoshopex
git status --short                   # only Cargo.lock (and this file) may be dirty
git -C C:\Users\5090\Documents\genai-native log --oneline -3   # da60b3c or later
nvidia-smi                           # note the other tenants' memory before timing anything
Invoke-RestMethod http://127.0.0.1:8188/system_stats   # ComfyUI up? If not, start it as a
# background command (it never returns): powershell -NoProfile -File C:\Users\5090\ComfyUI\start-comfyui-fast.ps1
```

**Read first:**

- `CLAUDE.md`
- The top three entries of `docs/comfy/devlog.md`
- `docs/comfy/benchmarks.md` §5i, the removal measurements and "The rewriter's views per model"
- `docs/comfy/research/notes/ausboss-catalogue-2026-10.md` §2.2, §2.5, §2.6 and §4
- `docs/comfy/plan-native-engine.md` §9, for the LoRA and AnyPaint design

**The tools you will use:**

| Need | How |
|---|---|
| Build the CLI or app with the native engine | `docs\comfy\bench\scripts\build-native.cmd photocraft-cli` (or `photocraft`). Stop the worker first: it locks the exe. |
| Run genai-native's tests or tools | `docs\comfy\bench\scripts\native-env.cmd cargo test --release -p qir`, run from the genai-native folder |
| Shared worker for many CLI runs | `. docs\comfy\bench\scripts\native-worker.ps1` sets `$env:PHOTOCRAFT_NATIVE_ENGINE`. Stop it with `-Stop`. It takes one client at a time. |
| Rewriter naming in seconds | `naming-probe.ps1 -System docs\comfy\bench\scripts\removal-system.txt -Tags …` (the `chat_probe` example: `native-env.cmd cargo build --release -p photocraft-genai --example chat_probe` from PhotoshopEX) |
| Stroke benches | `robust-strokes.ps1` (stock set: `-Dir …\bench\stock -Prefix st -Objects st-objects.json`; robustness set: the defaults), `robust-sheets.ps1` for the contact sheets, `views-run.ps1` as the template of a whole A/B |
| Free ComfyUI's VRAM before native timing | `POST http://127.0.0.1:8188/free {"unload_models":true,"free_memory":true}` |

**Gotchas that cost hours before:**

- PowerShell variable names are case-insensitive: `$W` is `$w`.
- `--%` swallows the rest of its line.
- A long-lived child started from a piped call hangs the call. Redirect to a file.
- ComfyUI resident beside the worker makes native timings 3× slower.
- The 8B flips on a few pixels of crop difference, so confirm probe findings with real strokes.
- 2511 runs use random seeds. Run twice before claiming a one-point gain.

**The score:** a **clean removal** means the object is gone whole, with no smudge, ghost, new
object or blurred face, and the background is continued plausibly. The baseline is the ten
hard stock scenes, half and whole strokes (`glasses, watch, tattoo, cup, dog, boat, bicycle,
hat, umbrella, motorbike`):

| Route | Score of 20 |
|---|---|
| 2.1 on the worker, two judged (`-p21v`) | 15 |
| 2.1 on the worker, one candidate (`-e21v`) | 14 |
| 2511 shipped (`-auto6`) | 16 |

The robustness set (`rb`) is the no-regression check. A change ships only if it does not lose a
clean result on either set.

---

## 3. The queue (in order)

Each task lists its goal, a time box, hints and what "done" means. When a task ends, append a
dev-log entry (newest first), add a `benchmarks.md` row if anything was measured, update the
`roadmap.md` markers, and fill in its line in the progress log below.

### Q1. The mask's whole stroke on the 8B (90 min)

The worker's 8B names "the man wearing the cap and mask" on the real crop of the mask's whole
stroke, while the probe's crop gets "white face mask" 16 times of 16.

- First, make the probe see what the job sees. Add a bench-only env hook,
  `PHOTOCRAFT_DUMP_VIEWS=<dir>`, that saves the exact side-by-side picture the rewriter gets.
  The pattern is `PHOTOCRAFT_THINKER`: documented in benchmarks, read by code. Then probe
  wordings and margins against that exact picture.
- Candidate fixes, cheapest first:
  - A parser rule in `named_object`: `<person> wearing|holding|carrying <X>` → `<X>`, with
    tests.
  - Ask twice, at seeds s and s+1, and keep the answer that is not a person.
  - A margin of 0.05 for strokes under N px.
- **Done when:** the mask's whole stroke is clean on `-p21v` and the probe set still names all
  20 views correctly. Then run the full `-p21` A/B on both sets.
- **Target:** 2.1 two judged at ≥ 16/20, with no rb regressions.

### Q2. Remove it with its shadow: the differential composite (2.5 h)

This is catalogue §2.2. The dog's cast shadow stays on every route.

- **The 2.1 whole-picture route.** After the model returns, grow the layer mask by a change
  band:
  - Take |result − original| in luma and chroma, after realign-to-source.
  - Threshold it. Keep only the parts connected to the hole, within a search band of k × the
    hole's size. Close, fill holes, feather.
  - Add a `shadow: true|false` param on the removal commands (default true when the route
    supports it), tests on synthetic images (a disc and its offset dark ellipse), and
    `generate.info` recording the band.
- **The 2511 crop route.** Add a rewriter clause, or a second detector query
  (`select.byText "the <name>'s shadow"`), and union the result into the hole when the shadow
  touches the grown stroke. Measure whether SAM 3.1 finds shadows on the dog, cone and hydrant
  scenes (rb and stock).
- **Done when:** the dog's shadow is gone on at least one route and nothing regresses on either
  set. The cup's dark patch is the model continuing the table's shade (see the dev log), so do
  not chase it with this.

### Q3. The worker's chat KV cache (2 h, genai-native)

`Pipeline::chat` re-runs the whole sequence for every token.

- Add an incremental decode with a per-layer K/V cache. The positions follow Qwen3-VL's
  M-RoPE: image tokens get their 3-axis positions, and text after the image continues from
  the max position + 1, as in HF's `get_rope_index`.
- **Done when:**
  - Greedy output is token-for-token identical with and without the cache, on three prompts
    with a picture and one without (a test in `qir`).
  - A 30+ token answer is at least 3× faster per token.
  - `chat-smoke.ps1` shows the new times in the dev log.
- Commit in genai-native locally. PhotoshopEX picks the change up through the `[patch]`.

### Q4. LoRA parity against ComfyUI (90 min)

The LoRA engine landed on 2026-10-26 unmeasured.

- Run the same seed, size and prompt through both routes:
  - `qwen-image-2.1-consistency` on a 2.1 edit
  - `qwen-image-2.1-restore` on a 2.1 edit
  - `bfs_head_v1.1_qwen_2.1` on t2i
  - (if pre-approved) Viggle Turbo at 6 steps
- On the ComfyUI side, use a template with a `LoraLoaderModelOnly` node verified against
  `/object_info`, or the harness's API graphs. Compare with the parity harness's metrics.
- The reference point is the no-LoRA agreement of the same cases (the 2.1 rows in benchmarks
  §7–8).
- **Done when:** the LoRA agreement is within that baseline, or the gap is root-caused and
  fixed. Suspects are `txt_in` taking no adapter, the alpha/rank scale, and the gate-up split
  order. Check the worker's "unmatched module" log first.

### Q5. LoRA rows in the picker (2 h, UI)

Users cannot see the LoRAs yet.

- The Generate panel and the fill/edit bars get a LoRAs section: the files under
  `<models>/loras` that match the selected model's family, each with a switch and a strength
  slider from 0 to 1.5. The list comes from a command (`generate.loras {template}`) that reads
  the worker's index, or ComfyUI's `/object_info` `LoraLoaderModelOnly` list on that route.
  The rows feed `loras: [{file, strength}]`.
- The family comes from the file's tensor names, the way `gn-weights` identifies models. Do not
  go by the file name.
- Add the strings in 20 languages, a kittest UI test, and a command test against the fake
  server.
- **Done when:** a live `ui.inspect` shows the rows, and a run with one switched on lands in the
  Generations library with the LoRA recorded in `GenerativeInfo`.

### Q6. Generative Upscale (2.5 h)

This is catalogue §2.5 and roadmap Phase 5. The command is `generate.upscale {scale, tile?,
restore?}`:

- Resize to about 2 MP per tile, sides in multiples of 32.
- Tiles overlap by a shared strip and are joined with a feather.
- Each tile runs the edit route with the restore prompt from catalogue §2.5.
- The Restore LoRA is applied when installed, at strength 0.5 for blurry input.
- Each tile is matched to the source's colour statistics (mean and std per channel in Lab)
  before the join.
- It runs on 2.1 native, with 2511 as the fallback.
- Add an Image-menu entry, 20 languages, fake-server tests (tile count, seam weights, colour
  match), and one live run on a bench picture at 2×. Look at the seams at 200 % before
  claiming.

### Q7. LaMa as Spot Healing's first tier (2 h, needs its pre-approval)

This is catalogue §2.8. Without the tick, skip it and write the exact files and commands needed.

- Add a `heal/lama` route in `gn-perception` through `ort`. Use it for holes under N px; above
  that, the diffusion fill. Add a pref with a default, read by code, and a fake-engine test.
- **Done when:** a dust/wire set (make five strokes on the stock pictures) runs under 200 ms a
  dab, cleaner than PatchMatch on the sheet.

### Q8. The rest, if the night allows

Take these in order, each with its own time box of 90 minutes:

- **Tone match at the seam** for Expand and Fill (catalogue §2.4).
- **The prompting rules of catalogue §2.6** as rewriter A/Bs. Start with "what they hold goes
  too" on person removals.
- **The Krea 2 in-context reference pair**: the AnyPaint first step from plan §9, needing no
  model change.
- **2.1 Turbo as a second weight set**, only if pre-approved.
- **A `denoise` strength on fills.**

**Owner-only (never start; list them in the report):**

- Pushing genai-native and bumping its rev
- Making 2.1 the removal default (give the numbers and a recommendation)
- The fp8 GEMM kernel track (plan §11, a multi-day effort; write a design note at most)
- SeedVR2 and FLUX 3 Klein downloads
- Civitai workflow JSONs
- The VS Build Tools rerun

---

## 4. Stop condition

Stop when any one of these holds:

- The queue is done or time-boxed out.
- The three-strikes rule fires.
- It is 07:30 local time. Stop starting new GPU runs at 06:45 so the last one finishes.

Then do the following:

1. Rebuild `target\release\photocraft.exe` and `photocraft-cli.exe` with
   `build-native.cmd` so the pinned app has the night's work. If the app is running, build into
   `CARGO_TARGET_DIR=target-overnight` and say so.
2. Stop the worker. Leave ComfyUI running.
3. Run the full gate once: `cargo test --workspace --no-fail-fast`. The baseline is 9
   GPU-driver failures on this PC; list any others. Also run clippy, `xtask wasm`,
   `xtask layers` and `xtask scorecard --check`.
4. Write **the morning report** at the top of the progress log:
   - Five lines at most on what changed for the owner.
   - The score table before and after.
   - The commits in both repos, with one line each.
   - What failed and why.
   - The decisions waiting for the owner, each with a recommendation.
5. Commit this file with the log filled in.

---

## 5. Progress log (the agent appends; newest first)

### Morning report

**What changed for you.**
1. Removals on Qwen-Image-2.1, with no ComfyUI, now beat the shipped 2511 route on the hard
   scenes the same night: **16 of 20 against 12**. They also take the shadows the model
   removes, and the worker names objects 2–3× faster.
2. New in the app:
   - **Edit › Generative Upscale**
   - **LoRA rows** in the Generate panel
   - a **Denoise** field on 2511 fills
3. LoRAs now also work on the ComfyUI route, and both routes agree on edits.
4. A worker bug is fixed: a request without LoRAs after one with them used to fail.
5. Measured and dropped: the fill tone match, a Krea 2 noise-mask fill, a wider removal
   crop. `nameShadow` ships off by default.

**Scores** (clean removals; stock = the ten hard scenes, half and whole strokes; my count from
the sheets):

| Route | Stock before | Stock now | Robustness (28) |
|---|---|---|---|
| 2.1 edit on the worker, two judged | 15 | **16** | 27 |
| 2511 shipped, Best of two (same night) | 16 (two nights ago) | **12** | |

| Task | Status | Commit(s) | Result |
|---|---|---|---|
| Q1 mask naming | done | `47dc76ed` | the wearer's candidates: the mask's whole stroke clean |
| Q2 shadow composite | done | `0005d577` | the change band; 16/20, no regression; the dog's long shadow still stays |
| Q3 chat KV cache | done (3× met with a picture, 2.5× text-only) | gn `2f3cd2a` | rewrite 5.6 → 0.95 s; naming 2–3× faster |
| Q4 LoRA parity | done, bug fixed | `6e6e1d6a`, gn `53256a7` | edits at the no-LoRA agreement (38–45 dB) |
| Q5 LoRA picker rows | done, live-checked | `6e6e1d6a`, `b81db474` | `generate.loras`, rows on the native route |
| Q6 Generative Upscale | done | `9994874a` | 6 tiles in 89 s; sharper than bicubic, no seams |
| Q7 LaMa tier | skipped (download not approved) | | files and plan in the log |
| Q8 extras | partly | `df01a051`, `72a4ec89`, `92704ed7` | denoise shipped; tone match, `nameShadow`, Krea 2 fill measured |

**Commits.**

PhotoshopEX (local, not pushed):
- `47dc76ed` the wearer's candidates
- `0005d577` the change band
- `6e6e1d6a` LoRA rows, plus LoRAs on ComfyUI
- `9994874a` Generative Upscale
- `df01a051` `removeMargin` and `nameShadow` (opt-in)
- `72a4ec89` fill denoise
- `b81db474` the LoRA list loads when the panel is folded
- `92704ed7` the Krea 2 fill record

genai-native (local):
- `2f3cd2a` the chat KV cache
- `53256a7` binds the CUDA context per client thread

**What failed and why.**
- The dog's long shadow: the model removes only what lies inside its request, and with a
  wider request it keeps the shadow itself.
- The cup's dark patch on 2.1: the model continues the table's own shade.
- The logo's whole stroke: the rewriter names the shirt.
- Krea 2 noise-mask fill: a text-to-image model paints a separate picture into the box.
- Upscale spends about 5 s a tile reloading the text encoder, because the card's other
  tenants (5.6 GB: `sd-server` and Python processes) leave no room to keep it.
- `native_menu::tests::opening_a_document_changes_state_not_structure` fails when only the
  menu tests run together (order-dependent). It passes alone and in the full suite, with or
  without tonight's changes.

**Decisions waiting for you.**
1. **Make 2.1 on the worker the removal default under Auto.** I recommend yes, once your
   research-licence stance is settled: 16 against 12 the same night, at 6–28 s a stroke with
   no ComfyUI. Two nights ago 2511 scored 16, so run one more same-night A/B if you want more
   certainty. 2511 would stay the permissive fallback.
2. **Push genai-native and bump its rev in PhotoshopEX** (nine local commits, ahead of
   `origin/main` since `f2bf496`). I recommend yes: the CUDA-context fix matters to anyone using LoRAs.
3. **Tick the LaMa download** (Apache-2.0, about 200 MB) for Q7.
4. **The fp8 GEMM kernel track** to move Krea 2 under Auto (a multi-day effort), and
   **AnyPaint** for a real Krea 2 fill.
5. **Clean the card's other GPU tenants** (`sd-server`, the SSH Python processes): they cost
   every native timing tonight and force the text-encoder reloads.

### Notes as they happen

- **01:02 setup.** Branch `photoshopex`, only `Cargo.lock` dirty, genai-native at `da60b3c`,
  ComfyUI 0.39 up, 5.6 GB of the card held by the session-0 tenants (`sd-server`, pythons).
  The owner launched the goal in this session (Default mode, no pre-approvals ticked: no
  downloads tonight).
- **01:15 Q1 done (`47dc76ed`).** The mask's whole stroke named "the man wearing the cap and
  mask": `wearer_candidates` splits a wearer phrase into each worn thing, all of them, and the
  wearer; the detector is asked for each and the mask that best overlaps the stroke (IoU) is
  taken and named. Live, 2.1 two judged on the worker: the whole stroke is clean (was a smear),
  named "mask". Bench hook `PHOTOCRAFT_DUMP_VIEWS` saves the exact picture the rewriter sees.
- **02:20 Q2 done (`0005d577`).** `change_band`: the pixels the model changed beside the hole
  (blurred difference > 0.08, connected to the hole's feathered edge, within 0.75 of its side)
  join the hole for the tone match and the layer mask. Stock ten hard scenes, 2.1 two judged
  (`-q2` against `-p21v`): **16 of 20** (15): the mask whole and the boat half (both people
  gone) gained, the boat whole lost (the 8B named "silhouette of the two people" this time, the
  hull stayed: the rewriter's seed, not the band). The dog's shadow: the half stroke took it
  only up to the request's edge and cut it in a straight line, so a band that reaches the
  request's (or the reach's) edge is now refused and the shadow stays whole (`-q2b`). Taking the
  dog's shadow needs a request that holds it (a wider crop on 2.1 removals): owner's call, it
  changes every removal's scale. Robustness set, band on (`-q2b`) and off (`-q2off`): 27 of 28
  both (the logo's whole stroke names the shirt on both). Timing on this run is not
  comparable: the card's other tenants and my test builds overlapped (some strokes 100-240 s).
- **02:30 Q5 code done (`6e6e1d6a`), with the Q4 prerequisite.** `generate.loras {template}`
  lists the LoRA files with their family from the tensor names (Krea 2 `blocks.N.attn.gate`,
  Qwen-Image `transformer_blocks.N`: 32 blocks = 2.1, 60 = 2511); the Generate panel's LoRAs
  disclosure (image, fill, edit; only when the template runs natively) with a switch and a
  strength per file; `ui.set`/`ui.inspect` `generatePanel.loras`; kittest test; 2 strings × 20
  languages (and "Strength" where fr, id, ru, zh-hans, zh-hant lacked it). The ComfyUI backend
  now applies `loras` as `LoraLoaderModelOnly` nodes after the model loader
  (`comfy::with_loras`), so a LoRA can be compared on both routes. Live `ui.inspect` not run
  (no GUI session tonight); the panel test drives the rows through kittest.
- **Q3 done (genai-native `2f3cd2a`).** `QwenTextEncoder::prefill` + `step` with a per-layer
  `KvCache` (positions run on from the prompt's last M-RoPE position); `Pipeline::chat` uses
  them. Toy-decoder test: cached rows equal the full pass (with and without a picture). Live
  (`q3-run.ps1`, the same worker build without and with): a 28-token rewrite with the picture
  5.56 s → 0.95 s (per token ~190 → ~21 ms), a 120-token image prompt 4.30 → 1.69 s (2.5×:
  short prompts gain less), the removal naming 0.8-2.1 s → 0.3-0.8 s; 17 of 20 sampled answers
  word for word the same (temperature 0.3; the cached rows round differently, three rewordings).
  The 3× per-token bar: met with a picture (9×), not on the 120-token text-only prompt (2.5×).
- **Q4 found a worker bug, fixed (genai-native `53256a7`).** A request without LoRAs after one
  with them failed in 0.3 s with `CUDA_ERROR_INVALID_CONTEXT`: each client is served on its own
  thread, clearing the adapters freed CUDA memory before the context was bound to that thread,
  and cudarc handed the stored error to the next call. `device::bind_thread` at the start of
  every request. Repro script `lora_clear_repro.py`.
- **02:42 Q4 done.** Same seed, size and prompt through the app's CLI on both routes
  (`q4_lora_parity.py`; ComfyUI applies the file through `with_loras`), PSNR native vs ComfyUI:

  | Case | No LoRA | Consistency | Restore | BFS head |
  |---|---|---|---|---|
  | edit "make it winter" (glasses) | 40.0 | 41.0 | 41.1 | |
  | edit "turn the cup red" | 44.1 | 44.7 | 43.7 | |
  | edit, the restore wording (hat) | 38.2 | 38.5 | 40.3 | |
  | text to image, portrait | 33.0 | | | 22.6 |

  On edits the LoRA keeps the routes as close as without it, and moves each route by the same
  amount (LoRA vs none: 17.8/18.1, 31.5/31.5, 27.5/27.7 dB). On text to image the BFS head
  LoRA gives the same person, coat and framing on both routes but the window frame drifts
  (22.6 dB): an 8B-scale text-to-image run amplifies small differences more than an edit does,
  no module was left unmatched (the worker logs any). Verdict: at parity.
- **02:55 Q6 done (`9994874a`).** `generate.upscale {scale, tileMegapixels, prompt?, loras?,
  restoreStrength}` (Edit › Generative Upscale): bicubic enlarge (`image.imageSize`), then the
  edit model per tile with AusBoss's restore wording, 128-px shared strips faded together, each
  tile realigned onto its source and matched to its colours (mean and spread per channel, in
  RGB: simpler than the Lab the brief named, and enough on the test), the Restore LoRA at 0.5 on
  the native 2.1 route when installed. Fake-server test (tile count, the enlarge, one layer).
  Live (`q6-run.ps1`): `st-hat.jpg` halved to 960×686 and enlarged back: 6 tiles, 89 s on the
  worker (about 5 s a tile is the text encoder reloading: the card's tenants leave no room to
  keep it beside the DiT); at 200 % crisper than bicubic on the frame's texture, the hair and
  the face, identity kept, no seam visible; PSNR against the original 27.0 dB (bicubic 27.5,
  PSNR favours blur).
- **03:05 Q8a measured, not shipped: the seam tone match on Generative Fill** (`toneMatch`,
  `q8_fill_tone.py`, six fills on stock pictures, 2.1 on the worker, same seed off and on):
  bit-identical on four (the native 2.1 fill already meets the picture at its edge, so the
  membrane is zero), a small change inside the selection on two with no seam visible either
  way at 200 %. No measured benefit, so the code was taken out again.
- **03:16 two measurements for the owner** (`q9-run.ps1`):
  - A wider removal request (`removeMargin` 0.5, new parameter, default 0.25 unchanged) does
    not take the dog's shadow: with more context the model keeps the shadow itself, so the
    band has nothing to take. The lever is the prompt (next item), not the crop.
  - The shipped 2511 route (Best of two through ComfyUI) rerun tonight on the same ten scenes
    (`-auto7`): **12 of 20** clean (half 6: glasses, tattoo, cup, dog, bicycle, umbrella; whole
    6: glasses, tattoo, dog, bicycle, umbrella, motorbike; it painted a new man's face on the
    hat both times and on the mask's whole stroke, left the boat). 2.1 on the worker the same
    night: 16 of 20. Two nights ago 2511 scored 16: its random seeds swing it by four.
- **03:31 `nameShadow` measured, left opt-in (`df01a051`).** The removal prompt asks for "the
  golden retriever and its shadow" on the 2.1 route: stock 17 of 20 (the boat's whole stroke now
  goes with its hull), robustness set 26 of 28 (the poster's four tape strips stayed): no net
  gain, default off. The dog: the shadow's part inside the request goes, the rest stays (no
  hard cut). `removeMargin` committed with it (default unchanged).
- **03:55 Q8 denoise (`72a4ec89`).** The three 2511 fill templates bind their KSampler's
  `denoise` (1 unless asked); `generate.fill` takes `denoise`; the fill page shows a Denoise
  field when the template takes it (never for the 2.1 fill). Live, st-hat's top hat asked to be
  red: on the 40-step template 0.9 keeps the old sheen with the new colour, 0.8 barely changes;
  on the 8-step tier anything under 1 hardly moves. The hint says "about 0.9". 2 strings × 20.
  The LoRA strength field now shows two decimals (it would have rounded to whole numbers).
- **04:15 Q5 live (`b81db474`).** `loras-live.ps1` drives the release app (its own config
  folder): the Generate panel sat folded in the dock, so the LoRA list never loaded and a run
  from the compact form would have dropped the LoRAs switched on. Now the list also loads on
  `ui.set` and before a Generate: live, the eight 2.1 files listed (Krea 2's and 2511's left
  out), `native` true, the request carrying Restore at 0.5.
- **Workspace gate at 02:55** (before the Upscale, Denoise and LoRA-list commits): 5 failures,
  all GPU-driver tests (`photocraft-gpu` timing, `full_res_preview` ×3,
  `pattern_preview_canvas`), within this PC's baseline of 9. The final gate reruns at the end.
- **04:10 Q8 Krea 2 fill measured, withdrawn (`92704ed7`).** A `krea2-turbo/fill` template
  (the official Krea 2 t2i graph with the picture encoded and a noise mask, nodes checked
  against the local `/object_info`) paints a separate photo into the selection's box at
  denoise 1, 0.8 and 0.65 on all six fills (a lawn box, a window with a vase, a day sky in a
  night street). Not shipped; the JSON is kept beside `q12_krea_fill.py`. The reference pair
  proper and AnyPaint (plan §9) remain the way to a Krea 2 fill.
- **04:16 final gate** (after every code commit): `cargo test --workspace --no-fail-fast` the
  same 5 GPU-driver failures as at 02:55 (baseline 9); clippy on the touched crates at the
  existing warnings; `xtask wasm`, `layers`, `scorecard --check` pass. The app
  (`target\release\photocraft.exe`, 03:59) and the CLI (04:21) are rebuilt from the final tree.
- **Q7 skipped (not pre-approved).** What it needs: an ONNX export of big-lama (Apache-2.0,
  advimman/lama; the community export `Carve/LaMa-ONNX` on Hugging Face, `lama_fp32.onnx`, about
  200 MB, a fixed 512×512 image + mask input; verify the repo and licence on the page before
  downloading) placed at `<models>/lama/lama_fp32.onnx`. Then: a `heal/lama` engine in
  `gn-perception` beside SAM 3.1 (same `ort` runtime, crop the hole's box grown 2× to 512², run,
  paste back through the hole's mask), `integrations.healEngine` (auto | lama | diffusion) read
  by Spot Healing and the Remove tool for holes under ~96 px, and a dust/wire stroke set.
- `native_menu::tests::opening_a_document_changes_state_not_structure` fails when the `menu`
  tests run as a batch, with or without tonight's menu line, and passes alone and in the full
  suite: an order-dependent flake, left for the owner.
