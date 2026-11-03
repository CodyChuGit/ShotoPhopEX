# Split into Layers: the master spec and one-shot prompt

Written 2026-11-03 for the owner's ask: make Split into Layers take a complex scene (the café
photo: seven people at overlapping depths, a table with cups and playing cards in front, hats,
smoke, a busy wall, several partly hidden figures) apart into clean, separate layers. The first
part says what the best Split does. The second part is the prompt: hand it, as it stands, to a
session with this repository, the owner's PC and the native worker, and it has what it needs to
build the rest. The third part is the status: what landed, what is measured, what is open.

## 1. What the best Split does

On the café photo the best Split gives about twelve layers, each whole, stacked in depth order,
over a background with nobody in it. Moving any layer reveals a finished picture underneath,
never a hole or a ghost.

| Layer, top to bottom | What it holds | What had to happen |
| --- | --- | --- |
| Playing cards | The cards on the table | Nothing: they hide only table |
| Cup 1, Cup 2 | Each cup on its own layer | The table surface repainted where each cup stood |
| Table | The whole front table | The laps and legs of the people behind it completed |
| Old man in the hat | Whole, hair strand by strand | His lap, hidden by the table, completed |
| Man in the black hat | Whole, cigarette and all | Nothing hides him; the frame cuts him and the frame stays |
| Woman in the blue blouse | Whole, the raised card included | Her lower body behind the table completed |
| Man with glasses | Whole | The shoulder hidden by the old man completed |
| Man in the striped shirt | Whole | Nothing hides him |
| Man in the white shirt and cap | Whole | The part behind the woman's arm completed |
| Man at the far left edge | Whole where the frame allows | Most of him was behind the man in the black hat |
| Smoke (later) | A semi-transparent haze layer | Dehazing; see the follow-ups |
| Background | The room: walls, pictures, calendar, the back table, nobody there | Every subject removed back to front, each hole filled with what was behind it |

Five rules make that result right rather than merely plausible:

1. **Whole subjects.** Where another layer stood in front of a subject, the fill completes the
   hidden part, so each layer stands alone. A hidden part is painted only when it was hidden: a
   subject cut by the picture's edge stays cut.
2. **Depth order is the layer order.** The nearest thing is the top layer. Order comes from the
   picture (who hides whom), not from size alone.
3. **Soft edges.** Hair, smoke wisps and motion blur keep partial alpha; nothing is cut with a
   hard line.
4. **Nothing invented in the open.** A fill may complete what was hidden; it may never add a
   person or thing that was not there. A fill that paints a person back is thrown away.
5. **Refuse rather than ruin.** When a hole cannot be filled cleanly after the seeds allowed,
   that subject keeps its visible part and its hole keeps the picture's own pixels (nothing
   invented), and the result says so. When no subject at all can be cleared, Split fails with
   nothing changed.

## 2. The one-shot prompt

> Everything from here to the next `##` heading is the prompt. It assumes nothing but this
> repository, the owner's PC and the native worker.

You are building the best possible **Split into Layers** for ShotoPhopEX (the fork
CodyChuGit/comfy-photocraft, branch `photoshopex`, a Rust workspace; the native generative worker
is the private crate set CodyChuGit/genai-native pinned in `crates/genai/Cargo.toml`). The goal:
one flat photograph of a busy scene goes in; every person and foreground thing comes out whole on
its own layer, in depth order, over a background with nobody left in it. The reference picture
is the café scene (seven people at overlapping depths, a front table with two cups and playing
cards, hats, smoke, a busy wall, figures cut by the frame); the regression pictures are the
owner's portrait (`C:\Users\5090\Downloads\images.jpg`, a woman and a dog) and his 12-megapixel
portrait (a large subject the fill once painted back). Work until the acceptance criteria at the
end hold, measured in the app.

### House rules

- Measure in the real app on the owner's PC (a Remote Control session), never only in `cargo
  test`. Build a scratch copy (`cargo build --release -p photocraft --features native-diffusion`
  into a target directory of your own, e.g. `target-split`), launch it with `--new-instance
  --control <port> --control-token-file <file>` and the owner's models folder, drive it over the
  control protocol (`photocraft-cli`, see `docs/development.md`), and quit only the instance you
  started (`app.quit` on your port). Never close the owner's windows or engines.
- One GPU: before any GPU job check `nvidia-smi` for other compute processes and `Get-Process
  cargo,rustc` for a build, and wait; re-check before every round. A GPU job beside a heavy build
  has crashed the PC.
- Models are bring-your-own. Never gate a model on licence grounds; never default research-only
  models off. New weights go in the ComfyUI models folder
  (`C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models`), hash verified, recorded in
  `docs/comfy/models.md`. LoRAs may be downloaded without asking; a base model gets a one-line
  heads-up first.
- Do not bring Qwen-Image-Layered back (the owner dropped it 2026-11-02: 5 of 10 seeds usable).
- Nothing invented: a fill may complete what was hidden, never add what was not there. Never
  show the owner a fill that repainted a person; check with SAM and delete it.
- Record what you measured in `docs/comfy/devlog.md` (newest first) and `docs/comfy/benchmarks.md`;
  keep the README's Split rows true. Rust tests run against the fake worker
  (`crates/genai/src/fake.rs`); the CI gate is `cargo xtask ci` (fmt, clippy, tests).
- One PR to `photoshopex` per coherent step; the devlog entry in the PR.

### The code as it stands

- `crates/engine/src/generate_split_subjects.rs`: the command `generate.splitLayers`
  (`SubjectsPlan`, `run`), the hole, the fill tries, the painted-back check `subject_left_in`,
  `ToneGrid`, the matte fallback. Parameters parsed in `generate_cmds.rs::run_split`; the
  command spec string (what the dialog draws) in `generate_cmds.rs::specs` under `SPLIT`.
- `crates/engine/src/select_ml_cmds.rs`: `SubjectFinder` (`subject_finder`, `instances`,
  `only`, `at_most`), `Instance = (coverage: Vec<f32>, bounds: Rect, pixels: u64)`,
  `detect`/`detect_any` (SAM 3.1 text route with `maxDetections`, `threshold`, an optional box),
  `without_specks`, `fill_enclosed`, `coverage_bounds`, `refine_coverage`, `vitmatte_alphas`.
- `crates/engine/src/generate_cmds.rs`: `removal_job` (the fill template the preferences give;
  under `auto` on the native worker `qwen-2.1/edit` as a whole-crop instruction with the hole
  smoothed over, `prefill`), `background_job` (`qwen-2.1/fill`, the hole flattened to grey,
  `BACKGROUND_PROMPT` "…with nobody there"), `fill_region` (runs a `FillJob` on a rect of the
  document with a coverage, adds result layers), `name_selected` (the vision-language model
  names what a hole holds, `EnhanceKind::Remove`), `enhance_prompt` (the chat: a system prompt
  per `EnhanceKind` plus a picture), `remove_grow`, `fill_rect`, `at_least`, `dilate`,
  `box_blur`, `embed_coverage`.
- `crates/genai/src/fake.rs`: the fake worker the tests use. `Options::segments` are the
  detector's rectangles (fractions of the request), `segments_for` per prompt, `matte`,
  `enhanced_text` (the chat's answer), `echo` (a fill hands the picture back), `color` (a
  fill's paint). After a fill the next detector run finds nothing (`State::removed`).
- `crates/ui-egui/src/dialogs/filter_dialog/generative.rs`: the Split dialog (prompt box,
  `layers` as a segmented row, switches from a fixed list, everything else under Advanced).
- The native worker serves: `sam3.1/segment` (text prompt, `maxDetections`, `threshold`,
  optional `bx/by/bw/bh`; masks come back best first, no scores), `sam3.1/segment-point`,
  `sam3.1/segment-box`, `vitmatte/refine` (picture + trimap), `qwen-2.1/fill` (mask as image
  2), `qwen-2.1/edit`, `krea2-turbo/fill` (AnyPaint), the 2511 fills, `qwen-2.1/outpaint`,
  and the Qwen3-VL chat (`enhance/edit` with a system prompt and a picture). There is no depth
  model yet.

### What to build

**1. Subjects per term.** The prompt is a list: split on commas, semicolons and " and "; a term
`cup:3` asks for at most 3 of that term. Ask SAM 3.1 once per term with `maxDetections` = the
layer budget left; every instance keeps its term as its label. With no prompt: ask `person`
(the budget), then `animal`, then the foreground things the vision-language model lists
(a new `EnhanceKind::Things`, system prompt: "You see a picture. List the separate physical
things in its foreground that a designer would put on their own layers and that are not people
or animals, nearest first, as short noun phrases of one to three words, comma-separated, at most
{k}. Reply with the list only."), each asked with `maxDetections` 2; if nothing at all was
found, Select Subject's fallbacks (`SUBJECT_FALLBACKS`) in turn. People and animals come before
things when the budget is short; within a term, larger first.

**2. Cleaning.** Drop specks (under 2% of the largest instance, or under 0.05% of the frame)
and whole-frame detections (over 90% of the frame). Drop a near-duplicate (mask IoU at or above
0.7 with an earlier instance). A part inside a whole (85% of the smaller mask inside the larger):
same term, or a thing the user did not ask for by name, drop the part; a term the user asked for
(the hat on the man when the prompt says `person, hat`), keep both and subtract the part from
the whole. `group` (`auto`): instances whose soft masks overlap by half of the smaller become one
layer; `touching`: any two whose masks touch; `none`. Note which frame edges each instance
touches: an edge that cuts a subject is never a hole to complete.

**3. Mattes.** As today: ViTMatte inside a band around each SAM mask, else the matte model, else
the classical refinement (`edges`). After every peel, matte again the instances whose masks grew
(section 5), for the completed part.

**4. Order (`order=auto`).** Decide, for every pair that touches (one mask dilated by 2 px meets
the other), who is in front by the contact cue: at each contact pixel take a window of radius
r = max(4 px, 2% of the smaller instance's larger side); the instance that fills less of the
window is the one bulging into the other, so it is in front (an occluder's silhouette is convex
into the thing it hides; the hidden thing is concave around it). Average `share(B) - share(A)`
over the contact pixels; a margin above 0.08 decides, else the lower bottom edge is nearer, else
the larger. For a pair that does not touch and whose boxes nest (80% of the smaller box inside
the larger, the smaller under 30% of the larger's area) the small one is in front (a cup on a
table, a hat on a head). Build the "in front of" graph, drop the weakest edges until it is
acyclic, and take a topological order with the bottom-edge rule as the tie-break. `order=size`
is the old rule (largest lowest), `order=found` the detector's order. Keep the pairwise function
pure and unit-tested on synthetic masks (a disc biting a rectangle; a rectangle under a disc;
two separate discs). Follow-up, not now: a depth model (Depth Anything V2 Small, Apache-2.0)
ported into genai-native makes the median depth inside each eroded mask the primary cue.

**5. Peeling (`complete=true`, the default).** Keep a working picture P, the document's
composite. For each subject, nearest first:

   a. Its layer = P × its alpha (the picture as it is now, the occluders already gone).
   b. The hole = its alpha above 0.1, grown by `remove_grow` of its bounds, inside the picture;
      the request rect is the hole's bounds grown the same way, at least 384 px a side, at most
      about 1 MP (`NATIVE21_REMOVAL_PIXELS`).
   c. Fill the hole in P. Try 1 is the removal (`removal_job`: the edit model told "Remove the
      {name}" with the hole smoothed over), `name` = the term for a thing, "person" for a person
      (never a description by clothes: the edit model then keeps the person and repaints their
      body). Tries 2 and up are the grey-box fill (`background_job`) with a peel prompt: "the
      picture continued behind what stood in the grey area: the same scene, the people and
      things around it completed as they would look with it gone, nothing new added"; the last
      subject (nothing left behind it) uses `BACKGROUND_PROMPT` ("…with nobody there"). Tone
      match the box fill as today (`ToneGrid`).
   d. Check the fill: ask SAM for every remaining term on the filled picture. A detection mostly
      inside the hole (30% of it inside, and at least 3% of the hole) that does not contain 60%
      of some remaining subject's visible mask is the subject painted back: throw the fill away
      and go to the next try. Five tries, then this subject is **incomplete**: its hole keeps
      the picture's own pixels (P is not changed), it is listed in the result, and peeling goes
      on. When no subject at all could be cleared, Split fails with nothing changed.
   e. Update the remaining subjects from the same detections: a detection that contains 60% of
      a subject's visible mask (same term) is that subject completed; its mask becomes the
      union of the two; matte it again where it grew. A subject SAM no longer finds keeps its
      mask.
   f. P = the filled picture; next subject.

   What remains of P is the background. With `complete=false` run as today: one hole, one fill,
   each layer its visible part.

**6. Placing and naming.** Background first (named "Background"), then the subjects back to
front, so the nearest is the top layer. Names: "{Term} {k}" ("Person 3", "Cup 1", "Table") in
stack order; with `name` given, "{name} k/n" as before. The result JSON: `layers`, `count`,
`subjects: [{layer, name, term, bounds, pixels, completed}]`, `incomplete` (names), `order`,
`foundBy`, `template`, `seed`, `fillMs`, `ms`, `edges`. Each layer remembers the run
(`GenerativeInfo`).

**7. Parameters and the dialog.** `layers` 2..12 (the segmented row shows 2, 3, 4, 6, 8, 12),
`prompt` (the list), `complete` (bool, on; the switch "Complete hidden parts"), `order`
(auto|size|found), `group` (auto|touching|none), `edges`, `seed`, `name`. Progress text names
the subject being peeled ("Filling behind Person 3 (4 of 9)").

**8. Smoke and haze (`haze`, off; phase 3).** Estimate the transmission t with the dark channel
prior (patch 15 px, airlight A = the brightest 0.1% of the dark channel, ω 0.95, guided-filter
refinement), make a "Haze" layer of colour A with alpha 1 − t, and dehaze the background by
(I − A(1 − t)) / max(t, 0.1). Place it above the background, below the subjects, only when the
mean of 1 − t over the picture is above 0.08; otherwise no layer. Measure on the café photo and
on a clear photo (no layer expected).

**9. Tests against the fake worker.** Extend `Options` with `occlusion: bool`: the fake's
`segments` are then full (amodal) rectangles, front first; the detector returns each remaining
one minus the remaining rectangles before it; a fill whose request mask covers half of a
rectangle's visible area removes it. Then test: the terms parser; the pairwise order cue; the
topological order; a peel of two overlapping rectangles where the back one's layer gains its
full rectangle; the painted-back fallback (`echo`) leaving the picture's pixels and reporting
`incomplete`; the refusal when nothing clears; props found through the VLM list
(`enhanced_text: "cup, table"` with `segments_for`); duplicate and part dropping; the names;
`complete=false` as the old behaviour; the parameters' validation. Keep the existing tests
passing or rewrite them for the new names.

**10. The in-app protocol.** For each of the café scene (layers 12, default prompt), the
portrait (layers 3) and the 12 MP portrait (layers 2), three seeds: time, layer count, how many
subjects completed, SAM's `person` count on the background (must be 0), and a contact sheet of
the layers (each over mid-grey, with the background) saved beside the run. Look at every sheet.
A fill that painted a person back is a defect even when the check caught it: note the seed.
Report warm times (second run) and the first run separately.

### Acceptance

- Café: at least nine layers; every person's layer whole where the table or another person hid
  them; the table whole where the cups stood; no person on the background; nothing added that
  was not there; the pairs (old man, man with glasses), (man in the black hat, far-left man),
  (table, each person behind it), (cup, table) ordered right; under 90 s warm on the RTX 5090.
- Portrait: woman and dog on separate layers, 5 of 5 seeds clean, under 15 s with ViTMatte.
- 12 MP portrait: 3 of 3 seeds clean or refused, never a repainted person.
- `cargo xtask ci` green; the devlog and benchmarks updated; the README's Split rows true.

### Follow-ups the prompt may queue, in order of value

1. A depth model in genai-native (Depth Anything V2 Small, Apache-2.0; ONNX is not wanted,
   port it to candle as ViTMatte was) as the primary order cue and a `depth` result per layer.
2. The vision-language model as a tie-break for touching pairs the contact cue cannot decide
   ("which is nearer the camera, the one marked 1 or 2?").
3. Krea 2 AnyPaint as a peel fill (it reads the whole scene; 1–3 s a fill) once its prompt
   writer can say "with the {name} gone".
4. The faint outline on the grey-box backdrop fill (2 of 3 seeds on the 12 MP portrait).

## 3. Status

Landed 2026-11-03 (the thread's PR; devlog entry of the same day):

| Item of §2 | State |
|---|---|
| 1 Subjects per term (list prompt, `cup:3`, people → animals → things Qwen3-VL lists → fallbacks) | done (`find_subjects`, `list_things`) |
| 2 Cleaning (specks, whole frame, IoU duplicates, parts, touching groups) | done (`clean`) |
| 3 Mattes per subject, re-matted after completion | done (`matte_all`) |
| 4 Order (contact bend, bottom edge, size, nested boxes; topological sort) | done (`generate_split_order.rs`); no depth model yet |
| 5 Peeling nearest first with a fill, a re-detection check and growth into the hole | done (`fill_hole`, the peel loop in `run`) |
| 6 Placing and naming ("Background", "Person 1" back to front, "Cup") | done |
| 7 Parameters (`complete`, `order`, `group`, `layers` to 12) and the dialog | done |
| 8 Smoke and haze as a translucent layer | not started (phase 3) |
| 9 Tests against the fake worker's occlusion model | done (8 Split tests + 4 order tests) |
| 10 In-app protocol on the café photo | see the devlog entry for the numbers |

Follow-ups stand as listed above; the depth model first.
