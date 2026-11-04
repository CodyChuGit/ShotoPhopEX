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
3. Krea 2 AnyPaint as a peel fill: tried 2026-11-04 as the first try of every hole and
   withdrawn the same day. It invented people (a copy of the woman beside her, 3 of 3 café
   seeds) and rooms (2 of 3 on images.jpg), and every switch between Krea 2 and the 2.1 family
   reloaded the other model, about 60 s a split. A retry needs its prompt writer ("with the
   {name} gone"), one model family for the whole split, and the checks now in
   (`fills_own_work`, `kept_share`).
4. The faint outline on the grey-box backdrop fill (2 of 3 seeds on the 12 MP portrait).
5. The grey-box 2.1 fills over large holes (the left half of the café background, every
   round: a blur under the full tone match, the grey box itself under the capped one). Round 6
   fills a big hole in strips and falls back to the classical patch fill; still to try: a final
   whole-background removal over all the holes once every subject is out, kept only if it
   passes the checks; or a fill model that keeps texture. Round 6 measured the strips: the
   boxes gone from six of eight backgrounds, but the room fogged rather than continued, seams
   between the strips, an invented near table, small invented people and rooms (caught since
   round 7 by the detector over every pixel the fill touched; round 7 still passed figures
   15–40 px tall, under the detector's floor in the whole frame, so since round 8 the detector
   is shown the touched pixels tile by tile), and 5.8–7.9 s a strip generation (5.3–5.8 in
   round 7). Round 7's blanks are gone and the seams softer; the room is still fogged.
6. The classical patch fill is now the last try of every hole (round 6); a cup refused five
   times no longer stays in the background. Still to decide: whether a big person's hole should
   take the patch fill at all, or stay in the background with `incomplete` saying so. Round 6:
   the patch ran three times (the near table twice, the striped man once) and was refused each
   time; since round 7 a thing's patch is judged by the pixels and for people only, the
   detector seeing the thing again in the copied surroundings by construction. Round 7: the
   patch accepted once (a sliver on a plain wall), refused four times as blank; the fedora man
   refused six times and left in the background, two of his strips only for "a table in the
   hole that was not there". Since round 8 such a try is taken as the last resort when every
   try of the hole is refused (never a person, an animal, a blank or a shape).
7. Speed: of a 131–163 s café run only 55–85 s is filling; the rest is detection, matting and
   a SAM check after every fill. Cache the detector per picture and check only the kinds that
   stand near the hole. The acceptance is under 90 s warm. Round 6 with the strips: 194–333 s,
   the strips 110–180 s of a café run. Round 7: 173–374 s, 26–52 generations a café run at
   4.5–5 s each, a hole that ends refused costing its whole ladder (17 generations for the
   fedora man); round 8 drops one strip pass from a big hole's ladder but adds the detector's
   tiles to every accepted fill (196–496 s; the scan before any fill 3–6 s more). Round 8's
   engine log shows SAM reloaded 11–27 times a run: the worker swaps it out for the fill model
   and loads it again for every check (a genai-native item: keep both resident, or batch the
   checks). Since round 9 the result's `detector` counts the requests, the tiles among them
   and their time: round 9 measured 36–56 s of a warm café split (89–119 requests, 30–38
   tiles), 77 and 97 s at 16 layers, 112 s cold, and 3.1 s of images.jpg's 6 (its six tiles
   were the scan before any fill and the finished-background look); since round 10 a plain
   backdrop is not scanned tile by tile there.
8. The woman's detached dark block below the table edge (her legs under the table, round 2–4):
   once the table is a subject, a part of a detection the table separates from the body goes.
9. Keep the rejected fills on request (a `keepTries` option), so a refusal can be seen and not
   only read in `tries`: done since round 8 (hidden layers under the background, "Person 2:
   strips refused (…)", their ids in `refusedLayers`).
10. The sweep's debris and lost subjects: round 5 counted a sliver of the fedora man's
    trousers below his mask and a person inside a framed photograph as subjects; round 6's
    debris rule (a find mostly inside what the fills painted) then dropped real subjects whose
    holes had grown, and the sweep stopped at seven of eleven. Since round 7 the first pass's
    list is authoritative (a subject stays pending until it is cut, re-found by overlap or cut
    by its first mask) and only new finds face the debris rule. Round 7: all eleven cut at 12
    layers in every seed, with the peel's layers, order and masks and no debris; fifteen of
    sixteen at 16 layers (round 8 the same, and at 16 layers a near-table chunk found on a
    later pass, stacked below everyone). Still to do: a later find much smaller than the first
    pass's of its kind, or inside a thing found earlier, is not a subject.
11. ViTMatte on the first split of a session: it returned nothing cold (round 5) and the split
    fell back to the whole-frame matte; asked once more now (round 6's cold edges are
    ViTMatte's), but the worker should load it at launch with the rest of the warm-up.
12. The strips' seams and the near table: in every round-6 peel run the strips of the front
    people's holes painted over the near table with an invented one, and the seams between
    strips show as steps and hard rectangles (blended over 48 px since round 7: clearly softer
    in round 7's sheets). The invented near table stands in every round-7 run where the fedora
    man left. A strip plan that follows the hole's shape and keeps what stands beside it, or
    the whole-background pass of (5).
13. Ghosts: a soft dark blob where the woman's head was passed four round-7 backgrounds
    (under a quarter of the hole's cells); since round 8 a cluster of six joined shaded cells,
    two per cent of the hole, is refused as a shape over part of the hole. Round 8: the blob
    gone from four backgrounds; the rule refused four tries, all at 16 layers, unverified by
    eye (no `keepTries` on those runs). Round 9: the same four, and soft smudges at the table
    in two sweep seeds under the 30 %.
14. The detector's floor: figures 15–40 px tall in the 1800-px café frame were under the
    0.05 % floor or missed outright; since round 8 the pixels a fill touched are shown tile by
    tile (640 px a side at least), and whoever was in the picture before any fill (the man in
    the framed photograph) is exempt. Round 8: the tiny figures gone from all ten café
    backgrounds, but the detector shown a tile calls fog a person (the striped man's strips,
    81×318 px of plain fog) and hands back masks over the whole tile (the portrait failed every
    split), the photograph man was half behind the waiter and so not exempt, and a part of a
    person waiting their turn counted as new. Since round 9: a detection with no more detail
    than fog is no figure, a tile's detection counts only under half its side, a figure there
    before and completed by at most twice its pixels is exempt, and a part of a listed person
    behind is that person. To measure again: false alarms by the kept refused fills, the fog
    rule against a real small figure painted softly. Round 9: the fog rule cleared the striped
    man and the portrait splits again (four of four), the tile-size rule held, nothing dismissed
    as fog was a figure; but the two other rules did not take: the completed part of a person
    waiting their turn comes back as a figure of its own, holding none of their mask, and the
    photograph man was mostly behind the waiter, more than twice his sliver. Since round 10 a
    completed part joined to a subject behind is theirs (`fragment_of`, grown into as well), what
    the cut subject hid of someone there before is any size (`was_there`), and each refusal
    records its detection's bounds (`at`) beside the figures found before any fill (`already`).
15. The edge person (round 9, seed 1 in both modes): every fill of his hole at the frame's edge
    paints an invented room with people (refused rightly, by the kept fills), the removal and
    the patch keep him; seeds 2 and 3 clear him. To try: the hole's crop padded by reflection
    so the fill sees context on every side, or more removal seeds for a hole at the edge.

## 3. Status

Landed 2026-11-03 (the thread's PR; devlog entry of the same day):

| Item of §2 | State |
|---|---|
| 1 Subjects per term (list prompt, `cup:3`, people → animals → things Qwen3-VL lists → fallbacks) | done (`find_subjects`, `list_things`); the budget counts the subjects kept, not the detections; the things one of each first, then up to three of each, so the table comes before a third cup (2026-11-04) |
| 2 Cleaning (specks, whole frame, IoU duplicates, parts, touching groups) | done (`clean`) |
| 3 Mattes per subject, re-matted after completion | done (`matte_all`); a sweep mattes each subject at its turn, the others' bodies kept out of its band (ViTMatte's `only`) |
| 4 Order (contact bend, bottom edge, size, nested boxes; topological sort) | done (`generate_split_order.rs`); no depth model yet |
| 5 Peeling nearest first with a fill, a re-detection check and growth into the hole | done (`fill_hole`, the peel loop in `run`); a fill that paints anyone new is rejected (`fills_own_work`), a person's silhouette the fill kept is refused by its pixels too (`kept_share`: round 4's front-left man came back almost pixel for pixel and the detector check passed him), a thing is judged on its own spot, and every try's verdict comes back in `tries` (2026-11-04); a fill is also judged against the ring around its hole (`blank_or_ghost`: blank where the picture has texture, or a dark or light shape over the hole), a big hole's grey box is filled in strips (`fill_in_strips`) and the classical patch fill is the last try when every fill is refused (`patch_fill`; round 5's big holes came back as the grey box, 2026-11-04); since round 7 the detector is asked after every fill for anyone painted into the pixels the fill touched (`someone_new`, no speck rule, down to 0.05 % of the frame: round 6's strips invented small people and rooms with every try accepted), the blank and ghost checks run per 32-px cell, the removal is the first try of every hole and a plain ring never gets strips (images.jpg's backdrop went to strips in round 6), and `leftBehind` lists the people still on the final background (2026-11-04); since round 8 the detector is shown the touched pixels tile by tile after every fill the frame-wide check passes (`anyone_small`: round 7's invented figures were 15–40 px tall), whoever was in the picture before any fill is nobody's work (`already_there`), a cluster of shaded cells is a shape over part of the hole (`ghost_cluster`), a thing painted where none was is taken as the last resort when every try is refused, and `keepTries` keeps the refused fills as hidden layers (`refusedLayers`) (2026-11-04); since round 9 a detection with no more detail than fog is no figure (`no_more_than_fog`), a tile's detection counts only under half its side, a figure there before and completed by at most twice its pixels is exempt, a part of a person behind is that person, and the result's `detector` counts the requests, tiles and time (2026-11-04); since round 10 a person's completed part seen on its own and joined to a subject behind is that subject's, in the check and in the growth (`fragment_of`), someone there before who stood against the cut subject is completed by any amount (`was_there`), each refusal records its detection's bounds (`at`), the result lists who was there before any fill (`already`), and a plain backdrop is not scanned tile by tile (2026-11-04) |
| 6 Placing and naming ("Background", "Person 1" back to front, "Cup") | done |
| 7 Parameters (`complete`, `sweep`, `order`, `group`, `layers` to 20) and the dialog | done (`layers` raised from 12 on 2026-11-04: the café's seven people, three cups, cards and table are twelve subjects) |
| 8 Smoke and haze as a translucent layer | not started (phase 3) |
| 9 Tests against the fake worker's occlusion model | done (45 Split tests + 4 order tests; the fake can paint someone in, `appears`, a thing for one term, `appears_for`, a figure seen only in a part of the picture, `appears_in_crops`, and overlook a subject a fill left in place, `overlooked`) |
| 10 In-app protocol on the café photo | rounds 1–9 in the devlog (2026-11-03/04); round 10 (a person's completed part joined to them, what the cut subject hid of someone there before, `at` and `already` in the result, no tiles on a plain backdrop) pending |
| 11 Sweep (`sweep`: find the subjects, cut the nearest, fill, look again until nothing is left; the owner's framing, 2026-11-04) | done (`sweep_subjects`); round 5: 13–27 % slower than the peel, the best room continuation of any round on one seed, but fill debris counted as people and the first pass's depth order lost on later passes; round 6's fix (the first pass's order for whoever is found again, a kind's cap over all passes, debris dropped) lost real subjects instead (seven of eleven cut, the striped man above the fedora man again); since round 7 the first pass's list is authoritative: a subject stays pending until it is cut, each pass re-finds it on the filled picture (`same_subject`) or cuts it by its first mask, every pending subject is protected from the fills, and only new finds face the debris rule; rounds 7 to 9: all eleven cut at 12 layers in every seed, the peel's layers and order, no debris (a near-table chunk found late at 16 layers in rounds 8 and 9) |

Follow-ups stand as listed above; the depth model first.
