# How ArtCraft delivers generation, and how PhotoshopEX should

Written 2026-10-10/11 from getartcraft.com and a local read of the public `storytold/artcraft`
repository (2026-10 head: the React frontend under `frontend/`, the Tauri/Rust desktop under
`crates/desktop/artcraft`). This is about the **UI and the experience**: what the user sees,
touches and gets back. ArtCraft's billing (credits) is left out on purpose; PhotoshopEX runs on
local compute, so where they print a credit cost we print a time. A write-up only; nothing in the
app changed. §6 says what to build, in order, with wireframes.

## 1. The shape of the app

ArtCraft is **one window, one top bar, one page at a time**. The top bar (56 px, dark) holds:

- **Home** (house icon) plus one tab per app: Create Image, Create Video, Create Audio, Image
  Editor (2D), 3D Stage. Hovering Home opens a two-column quick menu, *Create* and *Edit*, listing
  every app with an icon tile, a one-line description and a badge (NEW, BEST, SOON, BETA).
- In the middle: the scene title (editable) when a scene is open.
- On the right: **Upload images**, **Library** (the gallery modal), the **Task Queue** button with
  a count badge, Settings, the profile menu. (Their credits coin sits here too; skip it.)

The **Home page** is a card grid, "What will you craft today?", in two sections: Create (Image,
Video, Audio, Image to 3D, 3D World) and Edit (Image Editor, Remove Background, Angles,
Background Change, Storyboard, Moodboard, Video Editor, frame extractor, watermark removers…).
Cards are quiet: a tinted icon tile per app, uppercase mono label, badge, a sentence, an arrow.

Pages are lazy and independent; the 3D engine mounts only on its tab. Everything that must keep
running (job polling, completion events, sounds, toasts) is mounted in the shell, so a generation
finishes and announces itself whatever page you are on.

## 2. The Create pages (Image, Video, Audio)

One shared shell, `DesktopCreatePageShell`:

- An **empty state** that reads like a magazine spread: a big display heading ("Create Image"),
  a subtitle ("Add a prompt, then generate"), four thin corner registration marks. It stays up
  as a splash while the library loads and fades out (700 ms) the moment the feed has content.
- The **generation feed** fills the page and scrolls: in-progress tiles first, then failed rows,
  then completed items, newest first, merged with the library's history so the page is also the
  history. A **grid / list** toggle and an **autoplay** toggle for videos live in the top bar; a
  **select** toggle turns tiles into checkboxes with a floating "3 selected · Download" pill that
  saves each file to a chosen folder with the naming convention from Settings.
- The **prompt box floats at the bottom centre**, max width about 1000 px, a glass panel, with a
  gradient fade behind it so the feed stays legible underneath. The feed pads itself past the
  measured prompt-box height, so the box never covers a result.
- Bottom right: a Help button (tutorials, news, Discord, GitHub, YouTube) and a cost calculator
  (skip).

### 2.1 The prompt box, piece by piece

Every page's prompt box is the same object, configured by the **selected model's capabilities**
(`ImageModel`: `canUseImagePrompt` + `maxImagePromptCount`, `aspectRatios`, `resolutions`,
`qualityOptions`, `maxGenerationCount` / `defaultGenerationCount`, `maxPromptLength`, tags that
say which editors the model appears in). Reading left to right, top to bottom:

1. **Reference deck** (left of the text): a fanned stack of up to three tilted thumbnails with a
   "+N" badge, which expands on hover into a panel: drag to reorder, hover to remove, click to
   preview, "Clear all", and a "+" menu with *Upload* and *Pick from library*, each showing its
   limit ("2/9"). Drag-and-drop or paste anywhere on the box adds references; the overlay says
   what it accepts. The deck is hidden entirely for a model that takes no references.
2. **The prompt**: a one-row textarea that grows, placeholder "Describe what you want in the
   image...", a character counter against the model's limit (turns red past it), a fullscreen
   button that opens the same editor as a modal with the deck always expanded, and a collapse
   chevron under the box. `@` opens a **mention menu** of saved characters and media; a pick
   becomes an inline chip with a thumbnail. Enter submits only if "Enter to generate" is on
   (Shift+Enter is always a newline).
3. **Options row**: the **model pill** (creator icon + name) opening a rich picker, then **aspect
   ratio**, **resolution** (1K / 2K / 4K), **quality** (GPT models), each a popover with an icon
   and a title, and each present only when the model supports it.
4. **Right end**: the **count** picker (1–4, or the model's fixed counts), **Clear all** (asks
   first when references are attached), and the **Generate** button (icon button on the create
   pages, a labelled button in the editors) that carries the cost. Generate blocks itself for up
   to ten seconds after a click and the page adds a pending tile at once.

The **model picker** groups models by family when there are eight or more ("Seedance", "Veo",
"Nano Banana"…; a setting flattens it), newest version first inside a family, each row with a
creator icon, a name, a one-line description ("Stunning style and quality") and badges such as
"10 sec." or "45 sec." (their expected wait). A row with several providers shows a provider
submenu with icons. The pill in the box shows only the creator icon and the name.

### 2.2 What comes back

- A **pending tile** the size of a result: shimmer background, spinner, "35 %" and "~ 20s left"
  (a fake progress bar timed from the model's `progressBarTime`), the prompt clamped to three
  lines, the creator icon and model name, a thin progress bar, and for batches a banner
  "Generating 4 images · Results may appear one at a time". Hover: **Recreate**.
- A **failed row**: a red status chip ("Failed", "Cancelled by provider"), a dashed-underlined
  reason with the message in a tooltip ("Images with faces are not allowed", "Text prompt
  violates content policy", "Generation failed"), the prompt, the time, Dismiss, Recreate.
- A **completed card**: the image at its own aspect ratio (capped at 5:7 so portraits do not
  dominate the masonry), a media-kind badge, the creator icon and model name, a persistent
  "Last viewed" badge on the item last opened, and on hover a pill of quick actions:
  **Recreate** (re-seed the page from the prompt record), **Make Video** (seed the video page
  with this image), **Share** (copy link), **Download**. Click opens the lightbox.

A result is always **a batch**, four images by default, rendered as four cards.

### 2.3 The Task Queue

A top-bar button (spinner while anything runs, a list-check icon otherwise) with a badge =
in-progress + unread completed. The popover is 400 px wide and has three sections:

- **In Progress**: 72 px thumbnail (the first reference image, dimmed, with "+2" when there are
  more), "Generating Image...", "Model — Provider", percent, a bar, "~ 45s left" or "Almost
  done...", the prompt in a marquee with a copy button. Dismiss appears after five minutes (a
  stale task).
- **Failed**: the same card in red with the typed reason; "Clear failed".
- **Completed**: thumbnail, kind, model, time ("Oct 10, 9:14 PM"), prompt, copy; click opens the
  lightbox; "Clear completed".
- Footer: **Show all** opens a modal with the same lists and Clear completed / Clear stale /
  Clear failed / **Remove all** (a confirmation each).

Every enqueue, completion and failure can play a **sound** (five separately choosable sounds in
Settings › Alerts, with preview), and errors raise a toast at the top right.

### 2.4 The lightbox and the Library

The **lightbox** shows the media large with prev/next through the feed order, and a side panel:
the author, the **Prompt** block (clamped to four lines, "See all", **Copy**), tags you can edit,
and actions: Recreate, Make Video, **Edit** (open in the Image Editor), **Image to 3D**, Share
link, Info, Delete.

The **Library** (gallery modal) is the whole account: filter chips All Assets / Image / Video /
Audio / Mesh / Splat / Uploaded, **folders** with colours and stars, tags, multiselect, refresh,
"New folder". Items **drag out of the Library onto a canvas** (a drag ghost follows the pointer
and the modal hides itself during the drag). The tile menu (hover ellipsis or right-click) is:
View image, Edit image, Add to Folder ▸, Remove from folder, Delete.

## 3. The Image Editor (the 2D canvas)

A Konva stage in the middle, and around it:

- **Left side toolbar**: Select (1), Shape (2: rectangle / circle / triangle, a colour picker for
  the selected shape), Brush (3: colour, size, opacity), **Mask** (4: the inpaint brush, shown
  only when the model supports masks), Eraser (5), Upload image, Canvas background colour,
  Undo / Redo, Delete. Key numbers are printed as corner badges on the buttons and update when
  the user remaps them. Ctrl + / Ctrl − change the brush size.
- **Inpaint toolbar** at the top centre when the mask tool is active: Add / Erase and a size
  slider with − and +.
- **History Stack** on the right: every generation is an **image bundle** of candidate tiles;
  clicking one makes it the new base image (the drawing state is remembered per base image, so
  going back restores your strokes); pending bundles show placeholder tiles with a count; tiles
  can be removed; "Clear".
- **Prompt editor** at the bottom: the same prompt box with an **Edit / Inpaint mode** switch
  (decided by the model's tags), aspect **Auto / Wide / Tall / Square**, resolution, **Fit** (frame
  the canvas), count, references, Undo / Redo, and a system-prompt toggle for models that need
  it. Generate composites the canvas (and the mask when inpainting) in a worker, enqueues, and
  the result arrives in the History Stack filtered by the request's own subscriber id.
- A **3D model overlay**: drop a mesh on the canvas, pose it, render it to pixels.
- A **blank canvas modal** with size presets; a **base image selector** (upload or library).
- Holding Ctrl for a moment shows a **shortcut cheatsheet**; a right-click **context menu** on
  nodes.

Other editors reuse the pattern: **Angles** (upload or pick an image, drag a camera around an
**orbit sphere** that snaps to 45° rotations, tilts of −30/0/30/60 and zoom 0/5/10, with sliders,
Generate, results in a History Stack, Download); **Remove Background** (an entry card for upload
or library, a reveal animation, hold-to-compare the original, a processed list, Download);
Storyboard (shots on a timeline with a metadata editor); Moodboard (a Figma-like board with
rating keys 0–5 and a grid).

## 4. Settings and small things that make it feel finished

- **General**: Enter to generate; group models by family; keep the shortcut cheatsheet open.
- **Downloads**: auto-download results; a download directory; a filename convention
  (`{model}_{date}` or the ArtCraft default).
- **Appearance**: five themes (Default gray, Light, Black, Aurora, Sunset).
- **Keybinds**: presets (Gamer, Blender-like), remap any action with a capture field, search.
- **Alerts**: sounds on and off, one sound each for enqueue ok / enqueue failed / generation
  ok / generation failed / file deleted, with preview.
- Copy and type: uppercase mono "HUD" labels for section titles, 3 px radii, glass panels,
  display serif for headings, dim secondary text, hover-revealed actions, confirmations only for
  destructive clears, toasts for errors, progress always visible somewhere (tile, queue badge,
  spinner in the top bar).

## 5. The principles behind it

1. **Everything is a job you can walk away from.** Enqueue returns at once, a pending tile appears
   where the result will land, the queue badge counts it, a sound announces it, and the result is
   there when you come back, on any page.
2. **One prompt box, everywhere, shaped by the model.** Controls appear and disappear with the
   model's capabilities; nothing is greyed out "because this model cannot".
3. **Results accumulate; nothing is lost.** The create page is the history; the editor keeps a
   stack of candidates per step; the Library keeps everything with folders and tags; every result
   knows its prompt and can be **recreated** in one click.
4. **A result is a batch.** Four by default; you pick.
5. **Hover reveals, click commits.** Cards are clean until you hover; destructive actions confirm.
6. **The model is a named thing with a face.** Creator icon, family, a wait-time badge, a
   one-line character ("Fast af", "Stunning style and quality").

## 6. PhotoshopEX: the same experience inside a Photoshop-style editor, on local compute

PhotoshopEX is an editor with documents, layers, a dock and a title bar, not a page-per-app
studio. The ArtCraft experience maps onto it as **three surfaces** plus a few touches, and the
"credit" everywhere becomes **a time estimate from our own measurements**.

### 6.1 The Generate panel (the prompt box)

A dockable panel (Window › Generate; also the home of today's task bar when nothing is selected),
and the task bar under a selection becomes a compact form of it.

```
┌ Generate ─────────────────────────────────────────────────────┐
│ [▣▣▣+2]  Describe what you want in the image…                 │
│          …                                            87/4000 ⤢ │
│ ─────────────────────────────────────────────────────────────── │
│ [◆ Krea 2 Turbo ▾] [▭ 3:2 ▾] [1K ▾]        [4 ▾] [✕] [✨] [Generate ~24 s] │
│ Mode  ( Image | Fill | Edit | Expand )     into: ( new document | layer ) │
└───────────────────────────────────────────────────────────────┘
```

- **Reference deck**: layers of the open document, files, or Library items; the cap comes from
  the template (Qwen-Image-2.1 up to 10, Krea 2's style reference 1, the 2511 edit templates 0).
- **Model pill → picker grouped by family** (Qwen-Image-Edit 2511 · Qwen-Image 2.1 · Krea 2 ·
  Flux…), each row with the family icon, the template name, a one-line character and a badge of
  **measured seconds on this machine** ("~16 s", from `benchmarks.md`, scaled by request size and
  count), an "installed" or "files missing" mark, and the licence tag (research models only with
  the opt-in).
- **Aspect and size** for Image, hidden for Fill/Edit (the selection decides). **Count** 1–4.
  **✨ Enhance** (shipped). **Generate** shows the ETA, not a coin.
- **Mode** is the command: Image, Fill (needs a selection), Edit, Expand; **into** says where the
  result lands (a new document, or layers above the active one). Enter to generate is a
  preference.

### 6.2 The Generations panel (the feed and the queue)

A dockable panel listing every generation of this session and, from the persistent job store,
earlier ones; a grid / list toggle; a badge in the title bar that counts running jobs.

```
┌ Generations ───────────────── [⊞ grid | ☰ list] [select] ┐
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐             │
│ │▒▒▒▒▒▒▒▒│ │        │ │        │ │        │  ← pending   │
│ │ 38 %   │ │  ✓     │ │  ✓     │ │  ✓     │    tiles,    │
│ │~10 s   │ │        │ │        │ │        │    real %    │
│ └────────┘ └────────┘ └────────┘ └────────┘             │
│  Generating 4 images · Qwen-Image-Edit 2511 Lightning     │
│  "a red kite over the harbour"                            │
│ ✗ Server unreachable · "stormy sky" · Edit · 9:14 PM  [↻][×]│
│ ┌────────┐ ┌────────┐ …  hover: [Place] [↻ Similar] [⤓] [⎘] │
└──────────────────────────────────────────────────────────┘
```

- **Pending tiles** with **real progress** (ComfyUI's WebSocket) and an ETA from the template's
  measured time; the prompt, the template name, "Generating 4 · results appear one at a time".
- **Failed rows** with typed reasons we already produce (server unreachable, model files missing,
  out of memory, cancelled, the server's error), Dismiss, **Recreate**.
- **Completed cards**: the result layer's thumbnail; hover actions **Place** (as a layer in the
  active document, or open as a document), **Similar** (Generate Similar, which already exists),
  **Save to disk** (the Downloads convention `{model}_{date}`), **Copy prompt**; click opens the
  layer and the Properties panel's generative section (prompt as typed, the enhanced prompt, seed,
  template, time: our `generate.info`). A **Last viewed** mark.
- Clear completed / failed, and a **Remove all** with a confirmation.

### 6.3 The Variations strip (the History Stack)

Fill, Edit and Image with count > 1 already make candidate layers and a 1/3 switcher. Show them
as ArtCraft shows a bundle: a strip of thumbnails at the edge of the canvas (right or bottom,
a preference), the shown one highlighted, pending placeholders with a count, click to show, ×
to drop a candidate, "Keep" to flatten the choice. Generate Similar appends a new bundle to the
strip. The strip is the thing a user looks at while deciding; the Layers panel is the truth.

### 6.4 Smaller touches, same order of value

- **Generate hub**: a Home-style card page (a modal or the welcome screen) with Create (Image)
  and Edit (Fill, Edit, Expand, Remove Background, Split into Layers, Angles) cards, badges NEW
  and SOON, one sentence each. Also reachable as Help › Generative features.
- **Angles**: a dialog with ArtCraft's orbit sphere (snap to 45°, tilt, zoom) over the 2511
  angles LoRA when its files are installed (the model exists in their catalogue as
  `qwen_edit_2511_angles`; a permissive LoRA to verify in `models.md` first).
- **Sounds and toasts**: a sound on completion and on failure (Preferences › AI Integrations ›
  Alerts, off by default), the status-bar toast we have.
- **Enter to generate**, **group models by family**, **auto-save results** with a naming
  convention, in Preferences › AI Integrations.
- **Keyboard**: Ctrl+Enter generates from the panel; Esc cancels the running job (exists).
- **Library**: not a separate store. The document's layers and the OS file system are the
  library; the Generations panel's history plus "Save to disk" cover the rest.

### 6.5 Build order

| Step | What | Why first |
|---|---|---|
| 1 | The persistent job store and the **Generations panel** with pending, failed and completed items, Place, Similar, Save, Copy prompt | It is the experience's backbone: jobs you can walk away from, results that accumulate |
| 2 | The **Generate panel** (deck, prompt, pill, aspect/size, count, ETA on the button, modes, "into") and the task bar as its compact form | One prompt box everywhere, shaped by the template |
| 3 | Template **capability flags** and the **family-grouped picker** with measured-time badges and installed marks | The panel reads them; the picker gets its face |
| 4 | The **Variations strip** over the candidate layers | Choosing is the moment that matters |
| 5 | Generate hub, sounds, Enter to generate, auto-save convention, Angles | Finish |

Each step keeps the fork's rules: commands first (`generate.jobs`, `generate.place`, panel state
over `ui.set` / `ui.inspect`), tests against the fake server, no shipped weights, research models
behind the opt-in.

## Appendix: what ArtCraft charges for

Only generation through ArtCraft's own hosted provider, in credits (plans from $8 to $48 a
month; refunded on failure; bring-your-own Fal, Replicate and Google keys or Midjourney and Grok
logins bypass them). Everything in this document is free in ArtCraft and free here.

## Sources

- getartcraft.com: home, `/apps`, `/apps/photocraft`, `/faq`, `/support`, `/tutorials`,
  `/pricing` (2026-10-10)
- github.com/storytold/artcraft: `frontend/apps/artcraft/app/src` (`pages/MainApp.tsx`,
  `components/signaled/TopBar/*`, `components/generation-feed/*`, `pages/PageImage`,
  `pages/PageDraw`, `pages/PageAngles`, `pages/PageRemoveBackground`, `pages/PageApps`,
  `config/appMenu.tsx`), `frontend/libs/components/{promptbox,model-selector,generation-list,
  gallery-modal,lightbox-modal,pagedraw,settings-modal,keybinds,help-menu,popover}`,
  `frontend/libs/model-list`, `crates/desktop/artcraft/src/core/commands/{generate,task_queue,
  app_preferences}`, `_database/sql/artcraft_migrations/…create_tasks_table.sql`
