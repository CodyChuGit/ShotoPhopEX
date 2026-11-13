# Unresolved risks: items for legal or maintainer review

Reviewed 2026-11-07. Each item needs a person's decision: either a lawyer's view (**LEGAL**) or the
maintainer's call (**MAINTAINER**). Nothing here is a legal conclusion. "Release" means the next
published ShotoPhopEX release; v0.1.0 is already public.

| ID | Item | Type | Severity | Blocks the next release? |
|---|---|---|---|---|
| U-1 | MSI installs as upstream PhotoCraft's product | MAINTAINER | High | Mostly fixed for 0.1.2 (own UpgradeCode, publisher, folder); exe name and ProgIDs still shared, needs a Windows upgrade test |
| U-2 | MOSS Edit adapter has no stated licence, and the 12/16 GB pack downloads it | LEGAL + MAINTAINER | High | No, but decide before promoting the low-VRAM pack |
| U-3 | In-app acceptance replaces the licensor's own gate for repacked models | LEGAL | Medium | No |
| U-12 | Native-engine ports were written against GPL-3.0 reference code | LEGAL | Medium | No |
| U-14 | Name and icon resemble Adobe Photoshop's | LEGAL | Medium | No |
| U-15 | ArtCraft name used as a link in the app | MAINTAINER (+ LEGAL) | Medium | Fixed for 0.1.2 |
| U-16 | NVIDIA EULA distribution conditions | LEGAL | Medium | No |
| U-19 | Copyright in AI-assisted contributions | LEGAL | Medium | No |
| U-13 | Apache-2.0 §4(b) change notices in `qir` | MAINTAINER | Low | No |
| U-4 … U-11, U-17, U-20 … U-22 | Smaller provenance and documentation items | as listed | Low | No |

---

### U-1. The MSI installs as upstream PhotoCraft's product (BLOCKER for the next MSI)

**Status (2026-11-13, owner chose to remove the branding).** From 0.1.2 the MSI has its own
UpgradeCode `{7A30FAE6-70CF-4CB3-A7C7-384E37ECCBB2}`, Manufacturer "ShotoPhopEX", install folder
`Programs\ShotoPhopEX` / `Program Files\ShotoPhopEX` and remembered-folder key
`Software\ShotoPhopEX`; the vendored craft-update matches Publisher "ShotoPhopEX". An `<Upgrade>`
row removes ShotoPhopEX 0.1.0 and 0.1.1, which used upstream's UpgradeCode. Residual: that row
matches any product under the old code at 0.1.0–0.1.1, which would include an upstream PhotoCraft
0.1.x install from early October 2026 (upstream is on 0.23); the exe is still `photocraft.exe`, so
the `App Paths` key and the ProgIDs `Photocraft.Image` / `Photocraft.PhotoshopDocument` are still
shared with upstream. Needs a real upgrade test from a 0.1.1 install on Windows before release.
The text below is the original finding.

**Evidence.** `packaging/windows/photocraft.wxs` and the v0.1.0 MSI (`msiinfo export … Property`):
`UpgradeCode {D9660F32-2D0A-4ABE-B73D-17BE25F1542C}` is upstream PhotoCraft's own (same value in
`storytold/photocraft` main); `Manufacturer` is "Learning Machines LLC" (upstream's company); the
install folder is `Program Files\ArtCraft\PhotoCraft` (or `%LOCALAPPDATA%\Programs\ArtCraft\PhotoCraft`);
file associations use upstream's ProgIDs (`Photocraft.Image`, `Photocraft.PhotoshopDocument`).

**Why it matters.** (a) Windows Installer treats ShotoPhopEX and PhotoCraft as one product line:
installing one can replace or block the other (PhotoCraft 0.21 is "newer" than ShotoPhopEX 0.1,
so a user with PhotoCraft installed may be refused, and installing PhotoCraft later would
silently replace ShotoPhopEX in the same folder). (b) Settings › Apps shows the publisher as
Learning Machines LLC, which attributes this fork to upstream's company. (c) The `ArtCraft` folder
name is one of the ArtCraft marks (`docs/brand/LICENSE-brand.txt` asks forks not to present
themselves as ArtCraft products).

**Why it was not changed here.** Changing the UpgradeCode or folder breaks upgrades of existing
v0.1.0 installs unless a migration is written and tested on Windows, and the shared update library
(`craft-update`, `src/detect.rs`) recognises installed MSIs by `Publisher == "Learning Machines LLC"`,
so changing the Manufacturer alone would break in-app updates for MSI users.

**Recommended action.** In one PR, tested on a Windows machine with (i) a v0.1.0 ShotoPhopEX
install and (ii) an upstream PhotoCraft install: a new UpgradeCode; an `<Upgrade>` entry that
removes old installs only when their ProductName is ShotoPhopEX (custom action or
`AppSearch` on the uninstall key); install folder `ShotoPhopEX`; Manufacturer "ShotoPhopEX";
new ProgIDs; and a `craft-update` patch (or local override) so detection accepts the new
publisher. The exe's own version resource was changed in this pass (FIXED: CompanyName and
LegalCopyright no longer name Learning Machines LLC alone).

### U-2. MOSS Edit adapter: no licence stated

`chinoll/Moss-Image-Edit` states no licence for its weights (its code is GPLv3). Without a licence
no use is clearly permitted. The 12/16 GB pack downloads it by default for MOSS Edit. This pass
labels it "No licence stated / commercial use unknown" in Get AI Models. **Options:** ask the
author for a licence; move it from the default pack to an opt-in extra; or keep it and rely on
the label. The Mac thread plans to use it too.

### U-3. In-app acceptance instead of the licensor's gate

Krea (`krea/Krea-2-Turbo`, gated), Meta (`facebook/sam3`, `facebook/sam3.1`, manual approval)
and Black Forest Labs (FLUX.3 Klein, gated) make users accept their licence on Hugging Face.
Get AI Models downloads Comfy-Org's ungated repacks (and the Mac thread plans an ungated mflux
conversion). This pass added a licence list and a mandatory acceptance tick box, but that records
nothing with the licensor. **Question:** is that sufficient, or should these files be fetched from
the licensor's own gated repository with the user's Hugging Face token? Meta's "manual" gate
also screens requesters (trade controls), which a repack bypasses entirely.

### U-4. Encoders hosted in differently licensed repacks

Qwen3-VL-4B (in `Comfy-Org/Krea-2`, repo licence Krea 2 Community) and Qwen3-VL-8B (in
`Comfy-Org/Qwen-Image-2.1`, repo licence Qwen Research) are Apache-2.0 upstream. The manifest
labels them Apache-2.0. Confirm the repack files are unmodified conversions of the Apache weights.

### U-5. Outpaint and Restore LoRAs name "qwen-research-license" without a text

`ausboss/Qwen-Image-2.1-Outpaint-LoRA` and `…-Restore-LoRA` link no licence text. They are labelled
Qwen Research (non-commercial). Ask the author or treat as non-commercial.

### U-6. CLIP tokenizer (SAM 3.1 text prompts)

`openai/clip-vit-large-patch14` states no licence on Hugging Face; OpenAI's CLIP code repository
is MIT, and the model card calls deployed use out of scope. The app downloads only
`tokenizer.json` (a vocabulary). Low risk; confirm or ship SAM's own tokenizer instead.

### U-7. FLUX.2 small VAE provenance

`flux2-vae_small` comes from `Comfy-Org/MOSS-Image` (Apache-2.0). Its origin relative to Black
Forest Labs' FLUX.2 VAE releases was not traced.

### U-8. Two Qwen tokenizer files from the ComfyUI repository

`merges.txt` and `tokenizer_config.json` are fetched from ComfyUI's GitHub repository (GPL-3.0) at
a pinned commit. The data originates in Qwen's Apache-2.0 tokenizer. Low risk; fetching them from
Qwen's own repository would remove the question.

### U-9. TAESD-family preview decoders

`madebyollin/taew2_1` returned 404 on 2026-11-07; the MIT licence recorded in `docs/comfy/models.md`
was not re-verified. Not downloaded by the app.

### U-10. Krea's content-filtering clause

Krea's card says deployers "are required to implement content filtering measures or equivalent
review processes". The app runs locally, bring-your-own, with no filtering. Is a local editor
that offers a download a "deployer"?

### U-11. Crates whose packages ship no licence file

24 to 26 shipped crates (accesskit, candle-nn/-transformers/-kernels, wasmi, zune-*, fax,
profiling, spirv, simd_helpers, alloc-stdlib and others; the build prints the list) ship no
licence text in their crates.io package. `THIRD_PARTY_NOTICES.txt` reproduces the MIT text with
the crates' declared authors and the full Apache-2.0 text. `alloc-stdlib` is BSD-3-Clause and its
exact copyright line was not found in the package. Low risk; fix by fetching each crate's
repository LICENSE at its tag.

### U-12. Native-engine ports written against GPL-3.0 reference code

Source comments in `genai-native/` (e.g. `qir/src/model/krea2.rs`, `wan_vae.rs`, `lora.rs`,
`gn-perception`) say the Krea 2 DiT, Wan VAE, LoRA key handling and SAM 3.1 pipeline were
"clean-room" ports of the architecture "as ComfyUI runs it" (`comfy/ldm/...`, GPL-3.0), from
stable-diffusion.cpp (MIT), from comfyui-krea2edit nodes (licence not checked) and from
transformers/diffusers (Apache-2.0). A grep found no copied files, but reading GPL source and then
writing equivalent code is not a clean room in the usual sense, and no similarity analysis was
done. **Question:** do these files contain protectable expression from GPL-3.0 sources? If so,
the distribution terms of the binaries would change. Recommended: a similarity review of the
named files against the cited sources, and, where code follows a source closely, re-deriving it
from the papers and model configs or reproducing the MIT/Apache notices of permissive sources.

### U-13. Apache-2.0 §4(b) notices in `qir`

`genai-native/crates/qir` is umag/qwen-image-rs (Apache-2.0), heavily changed in this fork. §4(b)
asks that modified files carry prominent notices that they were changed. This pass states the
modification in `genai-native/NOTICE`; per-file notices were not added because another thread is
editing those files. Recommended: add a one-line "Modified by the ShotoPhopEX contributors" header
to changed qir files when that work settles. Also: `genai-native/NOTICE` said qir was "imported
with its history", but the fork's history rewrite (2026-11-07) left only Cody's commits on those
paths, so upstream authorship is now recorded only in `crates/qir/NOTICE` and `LICENSE`.

### U-14. Name and icon close to Adobe Photoshop

"ShotoPhopEX" is a spoonerism of "Photoshop", and the "Sp" tile uses Photoshop's exact icon
colours (navy `#001E36`, blue `#31A8FF`, a rounded square with two letters). The README disclaims
any Adobe affiliation. **Question:** trademark or trade-dress confusion risk. This pass changed
nothing (the name and icon are the owner's decisions).

### U-15. ArtCraft name in the app

**Status (2026-11-13, owner chose to remove the branding).** Fixed for 0.1.2: the link is now
**Help › Upstream Project on GitHub** and "Based on PhotoCraft" on the start screen, both to
<https://github.com/storytold/photocraft>; the About link row reads "GitHub · PhotoCraft". The
logo files in `docs/brand/` were removed from the repository. The text below is the original finding.

The start screen's link rail and **Help › ArtCraft Website** show "ArtCraft" linking to
getartcraft.com (`crates/ui-egui/src/links.rs`, `canvas/start_screen.rs`, `engine/src/shell_commands.rs`).
The brand licence lets a fork say in plain text that it is "based on PhotoCraft by the ArtCraft
team" but asks forks to remove the ArtCraft marks. The owner kept these links on 2026-11-02.
Recommended: replace the "ArtCraft" link with plain text such as "Based on PhotoCraft by the
ArtCraft team", or get the ArtCraft Team's written OK.

### U-16. NVIDIA CUDA EULA conditions

The packages redistribute `cudart`, `cublas`, `cublasLt` and `curand` (all listed in the EULA's
Attachment A as distributable). Open points: §1.1.2(5) requires the terms under which the
application is distributed to be "consistent with" the EULA, including protection of NVIDIA's IP;
the app is MIT/Apache. This pass makes the packages state that the NVIDIA files are not covered by
the MIT/Apache licences and ships NVIDIA's EULA beside them (`runtime/NVIDIA-CUDA-EULA.txt`,
release workflow). Attachment A names the files generically (`cudart.dll`, `cublas.dll`); the
Windows files are versioned (`cudart64_13.dll`). v0.1.0 shipped without the EULA text.

### U-17. README recordings: source pictures

The GIFs in `docs/images/genai/` show edits of pictures whose origin (generated, stock, or the
owner's own photos) is not recorded. Record each source, or re-record from generated or own
pictures.

### U-19. Copyright in AI-assisted contributions

Much of this fork's code and documentation was written with AI coding agents under the owner's
direction. Copyright protection for AI-generated material is unsettled and varies by country.
The licences still apply to whatever is protectable; the NOTICE claims copyright only in Cody
Chu's "own contributions". No action needed unless the owner relies on enforcing copyright.

### U-20. Automatic update check

At launch the app asks GitHub's API for the latest release, which discloses the user's IP address
(and whatever request headers the HTTP client sends, not reviewed in detail) to GitHub. It is on by default and can be turned off. The README
now says so. No personal data is otherwise collected or sent. Low risk; a first-run prompt would
remove it.

### U-21. Benchmark methodology

The README's ComfyUI comparison does not record the ComfyUI version or NVIDIA driver; the README
now says the numbers are the project's own, on one machine, and indicative. Record those values
at the next benchmark run.

### U-22. Linux and Windows builds differ

The Linux tarball is built without the `heif` (HEIC/HEIF import) and `alerts` features that the
Windows packages have. Not a legal issue; check that the README's feature claims hold on Linux.
