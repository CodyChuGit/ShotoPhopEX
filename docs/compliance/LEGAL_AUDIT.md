# ShotoPhopEX legal and compliance audit

Date: 2026-11-07. Scope: the repository at this change (fork of storytold/photocraft, merged up
to PhotoCraft v0.21.0), its Git history, dependencies, build and release workflows, and the
published v0.1.0 release files.

**This is an engineering audit, not a legal opinion.** It cannot establish every right or rule
out liability, and it does not certify the project as compliant. Items that need a lawyer or the
maintainer are in [`UNRESOLVED_RISKS.md`](UNRESOLVED_RISKS.md).

Companion documents: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) (inventory),
[`MODEL_LICENSES.md`](MODEL_LICENSES.md), [`RELEASE_COMPLIANCE.md`](RELEASE_COMPLIANCE.md),
[`SECURITY_AUDIT.md`](SECURITY_AUDIT.md), SBOMs in [`sbom/`](sbom/).

Owner's intent (as given for this audit): ShotoPhopEX stays free and open source, dual-licensed
MIT OR Apache-2.0 where permissible; anyone may fork, modify, redistribute and use it
commercially; legally required notices are kept, but no extra attribution is demanded; upstream
rights are preserved; no proprietary terms, registration, telemetry or monetisation are added.

## Status labels

**FIXED**: verified issue corrected and tested (or, for packaging, corrected and checked as far
as this environment allows; see Validation). **VERIFIED**: reviewed with evidence, no change
needed. **UNRESOLVED**: insufficient evidence or permission. **BLOCKER**: should prevent the
affected release. **RECOMMENDATION**: optional, not a legal requirement.

## Findings

### Licensing and copyright (phases 1 and 2)

| # | Finding | File(s) | Evidence | Severity | Status / action |
|---|---|---|---|---|---|
| L-1 | Upstream PhotoCraft is MIT OR Apache-2.0 at the merged revisions; no CLA or DCO; contributions are inbound MIT OR Apache-2.0 | `LICENSE-*`, `docs/contributing.md`, upstream repo | Upstream root licences, NOTICE, AGENTS.md read 2026-11-07; `docs/contributing.md` | — | VERIFIED: dual licensing of upstream code is authorised |
| L-2 | `NOTICE` (added 2d6b481) said both MIT and Apache-2.0 require keeping the whole NOTICE file and the LICENSE-MIT copyright lines, and asked forks to credit "ShotoPhopEX by Cody Chu" in their README or About screen | `NOTICE`, `README.md` | MIT has no NOTICE requirement; Apache-2.0 §4(d) requires only the attribution notices that pertain to the derivative work; the credit request read as a licence condition | High (overstated obligations and an extra condition, against the owner's intent) | **FIXED**: NOTICE and README now summarise each licence's actual obligations and state "Public attribution to ShotoPhopEX by Cody Chu, with a link to https://github.com/CodyChuGit/ShotoPhopEX, is appreciated but is not an additional license requirement." |
| L-3 | `NOTICE` said "all other changes and additions made in this repository are by Cody Chu and the ShotoPhopEX contributors", which includes upstream merges and third-party code | `NOTICE` | upstream merges (#99 etc.), qir from umag | Medium | **FIXED**: upstream authors keep copyright in their work; Cody Chu claims copyright only in his own contributions; other contributors keep theirs |
| L-4 | `LICENSE-MIT` second line claimed copyright for "Cody Chu … and the ShotoPhopEX contributors" next to upstream's line | `LICENSE-MIT` | — | Low | **FIXED**: "Portions Copyright (c) 2026 Cody Chu … and other ShotoPhopEX contributors". Upstream's line unchanged. No copyright line added to any upstream source file (none of the fork's files carry per-file copyright headers) |
| L-5 | `genai-native/crates/qir` is Apache-2.0 only (umag/qwen-image-rs), but README and NOTICE presented the whole project as MIT OR Apache-2.0; `genai-native/NOTICE` did not cover gn-diffusion and gn-segment | `README.md`, `NOTICE`, `genai-native/NOTICE` | `qir/Cargo.toml` `license = "Apache-2.0"`, `qir/LICENSE`; umag/qwen-image-rs LICENSE (Apache-2.0) | Medium | **FIXED**: the exception is stated in README, NOTICE and genai-native/NOTICE |
| L-6 | Apache-2.0 text complete and unmodified | `LICENSE-APACHE`, `genai-native/LICENSE-APACHE` | `diff` against apache.org's text: identical through "END OF TERMS AND CONDITIONS"; only the optional appendix is omitted | — | VERIFIED |
| L-7 | Upstream notices preserved: PhotoCraft NOTICE section, ArtCraft brand notice, third-party list, sctk-adwaita and emoji-icon-font MIT texts | `NOTICE` | diff of this change: the upstream section is unchanged below the separator | — | VERIFIED |
| L-8 | egui's default fonts (Hack, Noto Emoji, Ubuntu Light) are embedded in the binaries but were missing from ATTRIBUTION.md and NOTICE (an upstream gap too) | `crates/ui-egui/build.rs`, `ATTRIBUTION.md`, `NOTICE` | build.rs deflates `HACK_REGULAR`, `NOTO_EMOJI_REGULAR`, `UBUNTU_LIGHT` | Medium | **FIXED**: rows and NOTICE entry added; texts in `THIRD_PARTY_NOTICES.txt` |
| L-9 | `jpeg-encoder` is (MIT OR Apache-2.0) AND IJG; the IJG acknowledgement was missing | `NOTICE` | crate manifest | Low | **FIXED**: IJG statement added to NOTICE |
| L-10 | ATTRIBUTION.md had two stale `assets/app-icon/` rows ("Px" tile, upstream kitsune) | `ATTRIBUTION.md` | files in `assets/app-icon/` | Low | **FIXED**: one row for the "Sp" icon; rows for `docs/images/sp-logo.png` and `docs/images/genai/` added |
| L-11 | NOTICE claimed "all official releases" embed craft-fonts | `NOTICE` | release workflow sets no `CRAFT_FONTS_DIR`; no craft-fonts names in the binaries | Low | **FIXED**: ShotoPhopEX builds stated to be without craft-fonts (upstream text kept) |
| L-12 | Native-engine model code written against GPL-3.0 references (ComfyUI) | `genai-native/` | source comments | Medium | **UNRESOLVED** U-12 |
| L-13 | Apache §4(b) change notices for modified `qir` files; upstream history no longer in Git | `genai-native/crates/qir` | history rewrite 2026-11-07 | Low | **UNRESOLVED** U-13; modification statement added to genai-native/NOTICE |
| L-14 | Copyright in AI-assisted contributions | whole fork | — | Medium | **UNRESOLVED** U-19 |
| L-15 | Submodules | — | none (`git submodule status` empty, no `.gitmodules`) | — | VERIFIED |

### Third-party components (phase 3)

| # | Finding | Evidence | Status |
|---|---|---|---|
| T-1 | 632 crate versions ship across both targets; all declared licences are permissive or have a permissive option; no GPL/LGPL/MPL code in the shipped set | `cargo tree`, `cargo metadata` (THIRD_PARTY_LICENSES.md) | VERIFIED |
| T-2 | v0.1.0 packages carried none of the crates' licence texts (MIT/BSD/ISC/Unicode require them in binary distributions) | package listings | **FIXED** for future packages: `packaging/third_party_notices.py` writes `THIRD_PARTY_NOTICES.txt` (all crates' licence files, assets, CUTLASS, genai-native notices) into the MSI, ZIP and tarball |
| T-3 | 24–26 crates ship no licence file in their packages | notices script warnings | **UNRESOLVED** (low) U-11; standard texts reproduced |
| T-4 | SBOM | — | **FIXED**: SPDX 2.3 per platform in `docs/compliance/sbom/` and generated with each release |

### NVIDIA and CUDA (phase 4)

| # | Finding | Evidence | Status |
|---|---|---|---|
| N-1 | Each package ships exactly `cudart`, `cublas`, `cublasLt`, `curand`, unmodified and NVIDIA-signed (Windows); all four are Attachment A distributables | `RELEASE_COMPLIANCE.md` | VERIFIED |
| N-2 | No NVIDIA EULA text or notice beyond a one-line `THIRD_PARTY.txt` | package listings | **FIXED** for future packages: EULA from NVIDIA's archive/wheel shipped as `runtime/NVIDIA-CUDA-EULA.txt`; THIRD_PARTY.txt states the libraries are not under MIT/Apache |
| N-3 | EULA §1.1.2(5) consistency of distribution terms; versioned DLL names | EULA | **UNRESOLVED** U-16 |

### Models (phase 5)

| # | Finding | Status |
|---|---|---|
| M-1 | No weights are bundled or re-hosted; downloads are user-initiated from publishers' pages or Comfy-Org repacks | VERIFIED |
| M-2 | Get AI Models downloaded non-commercial (Qwen-Image-2.1 family, Krea 2 General Edit v2), revenue-capped (Krea 2) and gated-at-source (Krea, SAM 3.1) weights with no licence shown and no acceptance | **FIXED**: licences listed per file with links and commercial status; Download requires accepting them; `models.get` requires `acceptLicenses: true`; tests added |
| M-3 | README said ViTMatte is MIT (it is Apache-2.0) and summarised Krea 2 and SAM licences loosely | **FIXED** |
| M-4 | MOSS Edit adapter has no licence; acceptance vs source gates; several provenance items | **UNRESOLVED** U-2 … U-10 |

### Trademarks and branding (phase 6)

| # | Finding | Status |
|---|---|---|
| B-1 | App name, window title, Start menu entry, desktop entry, About, installer UI, README: ShotoPhopEX | VERIFIED |
| B-2 | Exe version resources named "Learning Machines LLC" (upstream's company) and "the PhotoCraft authors" | **FIXED**: "ShotoPhopEX contributors"; "the PhotoCraft authors and the ShotoPhopEX contributors" |
| B-3 | MSI UpgradeCode, Manufacturer, `ArtCraft\PhotoCraft` folder and ProgIDs are upstream's | **BLOCKER** for the next MSI, **UNRESOLVED** U-1 (needs migration and Windows testing; `craft-update` depends on the Manufacturer string) |
| B-4 | Internal names kept for compatibility: crate names, `photocraft.exe`, app id `ai.storyteller.photocraft`, `.pcraft`, settings folders (`%APPDATA%\Photocraft`, `~/.config/photocraft`), release file names (`photocraft-<v>-…`, parsed by the updater) | VERIFIED necessary for compatibility; RECOMMENDATION: leave as is |
| B-5 | "ArtCraft" link and Help menu item in the app | **UNRESOLVED** U-15 |
| B-6 | ArtCraft logos are not in any package | VERIFIED |
| B-7 | README identifies the fork as independent and not endorsed; factual credit to PhotoCraft kept | **FIXED** (wording strengthened: "an independent fork, not an official PhotoCraft release") |
| B-8 | Name and icon resemble Adobe Photoshop's | **UNRESOLVED** U-14 |
| B-9 | `photocraft-cli.exe` embeds `assets/app-icon/photocraft.ico`, which is now the Sp tile | VERIFIED |

### Privacy and network (phase 7)

| # | Finding | Status |
|---|---|---|
| P-1 | Image editing and inference run locally; no telemetry, analytics, accounts, crash upload or remote inference in the code | VERIFIED (`SECURITY_AUDIT.md` › Network exposure) |
| P-2 | README said "Nothing leaves your PC" and "Does anything go to the cloud? No", but the app checks GitHub for updates at launch by default and downloads models and CUDA libraries on request | **FIXED**: README states what is local and each network use, and how to turn off the update check |
| P-3 | Personal data: none collected; network requests disclose IP address to GitHub, Hugging Face and NVIDIA | VERIFIED; U-20 (low) |

### Security (phase 8): see `SECURITY_AUDIT.md`

S-1 (FIXED) `rustls-webpki`/`webpki-root-certs` advisories in the updater's TLS stack; loopback
check before bind (FIXED, hardening); every listener is loopback-only and off by default
(VERIFIED); no secrets in the repository (VERIFIED).

### Release packaging (phase 9): see `RELEASE_COMPLIANCE.md`

| # | Finding | Status |
|---|---|---|
| R-1 | Checksums match | VERIFIED |
| R-2 | **v0.1.0 MSI ships neither LICENSE-MIT nor LICENSE-APACHE** | **BLOCKER** for the v0.1.0 MSI as published; **FIXED** in `package.ps1` for the next build |
| R-3 | No third-party notices or NVIDIA EULA in any v0.1.0 package | **FIXED** in packaging (T-2, N-2) |
| R-4 | README's SmartScreen text presented the warning as a formality | **FIXED**: explains it means the publisher is unverified, and to verify the hash first (and the hash's limits) |
| R-5 | Unsigned; no provenance attestation; reproducible builds not attempted | VERIFIED unsigned (owner's decision); RECOMMENDATIONs in `SECURITY_AUDIT.md` |

### Product claims (phase 10)

| Claim (README) | Finding | Status |
|---|---|---|
| "100% Rust" badge | The engine compiles CUDA C++ kernels (SageAttention, ConvRot, CUTLASS templates) and loads NVIDIA's libraries | **FIXED**: badge now "Rust + CUDA kernels"; prose "Written in Rust" kept |
| "Nothing leaves your PC", "No cloud" | Inference is local; network is used for updates and downloads | **FIXED** (P-2); "No cloud" kept for inference, which is accurate |
| "Faster than ComfyUI" / "faster on 10 of 12" | The table shows 10 of 12 rows faster, measured by the project on one RTX 5090, without ComfyUI version or driver recorded | Kept; **FIXED** qualification added; U-21 |
| Hardware: RTX 30/40/50, 12 GB+, driver 580+, Windows 10/11, Linux glibc 2.39 | Consistent with the build (`CUDA_COMPUTE_CAP` 86, `QIR_CUDA_ARCHS` 86/89/120, CUDA 13) and the README's own status column (24 GB "not yet benchmarked") | VERIFIED as stated with its caveats |
| "Mac … planned" | Planned, not shipped | VERIFIED |
| "One executable plus NVIDIA's libraries" | Two executables | **FIXED** (sentence replaced) |
| Licence statements | qir exception, model licences | **FIXED** (L-5, M-3) |
| Security/privacy guarantees | No sandbox or malicious-file safety is claimed | VERIFIED |

## What was changed

- `LICENSE-MIT`, `NOTICE`, `README.md`, `ATTRIBUTION.md`, `genai-native/NOTICE`: L-2 … L-11, B-7,
  P-2, R-4, claims.
- `crates/engine/model-packs.json`, `crates/engine/src/model_packs.rs`,
  `crates/ui-egui/src/get_models_ui.rs`: licence data, licence list and acceptance (M-2).
- `apps/photocraft/build.rs`, `apps/photocraft-cli/build.rs`: version resources (B-2).
- `crates/automation/src/rpc.rs` (+ test): loopback check before binding.
- `Cargo.lock`: `rustls-webpki` 0.103.15 → 0.103.16, `webpki-root-certs` 1.0.9 → 1.0.10 (S-1).
- `packaging/third_party_notices.py` (new), `packaging/third-party/LICENSE-cutlass.txt` (new),
  `packaging/windows/package.ps1`, `packaging/windows/runtime.ps1`,
  `packaging/linux/package-native.sh`, `.github/workflows/shotophopex-release.yml`: R-2, R-3, T-2,
  T-4, N-2.
- `docs/compliance/` (new): this report, the five companion documents and two SBOMs.

Not changed (needs approval or a decision): MSI identity (U-1), the ArtCraft link (U-15), the
MOSS Edit adapter's place in the default pack (U-2), the product name and icon (U-14), any
dependency versions beyond the two security patches, any public API (only `models.get` gained a
required parameter; remote automation could not call it before either).

## Validation

| Check | Result |
|---|---|
| `cargo fmt --all --check` | Pass |
| `cargo test -p photocraft-engine --lib model_packs` (9 tests, 2 new) | Pass |
| `cargo test -p photocraft-automation --lib rpc::tests::tcp` (incl. new `tcp_refuses_non_loopback_addresses_before_binding`) | Pass (5 tests) |
| `cargo test -p photocraft-engine --test panic_hunt` | Pass (1 passed, 1 ignored by design) |
| `photocraft-ui-egui` (Get AI Models licence section) | Compiles (lib and tests); it has no unit tests, so the tick box was not exercised in the running app |
| `cargo clippy -D warnings` on engine, automation, ui-egui | Engine and automation: clean. ui-egui: one `manual_range_contains` hit in untouched `canvas/tool_cursor/pen.rs` (pre-existing, newer clippy), none in changed files |
| `cargo audit` (root and genai-native locks) | Root: only dev-only `lopdf` and informational warnings remain; genai-native: `zip` advisories, not in shipped binaries |
| `packaging/third_party_notices.py` for both release targets | Runs; Windows 525 crates, Linux 577 |
| v0.1.0 artifacts: download, SHA-256, unpack, inventory | Done (`RELEASE_COMPLIANCE.md`) |
| Windows packaging (`package.ps1`, `runtime.ps1`, WiX), Linux `package-native.sh`, release workflow | **Not run**: no Windows, PowerShell, WiX, nvcc or CUDA here. The next release workflow run validates them; its new checks fail the build if a licence file is missing |
| `cargo check` for `x86_64-pc-windows-msvc` with `native-diffusion` | **Not run**: needs MSVC and CUDA; covered by CI |
| MSI upgrade tests | **Not run** (no MSI change made; required for U-1) |
| Licence-file completeness | Notices script warns on crates without licence files (U-11) |

## Are the v0.1.0 artifacts ready?

They are ready for **further legal review**, not cleared. As published, the MSI lacks both licence
texts and all three packages lack third-party notices and NVIDIA's EULA; the MSI also presents
itself as upstream's product. Packaging fixes land with this change for the next release; U-1
must be resolved (or the MSI withheld) before the next MSI is published.
