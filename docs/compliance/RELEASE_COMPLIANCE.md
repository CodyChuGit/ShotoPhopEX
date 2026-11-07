# Release compliance: v0.1.0 artifact review and release checklist

Reviewed 2026-11-07. The three v0.1.0 files were downloaded from
<https://github.com/CodyChuGit/ShotoPhopEX/releases/tag/v0.1.0> and unpacked in an isolated
directory (`unzip`, `tar`, `msiextract`/`msiinfo` from msitools; PE version resources read with
`pefile`). No release was published or changed.

## Checksums

| File | Size | SHA-256 (computed) | Matches `SHA256SUMS` |
|---|---:|---|---|
| `photocraft-0.1.0-windows-x64.msi` | 521,793,536 | `78084133f827ef1b8be080fb8586996596c9758e6bb5f66753c3cb3b1dedafd7` | Yes |
| `photocraft-0.1.0-windows-x64-portable.zip` | 578,601,812 | `dc80bd2f1b3ab5d5480356c14c7708f183f2b2660e605e0a1da24125601da091` | Yes |
| `photocraft-0.1.0-linux-x86_64.tar.gz` | 649,533,691 | `a4374c0e5b8883c00c62bc301b747aedf84187648235c3b372f4cb67060886b7` | Yes |

`SHA256SUMS` is produced by the release workflow's `draft` job from the same files it uploads, so
it detects corruption, not a compromised release (the README now says so).

## What each package contains

| Path | MSI | ZIP | TAR | Notes |
|---|:-:|:-:|:-:|---|
| `photocraft` / `photocraft.exe`, `photocraft-cli` | ✓ | ✓ | ✓ | Unsigned. Windows version resource: ProductName ShotoPhopEX, CompanyName **Learning Machines LLC**, LegalCopyright "the PhotoCraft authors" (fixed for the next build) |
| `LICENSE-MIT`, `LICENSE-APACHE` | **✗** | ✓ | ✓ | **v0.1.0 MSI ships neither licence text** (BLOCKER for that artifact; fixed in packaging) |
| `NOTICE`, `ATTRIBUTION.md` | ✓ | ✓ | ✓ | NOTICE in v0.1.0 overstated the licences' requirements (fixed) |
| `README.md` | ✗ | ✓ | ✓ | |
| Third-party crate licence texts | ✗ | ✗ | ✗ | **Missing in v0.1.0**: about 520–580 crates are statically linked; MIT, BSD, ISC and Unicode licences require their notices in binary distributions. Fixed: `THIRD_PARTY_NOTICES.txt` in every package |
| Font, icon, word-list, face-detection and CUTLASS licence texts | ✗ | ✗ | ✗ | NOTICE named them by repository path, which isn't in the package. Fixed: reproduced in `THIRD_PARTY_NOTICES.txt` |
| `runtime/` NVIDIA libraries | ✓ | ✓ | ✓ (`bin/runtime`) | See below |
| `runtime/THIRD_PARTY.txt` | ✓ | ✓ | ✓ | Named the EULA only; Windows copy still said "PhotoCraft's" (fixed) |
| NVIDIA EULA text | ✗ | ✗ | ✗ | Fixed: `runtime/NVIDIA-CUDA-EULA.txt` from NVIDIA's own archive/wheel |
| `portable.txt` | — | ✓ | — | |
| `install.sh`, `.desktop`, MIME, icons | — | — | ✓ | Desktop entry `Name=ShotoPhopEX`, app id `ai.storyteller.photocraft` (upstream's id, kept for compatibility) |
| Model weights | ✗ | ✗ | ✗ | None shipped (VERIFIED) |
| ArtCraft logos (`docs/brand/`) | ✗ | ✗ | ✗ | Not shipped (VERIFIED). The binary contains the text "ArtCraft" and a getartcraft.com link (U-15) |
| craft-fonts typefaces | ✗ | ✗ | ✗ | Not embedded (no `CRAFT_FONTS_DIR` in the release workflow; no BIZ UD/Shippori strings in the binaries) |

### MSI metadata (v0.1.0)

| Property | Value | Finding |
|---|---|---|
| ProductName | ShotoPhopEX | OK |
| Manufacturer | Learning Machines LLC | Upstream's company (U-1) |
| UpgradeCode | `{D9660F32-2D0A-4ABE-B73D-17BE25F1542C}` | **Same as upstream PhotoCraft** (U-1, BLOCKER for the next MSI) |
| Install folder | `ProgramFiles64\ArtCraft\PhotoCraft` (per-user: `%LOCALAPPDATA%\Programs\ArtCraft\PhotoCraft`) | ArtCraft name; collides with an upstream install (U-1) |
| Start menu shortcut | ShotoPhopEX | OK |
| ARP links | github.com/CodyChuGit/ShotoPhopEX (about, updates, issues) | OK |

## NVIDIA redistribution

| File (Windows / Linux) | Version (from the file) | Source in the build | EULA Attachment A? | Status |
|---|---|---|---|---|
| `cudart64_13.dll` / `libcudart.so.13` | CUDA Runtime 13.3.29 (Windows); Linux from the `nvidia-cuda-runtime` 13.5.37 wheel | NVIDIA redist archive (Windows, CUDA 13.3.1 manifest, SHA-256 checked) / NVIDIA pip wheel (Linux) | Yes ("CUDA Runtime": `cudart.dll`, `libcudart.so`) | VERIFIED distributable; UNRESOLVED U-16 on §1.1.2(5) |
| `cublas64_13.dll`, `cublasLt64_13.dll` / `libcublas.so.13`, `libcublasLt.so.13` | cuBLAS 13.6.0 (Windows); `nvidia-cublas` 13.10.0.20 (Linux) | as above | Yes ("cuBLAS": `cublas.dll`, `cublasLt.dll`, `libcublas.so`, `libcublasLt.so`) | as above |
| `curand64_10.dll` / `libcurand.so.10` | cuRAND 10.4.3.29 (Windows); `nvidia-curand` 10.4.5.25 (Linux) | as above | Yes ("cuRAND") | as above |

All four Windows DLLs carry NVIDIA's valid Authenticode signatures and are unmodified. No other
NVIDIA file is shipped: no driver library (`libcuda.so.1` is linked against at build time and
never packaged), no cuDNN, no NVRTC, no CUTLASS binaries (CUTLASS is header-only and compiled
into the kernels; its BSD-3-Clause notice is now in `THIRD_PARTY_NOTICES.txt`).

The EULA's distribution requirements (§1.1.2) and how the packages meet them:

1. *Material additional functionality*: the libraries serve a full image editor. Met.
2. *Accessed only by your application*: they sit in the app's private `runtime/` folder and are
   loaded by its worker. Met in practice.
3. *Sample-source notice*: no NVIDIA sample code is shipped. Not applicable.
4. *Developer tools internal only*: no developer tools (nvcc, headers) are shipped. Met.
5. *Distribution terms consistent with the EULA*: the packages now state that the NVIDIA files are
   NVIDIA's, under the EULA, not under MIT/Apache, and ship the EULA. Whether that is sufficient
   is U-16.
6. *Notify NVIDIA of non-compliance*: no action.

## Fixes made to packaging (for the next release)

| Change | Files |
|---|---|
| MSI now installs `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`, `ATTRIBUTION.md` and `THIRD_PARTY_NOTICES.txt` | `packaging/windows/package.ps1` |
| Portable ZIP and Linux tarball carry `THIRD_PARTY_NOTICES.txt`; the ZIP check in the workflow fails if any licence file is missing | `package.ps1`, `packaging/linux/package-native.sh`, workflow |
| `THIRD_PARTY_NOTICES.txt` generated per target from `cargo tree` of the shipped binaries with the release features, copying each crate's own licence files, plus fonts, icons, CUTLASS and genai-native notices | `packaging/third_party_notices.py` |
| An SPDX 2.3 SBOM per platform is written beside the release files (`*.spdx.json`) and uploaded with them; the in-app updater ignores the extension | `third_party_notices.py`, workflow |
| NVIDIA's EULA, taken from NVIDIA's own archive (Windows) or wheel (Linux), ships as `runtime/NVIDIA-CUDA-EULA.txt`; packaging refuses to assemble the runtime folder without it | `runtime.ps1`, `package-native.sh`, workflow |
| Exe version resource: CompanyName "ShotoPhopEX contributors", LegalCopyright "the PhotoCraft authors and the ShotoPhopEX contributors" | `apps/photocraft/build.rs`, `apps/photocraft-cli/build.rs` |

These packaging changes could not be run here (no Windows, WiX, nvcc or CUDA in this
environment). The notices script was run for both targets (Windows 525 crates, Linux 577). **The
next release workflow run is the test**: check that both jobs pass and that the files appear in
the packages before publishing.

## Release checklist

Before publishing a draft release:

- [ ] U-1 resolved for the MSI (or the MSI held back and only the ZIP and tarball published).
- [ ] Release workflow green on Windows and Linux.
- [ ] Each package contains `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`, `ATTRIBUTION.md`,
      `THIRD_PARTY_NOTICES.txt`, `runtime/THIRD_PARTY.txt` and `runtime/NVIDIA-CUDA-EULA.txt`
      (`msiextract`, `unzip -l`, `tar -tzf`).
- [ ] `photocraft-<v>-windows-x64.spdx.json` and `photocraft-<v>-linux-x86_64.spdx.json` are
      attached, and `SHA256SUMS` lists every attached file.
- [ ] `cargo audit` clean for shipped crates (dev-only advisories noted).
- [ ] No model weights, ArtCraft logos or personal paths in any package.
- [ ] Release notes: say "unsigned"; never say "signed", "verified" or "legally compliant".
- [ ] Hardware smoke test of generation on Windows and Linux (existing practice).
- [ ] New models added since the last release have a `license` id in `model-packs.json`.

Reproducible builds were not attempted: the Windows build embeds build date and commit
(`PHOTOCRAFT_BUILD_DATE`, `PHOTOCRAFT_BUILD_SHA`) and PDB paths, so byte-identical rebuilds are not
expected today (RECOMMENDATION: set `SOURCE_DATE_EPOCH`-style fixed values and `--remap-path-prefix`,
then compare two CI runs). Artifact signing is declined by the owner; build-provenance attestation
is a no-certificate alternative (`SECURITY_AUDIT.md`).

**Verdict on the published v0.1.0 artifacts:** they are ready for further legal review, not
cleared. The MSI lacks both licence texts, all three lack the third-party notices and the NVIDIA
EULA, and the MSI identifies itself as upstream's product (U-1). These are fixed in packaging for
the next release (except U-1); re-publishing v0.1.0 is the owner's call.
