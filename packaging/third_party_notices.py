#!/usr/bin/env python3
"""Write THIRD_PARTY_NOTICES.txt (and optionally an SPDX 2.3 SBOM) for a ShotoPhopEX release.

The Rust crates are the ones `cargo tree` resolves for the shipped binaries (photocraft and
photocraft-cli) with the release's features on the release's target, normal dependencies only
(build scripts and dev-dependencies are not shipped). Each crate's own licence files are copied
in full from its source in the cargo cache. The non-crate material compiled into the binaries
(fonts, icons, the word list, face-detection weights, CUTLASS templates, the native engine's
ported code) and the NVIDIA libraries in runtime/ follow.

This is a notice file, not a legal opinion; docs/compliance/ has the review.

    python3 -I packaging/third_party_notices.py --target x86_64-unknown-linux-gnu \\
        --features photocraft/native-diffusion,photocraft-cli/native-diffusion \\
        --out dist/THIRD_PARTY_NOTICES.txt [--spdx dist/sbom.spdx.json] [--inventory inv.json]

Standard library only; run from anywhere (it finds the repository from its own path).
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LICENCE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|UNLICENSE|AUTHORS)([-_.].*)?$", re.I)

# Licence texts in this repository for material compiled into the binaries (ATTRIBUTION.md has
# the per-file rows). Paths are relative to the repository.
REPO_NOTICES = [
    ("Inter font (assets/fonts/Inter-*.ttf)", "assets/fonts/OFL-Inter.txt"),
    ("JetBrains Mono font (assets/fonts/JetBrainsMono-Regular.ttf)", "assets/fonts/OFL-JetBrainsMono.txt"),
    ("Lucide icons (assets/icons/*.svg)", "assets/icons/LICENSE-lucide.txt"),
    ("Magnetic Lasso icon, adapted from Lil' Seal (Noun Project)", "assets/icons/LICENSE-noun-magnetic-lasso.txt"),
    ("Painting symmetry axis icons", "assets/icons/LICENSE-symmetry-axes.txt"),
    ("SCOWL English word list (assets/dict/en_US-scowl-50.txt.gz)", "assets/dict/LICENSE-SCOWL.txt"),
    ("libfacedetection weights and network (crates/algo)", "crates/algo/data/LICENSE-libfacedetection.txt"),
    ("Solarized colour scheme (Solarized Dark theme)", "crates/ui-egui/src/LICENSE-solarized.txt"),
    ("UI translations", "crates/ui-egui/src/i18n/LICENSE-translations.txt"),
    ("NVIDIA CUTLASS templates compiled into the native engine's kernels", "packaging/third-party/LICENSE-cutlass.txt"),
    ("qwen-image-rs (genai-native/crates/qir), Apache-2.0: NOTICE", "genai-native/crates/qir/NOTICE"),
    ("genai-native: ported and vendored code", "genai-native/NOTICE"),
]
# egui's default fonts, embedded by crates/ui-egui/build.rs from the epaint_default_fonts crate.
EGUI_FONT_NOTICES = [
    ("Hack Regular (egui default font)", "fonts/Hack-Regular.txt"),
    ("Noto Emoji Regular (egui default font), SIL Open Font License 1.1", "fonts/OFL.txt"),
    ("Ubuntu Light (egui default font), Ubuntu Font Licence 1.0", "fonts/UFL.txt"),
    ("egui-icons (egui default font)", "fonts/egui-icons.txt"),
    ("emoji-icon-font (egui default font)", "fonts/emoji-icon-font-mit-license.txt"),
]
MIT_TEMPLATE = """MIT License

Copyright (c) {holders}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
NVIDIA = [
    ("CUDA Runtime", "cudart"),
    ("cuBLAS", "cublas"),
    ("cuBLASLt", "cublasLt"),
    ("cuRAND", "curand"),
]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def shipped_crates(target, features):
    cmd = ["cargo", "tree", "--locked", "-q", "-p", "photocraft", "-p", "photocraft-cli", "-e", "normal",
           "--target", target, "--prefix", "none", "--format", "{p}"]
    if features:
        cmd += ["--features", features]
    seen = set()
    for line in run(cmd).splitlines():
        m = re.match(r"^(\S+) v(\S+)", line.strip())
        if m:
            seen.add((m.group(1), m.group(2)))
    return seen


def metadata():
    return json.loads(run(["cargo", "metadata", "--format-version", "1", "--locked", "--all-features"]))


def licence_files(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(directory, n)
        if os.path.isfile(p) and LICENCE_FILE.match(n):
            out.append(p)
    # Licence files kept in a LICENSES/ or licenses/ folder (REUSE layout).
    for sub in ("LICENSES", "licenses", "license"):
        d = os.path.join(directory, sub)
        if os.path.isdir(d):
            out += [os.path.join(d, n) for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n))]
    return out


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().rstrip() + "\n"


def spdx_expr(lic):
    if not lic:
        return "NOASSERTION"
    e = re.sub(r"\s*/\s*", " OR ", lic.strip())
    return e


def download_location(pkg):
    src = pkg.get("source") or ""
    if src.startswith("registry+"):
        return "https://crates.io/api/v1/crates/{0}/{1}/download".format(pkg["name"], pkg["version"])
    if src.startswith("git+"):
        return src
    return "NOASSERTION"


def usage_of(pkg):
    src = pkg.get("source") or ""
    if not src:
        return "first-party"
    return "git" if src.startswith("git+") else "crates.io"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spdx")
    ap.add_argument("--inventory")
    ap.add_argument("--version", default="")
    a = ap.parse_args()

    meta = metadata()
    ws = set(meta["workspace_members"])
    by_key = {(p["name"], p["version"]): p for p in meta["packages"]}
    shipped = shipped_crates(a.target, a.features)
    crates = sorted((by_key[k] for k in shipped if k in by_key), key=lambda p: (p["name"], p["version"]))
    third = [p for p in crates if p["id"] not in ws and (p.get("source") or "")]
    first = [p for p in crates if not (p.get("source") or "")]
    missing = []

    out = []
    w = out.append
    w("THIRD-PARTY NOTICES for ShotoPhopEX {0} ({1})\n".format(a.version or "", a.target).replace("  ", " "))
    w("=" * 78 + "\n\n")
    w("ShotoPhopEX is licensed under MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE, NOTICE), except\n"
      "genai-native/crates/qir, which is Apache-2.0 only. It includes the third-party material below,\n"
      "each under its own licence. This file reproduces those notices; it is not a legal opinion.\n"
      "Source for every crate is at the download location in the SPDX SBOM published with the release.\n\n")
    w("1. NVIDIA libraries in runtime/\n" + "-" * 78 + "\n")
    w("The CUDA Runtime, cuBLAS, cuBLASLt and cuRAND libraries in runtime/ are NVIDIA's proprietary\n"
      "software, redistributed under the NVIDIA CUDA Toolkit End User License Agreement (Attachment A,\n"
      "distributable components). They are not covered by ShotoPhopEX's licences. The EULA, with the\n"
      "third-party notices for these libraries (its Attachment B), is runtime/NVIDIA-CUDA-EULA.txt when\n"
      "the package was built with it, and https://docs.nvidia.com/cuda/eula/index.html.\n"
      "Copyright (c) NVIDIA Corporation. All rights reserved.\n\n")

    w("2. Material compiled into the binaries\n" + "-" * 78 + "\n\n")
    for title, rel in REPO_NOTICES:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        w("### {0}\n({1})\n\n{2}\n".format(title, rel, read(p)))
    fonts = next((p for p in meta["packages"] if p["name"] == "epaint_default_fonts"), None)
    if fonts:
        d = os.path.dirname(fonts["manifest_path"])
        for title, rel in EGUI_FONT_NOTICES:
            p = os.path.join(d, rel)
            if os.path.isfile(p):
                w("### {0}\n(epaint_default_fonts {1}, {2})\n\n{3}\n".format(title, fonts["version"], rel, read(p)))
            else:
                missing.append("epaint_default_fonts/" + rel)

    w("3. Rust crates ({0})\n".format(len(third)) + "-" * 78 + "\n\n")
    inventory = []
    for p in third:
        d = os.path.dirname(p["manifest_path"])
        files = licence_files(d)
        # A crate from a git workspace (egui, craft-libs) keeps its licence at the checkout's root.
        up = d
        while not files and (p.get("source") or "").startswith("git+") and os.path.dirname(up) != up:
            up = os.path.dirname(up)
            files = licence_files(up)
            if os.path.isdir(os.path.join(up, ".git")) or os.path.isfile(os.path.join(up, ".cargo-ok")):
                break
        lic = p.get("license") or ("see " + p["license_file"] if p.get("license_file") else "")
        w("### {0} {1}\nLicense: {2}\nSource: {3}\n".format(p["name"], p["version"], lic or "NOASSERTION",
                                                          p.get("repository") or download_location(p)))
        if p.get("authors"):
            w("Authors: {0}\n".format(", ".join(p["authors"])))
        w("\n")
        if not files:
            missing.append("{0} {1}: no licence file in the crate".format(p["name"], p["version"]))
            w("(The crate's package ships no licence file; its manifest declares: {0}.)\n".format(lic or "nothing"))
            if "MIT" in lic:
                holders = ", ".join(p.get("authors") or []) or "the {0} authors".format(p["name"])
                w("\n" + MIT_TEMPLATE.format(holders=holders))
            if "Apache-2.0" in lic:
                w("The Apache License 2.0 text is reproduced once, at the end of this file.\n")
            w("\n")
        for f in files:
            w("--- {0} ---\n{1}\n".format(os.path.basename(f), read(f)))
        inventory.append({
            "name": p["name"], "version": p["version"], "license": spdx_expr(p.get("license") or ""),
            "source": usage_of(p), "location": p.get("repository") or download_location(p),
            "licenseFiles": [os.path.basename(f) for f in files],
        })

    w("4. Apache License, Version 2.0 (full text)\n" + "-" * 78 + "\n\n" + read(os.path.join(ROOT, "LICENSE-APACHE")))
    text = "".join(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    if a.inventory:
        with open(a.inventory, "w", encoding="utf-8") as f:
            json.dump({"target": a.target, "features": a.features, "crates": inventory,
                       "firstParty": [p["name"] for p in first], "missing": missing}, f, indent=1)
            f.write("\n")

    if a.spdx:
        ver = a.version or next((p["version"] for p in first if p["name"] == "photocraft"), "0")
        doc_ns = "https://github.com/CodyChuGit/ShotoPhopEX/sbom/{0}/{1}/{2}".format(ver, a.target, uuid.uuid4())
        pkgs = [{
            "SPDXID": "SPDXRef-Package-shotophopex", "name": "ShotoPhopEX", "versionInfo": ver,
            "downloadLocation": "https://github.com/CodyChuGit/ShotoPhopEX", "filesAnalyzed": False,
            "licenseDeclared": "MIT OR Apache-2.0", "licenseConcluded": "NOASSERTION",
            "copyrightText": "Copyright (c) 2026 ArtCraft Team and the PhotoCraft contributors; portions Copyright (c) 2026 Cody Chu and other ShotoPhopEX contributors",
            "supplier": "NOASSERTION", "primaryPackagePurpose": "APPLICATION",
        }]
        rels = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Package-shotophopex"}]

        def sid(name, version):
            return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", "{0}-{1}".format(name, version))

        for p in crates:
            if p["name"] == "photocraft":
                continue
            src = p.get("source") or ""
            entry = {
                "SPDXID": sid(p["name"], p["version"]), "name": p["name"], "versionInfo": p["version"],
                "downloadLocation": download_location(p) if src else "https://github.com/CodyChuGit/ShotoPhopEX",
                "filesAnalyzed": False, "licenseDeclared": spdx_expr(p.get("license") or ""), "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION", "supplier": "NOASSERTION", "primaryPackagePurpose": "LIBRARY",
            }
            if src.startswith("registry+"):
                entry["externalRefs"] = [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                                          "referenceLocator": "pkg:cargo/{0}@{1}".format(p["name"], p["version"])}]
            pkgs.append(entry)
            rels.append({"spdxElementId": "SPDXRef-Package-shotophopex", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": entry["SPDXID"]})
        for name, short in NVIDIA:
            entry = {
                "SPDXID": "SPDXRef-Package-nvidia-" + short, "name": "NVIDIA " + name, "versionInfo": "NOASSERTION",
                "downloadLocation": "https://developer.download.nvidia.com/compute/cuda/redist/", "filesAnalyzed": False,
                "licenseDeclared": "LicenseRef-NVIDIA-CUDA-EULA", "licenseConcluded": "NOASSERTION",
                "copyrightText": "Copyright (c) NVIDIA Corporation", "supplier": "Organization: NVIDIA Corporation",
                "primaryPackagePurpose": "LIBRARY",
            }
            pkgs.append(entry)
            rels.append({"spdxElementId": "SPDXRef-Package-shotophopex", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": entry["SPDXID"]})
        doc = {
            "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
            "name": "ShotoPhopEX-{0}-{1}".format(ver, a.target), "documentNamespace": doc_ns,
            "creationInfo": {"created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "creators": ["Tool: packaging/third_party_notices.py"],
                             "comment": "Rust crates from cargo tree (normal dependencies of photocraft and photocraft-cli, features: {0}); NVIDIA libraries from the release packaging. licenseDeclared is each crate's Cargo.toml licence; nothing here is a legal conclusion.".format(a.features or "default")},
            "hasExtractedLicensingInfos": [{"licenseId": "LicenseRef-NVIDIA-CUDA-EULA", "name": "NVIDIA CUDA Toolkit End User License Agreement",
                                            "extractedText": "See https://docs.nvidia.com/cuda/eula/index.html", "seeAlsos": ["https://docs.nvidia.com/cuda/eula/index.html"]}],
            "packages": pkgs, "relationships": rels,
        }
        with open(a.spdx, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")

    print("{0}: {1} crates, {2} bytes, sha256 {3}".format(a.out, len(third), len(text.encode()),
                                                          hashlib.sha256(text.encode()).hexdigest()[:16]))
    for m in missing:
        print("warning: " + m, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
