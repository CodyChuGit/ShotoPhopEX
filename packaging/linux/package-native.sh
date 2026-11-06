#!/usr/bin/env bash
# Fork: build and package ShotoPhopEX with the native generative engine for Linux x86_64:
#
#   $DIST/photocraft-<version>-linux-x86_64.tar.gz
#       photocraft-<version>-linux-x86_64/{bin, bin/runtime, share/applications,
#       share/icons, share/mime, share/doc/photocraft, install.sh}
#
# bin/runtime holds the CUDA libraries the binaries link (the CUDA runtime, cuBLAS, cuBLASLt and
# cuRAND; libcuda.so.1 comes with the NVIDIA driver;
# NVIDIA CUDA Toolkit EULA, Attachment A). The binaries find them through their RUNPATH
# ($ORIGIN/runtime, set by apps/*/build.rs), and the app uses that folder as its runtime folder
# when Preferences › AI Integrations › Native runtime folders is empty, so only an NVIDIA driver
# is needed.
# No model weights ship.
#
# Run from the extracted folder (bin/photocraft), or install for the current user with
# ./install.sh (into ~/.local) or for everyone with sudo ./install.sh /usr/local.
#
# Usage: packaging/linux/package-native.sh --cuda-lib <dir> [--skip-build]
#   --cuda-lib   the folder holding libcudart.so.13, libcublas.so.13, libcublasLt.so.13 and
#                libcurand.so.10 (a CUDA Toolkit's lib64, or the nvidia-* wheels' cu13/lib)
#
# Needs: the build environment of the README's Linux section (CUDA_COMPUTE_CAP, CUTLASS_DIR,
# CUDARC_CUDA_VERSION, nvcc on PATH).
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
LINUX="$ROOT/packaging/linux"
APP_ID=ai.storyteller.photocraft

SKIP_BUILD=0
CUDA_LIB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --cuda-lib) CUDA_LIB="$2"; shift 2 ;;
    -h | --help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CUDA_LIB" ]] || { echo "error: --cuda-lib is required" >&2; exit 2; }
LIBS=(libcudart.so.13 libcublas.so.13 libcublasLt.so.13 libcurand.so.10)
for l in "${LIBS[@]}"; do
  [[ -f "$CUDA_LIB/$l" ]] || { echo "error: $l is not in $CUDA_LIB" >&2; exit 1; }
done

if [[ $SKIP_BUILD -eq 0 ]]; then
  cargo build --release --locked -p photocraft -p photocraft-cli \
    --features photocraft/native-diffusion,photocraft-cli/native-diffusion
fi
BIN="$CARGO_TARGET_DIR/release"

BASENAME="photocraft-$VERSION-linux-x86_64"
WORK="$CARGO_TARGET_DIR/linux-native-package"
STAGE="$WORK/$BASENAME"
rm -rf "$WORK"
mkdir -p "$STAGE/bin/runtime" "$STAGE/share/applications" "$STAGE/share/mime/packages" \
  "$STAGE/share/icons/hicolor/128x128/apps" "$STAGE/share/icons/hicolor/256x256/apps" "$STAGE/share/doc/photocraft"
install -m 755 "$BIN/photocraft" "$BIN/photocraft-cli" "$STAGE/bin/"
for l in "${LIBS[@]}"; do install -m 644 "$CUDA_LIB/$l" "$STAGE/bin/runtime/"; done
cat > "$STAGE/bin/runtime/THIRD_PARTY.txt" <<'EOF'
Third-party libraries in this folder, loaded by ShotoPhopEX's native engine (photocraft --engine).
Each is redistributed under its own licence; none is covered by ShotoPhopEX's MIT OR Apache-2.0 licence.

CUDA runtime libraries: NVIDIA CUDA Toolkit EULA, Attachment A (redistributable)
  libcudart.so.13, libcublas.so.13, libcublasLt.so.13, libcurand.so.10
EOF
install -m 644 "$LINUX/$APP_ID.desktop" "$STAGE/share/applications/$APP_ID.desktop"
install -m 644 "$LINUX/$APP_ID.mime.xml" "$STAGE/share/mime/packages/$APP_ID.xml"
install -m 644 "$ROOT/assets/app-icon/sp/sp-128.png" "$STAGE/share/icons/hicolor/128x128/apps/$APP_ID.png"
install -m 644 "$ROOT/assets/app-icon/sp/sp-256.png" "$STAGE/share/icons/hicolor/256x256/apps/$APP_ID.png"
for f in README.md LICENSE-MIT LICENSE-APACHE NOTICE ATTRIBUTION.md; do
  if [[ -f "$ROOT/$f" ]]; then install -m 644 "$ROOT/$f" "$STAGE/share/doc/photocraft/"; fi
done
cat > "$STAGE/install.sh" <<'EOF'
#!/bin/sh
# Install ShotoPhopEX into a prefix: ~/.local by default (no root), or e.g. /usr/local with sudo.
# Remove it again with: ./install.sh --uninstall [prefix]
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = "--uninstall" ]; then uninstall=1; shift; else uninstall=0; fi
prefix="${1:-$HOME/.local}"
files="bin/photocraft bin/photocraft-cli bin/runtime share/applications/ai.storyteller.photocraft.desktop
share/mime/packages/ai.storyteller.photocraft.xml share/icons/hicolor/128x128/apps/ai.storyteller.photocraft.png
share/icons/hicolor/256x256/apps/ai.storyteller.photocraft.png share/doc/photocraft"
if [ "$uninstall" = 1 ]; then
  for f in $files; do rm -rf "${prefix:?}/$f"; done
  echo "Removed ShotoPhopEX from $prefix. Settings stay in ~/.config/photocraft."
  exit 0
fi
for f in $files; do
  mkdir -p "$(dirname "$prefix/$f")"
  rm -rf "${prefix:?}/$f"
  cp -R "$here/$f" "$prefix/$f"
done
update-desktop-database "$prefix/share/applications" >/dev/null 2>&1 || true
update-mime-database "$prefix/share/mime" >/dev/null 2>&1 || true
gtk-update-icon-cache -q -t "$prefix/share/icons/hicolor" >/dev/null 2>&1 || true
echo "Installed ShotoPhopEX into $prefix. Start it from your app menu or run $prefix/bin/photocraft."
EOF
chmod 755 "$STAGE/install.sh"

# The binaries must find the bundled libraries through their RUNPATH, not the build machine's.
if command -v readelf >/dev/null; then
  readelf -d "$STAGE/bin/photocraft" | grep -q 'ORIGIN/runtime' || { echo 'error: photocraft has no RUNPATH to $ORIGIN/runtime' >&2; exit 1; }
fi
# Start the CLI with nothing from the build machine on the loader path but the driver library
# (DRIVER_LIB_DIR: a folder with libcuda.so.1, for a build machine without an NVIDIA driver).
LD_LIBRARY_PATH="${DRIVER_LIB_DIR:-}" "$STAGE/bin/photocraft-cli" --version

mkdir -p "$DIST"
tar -C "$WORK" -czf "$DIST/$BASENAME.tar.gz" "$BASENAME"
echo "$DIST/$BASENAME.tar.gz"
