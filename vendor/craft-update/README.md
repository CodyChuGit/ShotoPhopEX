# craft-update (ShotoPhopEX copy)

A copy of `crates/update` from [storytold/craft-libs](https://github.com/storytold/craft-libs) at
rev `6e03f65bf45181db77ba6229716e6624db9eb25b`, used in place of the git dependency through
`[patch."https://github.com/storytold/craft-libs"]` in the root `Cargo.toml`. MIT OR Apache-2.0,
as upstream (`LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`).

## What is different

ShotoPhopEX's release files are named `shotophopex-<version>-<os>-<arch>.<ext>`, while the app's
id (`App::id`, also the exe name and the MSI's registration) stays `photocraft`. Upstream's
`self_update::update` only looks for files named after `App::id`, so the app lists `shotophopex`
in `former_ids` and this copy makes `update` pick a download by every stem (`App::stems()`).
That is the first code change, in `update` and in `installs_from` (search for `ShotoPhopEX` in `src/self_update.rs`); `Cargo.toml`
spells out the versions upstream took from its workspace, and the screenshot test is left out.

## When craft-libs is bumped

Changing the craft-libs `rev` in the root `Cargo.toml` does not update this copy. Copy
`crates/update/src` and `crates/update/tests/fixtures` from the new rev over this folder, re-apply
the change above, and run `cargo test --manifest-path vendor/craft-update/Cargo.toml`.

A second change (also marked `ShotoPhopEX`): the Windows Installer publisher in `src/detect.rs`
(`PUBLISHER`, also used for the launcher's HKCU key in `src/msi_copy.rs`) is `ShotoPhopEX`, the
MSI's own Manufacturer since 0.1.2, instead of upstream's `Learning Machines LLC`. So the updater
only ever recognises this fork's installs, never an upstream PhotoCraft install.

A third change (also marked `ShotoPhopEX`): ShotoPhopEX releases are never code-signed, so the
self-update no longer requires a signature. On Windows `can_self_update_at` offers the update to
an unsigned copy and `signed_like` (in `src/self_update.rs`) accepts every file when the running
copy itself isn't signed; on macOS `Expect::self_update` (in `src/install.rs`) drops the Developer
ID team check. Downloads are still refused unless they come from the release's own GitHub download
link and match the SHA-256 the release publishes (GitHub's per-file digest or `SHA256SUMS.txt`);
the Mac bundle must also report the release's version.
