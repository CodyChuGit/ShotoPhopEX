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
That is the only code change, in `update` and in `installs_from` (search for `ShotoPhopEX` in `src/self_update.rs`); `Cargo.toml`
spells out the versions upstream took from its workspace, and the screenshot test is left out.

## When craft-libs is bumped

Changing the craft-libs `rev` in the root `Cargo.toml` does not update this copy. Copy
`crates/update/src` and `crates/update/tests/fixtures` from the new rev over this folder, re-apply
the change above, and run `cargo test --manifest-path vendor/craft-update/Cargo.toml`.
