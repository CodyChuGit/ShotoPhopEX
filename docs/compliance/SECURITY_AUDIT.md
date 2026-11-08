# Security audit (compliance pass, 2026-11-07)

Scope: network exposure, automation (control server, headless JSON server, MCP), secrets,
downloads, dependency advisories, and the release packages. It builds on the two earlier audits
of this fork (PR #77 before the first release, PR #90 on the fork's own code; see
[`SECURITY.md`](../../SECURITY.md) and [`book/src/security/`](../../book/src/security/overview.md)).
The PSD/RAW/codec parsers are upstream PhotoCraft's and were not re-audited here.

This document does **not** claim the app is sandboxed or safe against malicious files: it is not
sandboxed, and opening untrusted documents runs complex parsers in the user's own process.

Labels: **FIXED** (changed and tested), **VERIFIED** (reviewed, no change needed),
**RECOMMENDATION** (optional hardening), **UNRESOLVED** (needs a decision).

## Confirmed vulnerabilities

| # | Finding | Affected | Evidence | Severity | Status / action |
|---|---|---|---|---|---|
| S-1 | `rustls-webpki` 0.103.15: certificate verification may accept distrusted intermediates (RUSTSEC-2026-0382); `webpki-root-certs` 1.0.9 trusts distrusted emSign intermediates (RUSTSEC-2026-0383) | The in-app update check and updater (`craft-update` → `ureq` → `rustls`), Windows and Linux v0.1.0 | `cargo audit` on the root `Cargo.lock` | Medium (a network attacker with such a certificate could spoof GitHub's update responses; downloaded installers still go through the updater's own checks) | **FIXED** in `Cargo.lock`: 0.103.16 and 1.0.10 (semver-compatible patch releases). `cargo audit` no longer reports them. v0.1.0 binaries still contain the old versions. |

No other advisory affects code that ships:

| Advisory | Crate | Why it does not affect the release | Status |
|---|---|---|---|
| RUSTSEC-2026-0187 stack overflow on nested PDF objects | `lopdf` 0.38.0 | Dev-dependency of `photocraft-io` only (`cargo tree -i lopdf -e all`); not in the shipped binaries | VERIFIED; RECOMMENDATION: bump to ≥ 0.42 for test tooling |
| RUSTSEC-2026-0367/0368/0369/0429 ZIP extraction | `zip` 8.6.0 | Only in `genai-native/Cargo.lock` (its standalone workspace, via candle). The shipped binaries are built from the root lock, which has `zip` 9.3.0 (patched); candle reads `.npz` entries and never calls `extract` | VERIFIED; RECOMMENDATION: `cargo update -p zip` in `genai-native/` when the Mac thread's edits settle |
| Unmaintained/yanked warnings (`paste`, `proc-macro-error`, `rustybuzz`, `ttf-parser`, `zbus_names`, `zvariant*`; yanked `async-executor`, `clipboard-win`, `yoke-derive`) | various | Informational; no known vulnerability | RECOMMENDATION: track upstream replacements |

## Network exposure

| Surface | Default | Binding / auth | Evidence | Status |
|---|---|---|---|---|
| Desktop control server (`--control <port>`, `PHOTOCRAFT_CONTROL_PORT`, Preferences › Integrations › Control Port) | **Off** (port 0 = only with `--control`) | Binds `127.0.0.1` only; 256-bit random bearer token (`getrandom`) required before dispatch; request-line, connection and batch limits; Allow Agent Control off refuses even `--control` | `apps/photocraft/src/main.rs` `control_setup`, `crates/automation/src/security.rs` | VERIFIED |
| Headless JSON server (`photocraft-cli serve --listen`) | Off unless started | Loopback only, token, connection cap, idle shutdown | `crates/automation/src/rpc.rs` | **FIXED (hardening)**: the address is now resolved and refused **before** binding if any resolved address is not loopback; previously it bound first and closed the socket after checking, so `0.0.0.0` was briefly open. New test `tcp_refuses_non_loopback_addresses_before_binding`. Not exploitable before (no connection was accepted), so this is hardening, not a vulnerability fix. |
| MCP server (`photocraft-cli mcp`) | Off unless started by an MCP client | stdio; `--bridge` accepts only `127.0.0.1`/`localhost`/`::1` | `crates/automation/src/bridge.rs` | VERIFIED |
| Native engine worker (`photocraft --engine`) | Started by the app for generation | `127.0.0.1` ephemeral port, random token, pre-auth size and time limits | `genai-native/crates/gn-engine/src/server.rs`, `client.rs` | VERIFIED |
| Single-instance handoff | Always | Loopback (`Ipv4Addr::LOCALHOST`) or a local pipe | `apps/photocraft/src/single_instance.rs` | VERIFIED |
| Update check | **On** by default | HTTPS to `api.github.com` / `github.com` for `CodyChuGit/ShotoPhopEX` | `crates/ui-egui/src/lib.rs` `UPDATE_APP`, `updates_ui.rs`; pref `general.automaticallyCheckUpdates` | VERIFIED; documented in README (U-20) |
| Get AI Models | User-initiated | System `curl` (System32 on Windows), HTTPS only, size + SHA-256 check, HF token on stdin and never saved; denied to remote automation | `crates/engine/src/model_packs.rs`, `crates/automation/src/workspace.rs` | VERIFIED; FIXED licence acceptance (see `MODEL_LICENSES.md`) |
| App icon from URL | User-initiated | HTTPS only, 20 MB cap, System32 curl | `apps/photocraft/src/services.rs` | VERIFIED |
| Printing (IPP) | User-initiated | To the printer the user picks (local or network) | `crates/ipp` | VERIFIED |
| Telemetry, analytics, crash upload, remote inference, accounts | **None found** | — | `grep` for telemetry/sentry/analytics/crash-report clients and HTTP clients: the only HTTP client is `ureq` in `craft-update` | VERIFIED |

## Secrets and logging

| Item | Evidence | Status |
|---|---|---|
| Control token printed to stderr when generated (so the user can copy it) | `main.rs` `eprintln!("photocraft: control token: …")`; not written through the logger, so it does not reach `photocraft.log` | VERIFIED; RECOMMENDATION: prefer `--control-token-file` in docs, since a terminal transcript can capture stderr |
| Hugging Face token | Typed in Get AI Models, kept in memory, passed to `curl` on stdin, cleared after the run, never saved or logged | VERIFIED |
| Credentials in the repository | `git grep` for `hf_[A-Za-z0-9]{30,}`, `ghp_`, `github_pat_`, private keys found none | VERIFIED |
| Personal paths | No developer-machine paths in the repository (owner's rule) | VERIFIED by `git grep` at this commit |

## Filesystem and documents

Remote automation uses separately configured read and write roots, canonicalised, with symlink
checks; commands with ambient paths, `models.get`, preference sections that carry paths (models,
runtime folders) and whole-preference writes are refused at the automation boundary (#90 closed
the `integrations.runtimePath` DLL-loading path). Not re-tested in this pass beyond reading the
code: VERIFIED by review, with the limits listed in `SECURITY.md` › Current limitations.

## Release integrity

| Item | Status |
|---|---|
| Checksums: `SHA256SUMS` on the v0.1.0 release matches all three downloaded files | VERIFIED |
| Code signing | Not signed (owner's decision, 2026-11-06). README now explains what the SmartScreen warning does and doesn't mean instead of presenting it as harmless. |
| Build provenance | Built by `.github/workflows/shotophopex-release.yml` on GitHub-hosted runners with pinned action SHAs; no attestation is published. RECOMMENDATION: `actions/attest-build-provenance` on the release files (no signing certificate needed). |
| Supply chain of build inputs | CUDA Windows archives checked against NVIDIA's manifest SHA-256; Linux NVIDIA wheels pinned by version only; CUTLASS cloned by tag. RECOMMENDATION (from #90, still open): `pip --require-hashes` for the wheels and pin CUTLASS by commit. |
| In-app updater on unsigned builds | `craft-update` refuses to self-update the portable build unless both files carry the same Authenticode signer, so unsigned portable installs are told to download manually. MSI updates rely on publisher detection (see U-1). | VERIFIED (behaviour), no change |

## Tests run for the changes

See `LEGAL_AUDIT.md` › Validation.
