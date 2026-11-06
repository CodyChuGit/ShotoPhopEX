# ShotoPhopEX documentation

ShotoPhopEX is a fork of [storytold/photocraft](https://github.com/storytold/photocraft) that adds
local generative tools. The models run in the native `photocraft --engine` worker
([`genai-native/`](../../genai-native/)); there is no Python, ComfyUI or cloud service. Weights are
bring-your-own and are read from a folder in ComfyUI's `models` layout.

Everything upstream wrote still applies: read [`AGENTS.md`](../../AGENTS.md) first, then the
upstream [`docs/`](../), then this folder.

| Document | What it is for |
|---|---|
| [`architecture.md`](architecture.md) | The design: the `photocraft-genai` crate, the native worker and its protocol, templates, the `generate.*` commands and the UI. |
| [`models.md`](models.md) | The model catalogue: what each model does, its licence, memory, file names and sizes. |
| [`upstream.md`](upstream.md) | Fork hygiene: syncing with upstream, licence and brand obligations, naming. |

## Security model

- The editor starts the worker itself and talks to it over loopback TCP. Every connection must
  first send a 256-bit token drawn from the operating system's random source; a peer without it
  is refused before it can hold the worker or make it allocate.
- The control channel and MCP server are loopback-only and token-authenticated (upstream's
  `photocraft-automation` security layer: request size caps, an authentication deadline,
  a connection limit).
- No document, prompt or picture leaves the machine. The only network requests are the update
  check against this repository's releases and downloads the user starts (an app icon over HTTPS).
- Model files are parsed with bounds-checked readers (`genai-native/crates/gn-weights`); a
  malformed file is an error, not a crash.

Report a vulnerability as described in [`SECURITY.md`](../../SECURITY.md).
