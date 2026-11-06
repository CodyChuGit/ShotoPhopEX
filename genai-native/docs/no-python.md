# No Python

genai-native has no Python in it. The engine, its build and its tests need only Rust and, for
the CUDA features, nvcc. Nothing here asks a user, or a developer, to install Python or put it
on `PATH`.

Python was never part of the shipped engine or of `cargo build`. It was only in developer
tooling, and that tooling went in three directions (October 2026, removed after commit
`aa5ccdf59d`):

## Ported to Rust: `qir compare`

The comparison gates are now subcommands of the `qir` binary (`crates/qir/src/compare.rs`):

| Was | Now |
| --- | --- |
| `scripts/compare_png.py a.png b.png` | `qir compare png a.png b.png` |
| `scripts/compare_latent.py a b [--threshold t]` | `qir compare latent a b [--threshold t]` |
| `scripts/compare_dit.py ours oracle` | `qir compare dit ours oracle [--image-tokens 4096]` |
| `tools/layered_compare.py native --comfy ref ...` | `qir compare layered native ref ...` (same flags) |
| `tools/lora_compare.py` / `tools/qwen21_ref_compare.py before after` | `qir compare raw before after [--limit 1]` |

Each prints the same numbers as before and exits 1 when the gate fails. `lora_monitor.py`
(board memory once a second while a command runs) is
`nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv -l 1`
in a second terminal.

## Deleted: benchmarks and probes against ComfyUI, TensorRT and the old 4090 host

`crates/qir/bench/` (ComfyUI and TensorRT/ONNX timing), the LoRA, SAM 3.1 and 2511 probes and
profilers that drove an installed ComfyUI, `sam31_smoke.py`, and `scripts/host.sh` /
`scripts/setup-host.sh` (the Mac-to-WSL 4090 driver and its torch venv). The app has no ComfyUI
or ONNX Runtime route any more, so there is nothing left for them to measure. Their results
stay in the reports next to where they lived (`crates/qir/tools/*.md`, `docs/TENSORRT.md`,
`docs/COMFYUI.md`, `HANDOVER.md`).

## Deleted, recoverable: the fixture generators

Some Rust tests compare against reference tensors in `crates/*/tests/fixtures/`. Those files
were made once by running ComfyUI's or diffusers' own code (the "oracle") on the CPU. The
fixtures are committed, so the tests need nothing else. The generators cannot be ported to
Rust: a Rust copy would only check Rust against itself, which proves nothing.

They are only needed to remake a fixture, for example to follow a change in the reference
code. Restore the one you need from git, run it with that reference install's own Python, and
delete it again:

```
git show aa5ccdf59d:<path> > <path>
```

| Fixture | Generator at `aa5ccdf59d` |
| --- | --- |
| `crates/gn-segment/tests/fixtures/{channel_layer_norm,resize,roi_align}` | `tools/gn_segment_fixtures.py` |
| `crates/gn-segment/tests/fixtures/{vision,text,geometry,detector,interactive}` | `crates/gn-segment/tools/gn_segment_<name>_fixtures.py` |
| `crates/qir/tests/fixtures/wan_2.1_vae` | `tools/qir_wan_vae_fixtures.py` |
| `qwen25_text` | `crates/qir/tools/qir_qwen25_fixtures.py` |
| `qwen25_edit` | `crates/qir/tools/qir_qwen25_edit_fixtures.py` |
| `qwen25_vision_small` | `crates/qir/tools/qir_qwen25_vision_fixtures.py` |
| `qwen_image_dit`, `qwen_image_layered_vae` | `crates/qir/tools/qwen_image_dit_fixtures.py` |
| `layered_pipeline*` | `crates/qir/tools/layered_pipeline_fixtures.py` |
| `edit2511_prep_small`, `edit2511_lora_maps` | `crates/qir/tools/edit2511_fixtures.py` |
| `edit2511_sampler` | `crates/qir/tools/edit2511_sampler_fixture.py` |
| 2511 DiT (`--full`, local) | `crates/qir/tools/qwen_edit_2511_fixtures.py` |
| `krea2_style*` | `crates/qir/tools/krea2_style_fixtures.py`, then `krea2_style_encoder_fixtures.py`, `krea2_style_model_fixtures.py` |
| LoRA weight oracles (local) | `crates/qir/tools/lora_weight_fixture.py` |
| diffusers reference set (latents, embeddings, PNGs) | `crates/qir/scripts/oracle.py` |
| 2511 LoRA residency captures (`edit2511_pipeline` ignored test) | `crates/qir/tools/edit2511_lora_probe.py` |
| MOSS-Image parity outputs (`moss_parity`) | `ref_moss.py`, never in this repository |

One test still calls ComfyUI's Python directly: `comfy_int8::tests::comfy_kitchen_cpu_weight_and_embedding_parity`.
It runs only when ComfyUI's embedded Python is found (`COMFY_PYTHON`) and skips otherwise, so it
needs nothing on a normal machine.
