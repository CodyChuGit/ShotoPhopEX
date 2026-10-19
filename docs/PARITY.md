# Parity

A port is done when it matches the reference within these thresholds on the fixture set, not
when it produces a nice picture. The harness applies them (`gn_harness::metrics::Verdict`).

| What | Yardstick | Threshold |
|---|---|---|
| A block (attention, norm, VAE stage…) against its diffusers reference tensors | cosine similarity; max abs error in bf16 | cosine ≥ 0.9995; max abs ≤ 1e-2 |
| A VAE decode against the reference | PSNR | ≥ 50 dB |
| A picture at equal seed, engine vs engine | PSNR **or** SSIM on luma (samplers may legitimately diverge; the softer one decides and a human looks at the contact sheet) | PSNR ≥ 30 dB or SSIM ≥ 0.90 |
| A mask (SAM) against the reference route | IoU; boundary F at 2 px tolerance | IoU ≥ 0.97; F ≥ 0.95 |
| A mask while engines report bounds only (Phase 0) | bounds IoU | ≥ 0.90 |
| Determinism on the native engine | same seed twice | identical bytes |

Latency columns on every table: wall time, the engine's own time, cold start, peak VRAM.

## Fixtures

`crates/gn-harness/fixtures/`: `t2i.json` (12 prompts × 2 seeds at 1024², 4 at 1344×768 and
768×1344), `edit.json` (8 instructions, two with reference pictures), `fill.json` (8 rectangles
given as fractions), `sam.json` (16: text, point, negative point, box prompts over 8 bench
pictures). The bench pictures live in PhotoshopEX's `docs/comfy/bench/images/` (ignored there)
and are referenced by file name.

## References

Generated locally with `gn-bench run --engine comfy …` against PhotoshopEX's ComfyUI route at
Quality `best`; kept under `references/<date>-comfy-<suite>/`, ignored by git. A run folder
holds `<case>.png`, `<case>.json` (`Record`: wall and engine time, seed, the engine's result)
and `summary.md`. `gn-bench compare a b` writes the parity table.
