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

## Reproducing a ComfyUI route (what Phase 2 taught)

A seed-for-seed match with ComfyUI needs four things read from the **running** ComfyUI, not
from its source by eye: the noise (`torch.manual_seed` + CPU `randn` on the model-space latent,
ported in `qir::comfy`), the sigma table of the loaded model object
(`model.model_sampling.sigmas`: Qwen-Image-2.1 is a `FLUX`-type model whose `ModelSamplingFlux`
holds 10 000 exponentially shifted sigmas, not `ModelSamplingDiscreteFlow`'s 1000; the scheduler
then picks from that table), the timestep the DiT is handed (its bf16 rounding), and the prompt
tokens the encoder node produces. Check each against numbers the real objects print, and keep
the calibration: ComfyUI's own INT8 and bf16 routes agree on 26 of the 28 t2i cases, so a
native route at 24–26 of 28 is at parity, and the cases that miss are the same for everyone.

## Edits and fills: compare through the app (what Phase 3 taught)

The product builds an edit's request before any engine sees it: it resamples the picture to
its working size (a 1024² picture goes out at 880²), feathers the mask, wraps the prompt. A
direct `--engine qir` run of the edit suite therefore does not match the app's references
(2 of 8 at the gate) while the same engine behind the app matches 8 of 8. Measure edit and
fill parity with the app's CLI on both routes: `--engine comfy` for the references, then
`--engine comfy --pref integrations.engine=native --pref integrations.runtimePath=… --pref
integrations.modelsPath=…` for the native worker. The conventions that matter beyond §2.1's:
`TextEncodeQwenImage21` resizes every picture to multiples of 32 at `resolution 0`, the vision
tower and the VAE see the same resized picture, `LoadImage` hands over RGB (alpha dropped,
the VAE's alpha channel padded opaque), the mask of a fill is a grey picture as image 2, and the
rows before the target (text and references) are a prefix the model treats as t = 0.
