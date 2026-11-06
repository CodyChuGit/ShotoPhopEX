# gn-segment

SAM 3.1 and ViTMatte on [candle](https://github.com/huggingface/candle), used by ShotoPhopEX's
native engine for Select Subject, Select by Text, Object Selection and edge refinement.

- **Checkpoint validation.** `weights::validate(path)` reads only the safetensors header and
  reports missing tensors and shape mismatches before anything is loaded.
- **Model parts.** The vision backbone and both FPNs, the CLIP text encoder and resizer, the
  geometry encoder, the detector's scoring and segmentation heads, and the interactive prompt
  and mask encoders. Video propagation and memory modules are not included.
- **Flows.** Text prompts (`detect`), points and boxes (`interactive`), and ViTMatte alpha
  refinement of a mask's edge.

CPU is the default and is what the tests run on; the `cuda` feature enables candle's CUDA
backend. The tests compare against small recorded fixtures in `tests/fixtures/`; the tests that
need a real checkpoint skip when it is not present.
