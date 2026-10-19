# Porting a model

1. **Oracle first.** Pick the Apache-2.0 reference implementation (diffusers; ComfyUI's
   Apache-marked model files; sd.cpp's MIT C++ as a second reading). Never port from a repository
   under a research licence (Qwen's licence covers its "model code"). Record the source file and
   commit in `NOTICE`.
2. **Reference tensors.** Dump per-block inputs and outputs for three prompts at one seed from
   the oracle (small, committed under `gn-blocks/tests/refs/<model>/`); dump full outputs for the
   fixture set (local, ignored).
3. **Blocks before the model.** Add any block the architecture needs to `gn-blocks` with its
   parity test against the dumped tensors. Attention variants, RoPE variants, modulation and
   schedulers go here, never inline in a definition.
4. **The definition.** `gn-models/src/<model>.rs`: config, forward wiring over blocks, the
   conditioning recipe (prompt, references, image, mask → the transformer's inputs), the key map
   into `gn-weights`. Target a few hundred lines; a definition past ~600 lines has a block in it.
5. **Weights.** Key maps for the official layout, ComfyUI's repack naming and GGUF; dtype policy
   per tensor; a loader test on a hand-made file of each kind.
6. **Parity run.** `gn-bench compare` against the reference run; meet `docs/PARITY.md`.
7. **Speed run.** Only now: SageAttention, FP8/NVFP4 GEMMs, fused kernels, CUDA graphs, each
   with a before/after row in the benchmarks and the parity table re-run.
8. **Routing.** Flip the model's parity flag in the editor's routing table; it moves to the
   native engine under `auto` only when it is faster at parity on the owner's card.
