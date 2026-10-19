# Architecture

The authoritative plan is `docs/comfy/plan-native-engine.md` in the PhotoshopEX repository
(§3 for the picture, §3.2 for the layers, §3.4 for the process model, §3.5 for platforms). This
file restates the contracts a contributor needs while working here.

## Five one-way layers

```
gn-perception (SAM 3.1 via ONNX Runtime)      ── own runtime, no dependency on the layers below
gn-weights    (BYOM loader)                   ── depends on gn-kernels (device buffers)
gn-models     (thin definitions)              ── depends on gn-blocks, gn-weights
gn-blocks     (the library)                   ── depends on gn-kernels
gn-kernels    (the GPU behind a trait)        ── depends on nothing of ours
```

- **gn-kernels** owns the GPU: a `Kernels` trait (GEMM bf16/FP8/NVFP4, attention, norms, RoPE,
  SwiGLU, conv, upsample, casts), a device memory arena, streams, CUDA graphs. CUDA is the
  first implementation (sm_120 first, sm_89 second; Windows and Linux from the same `.cu`);
  Metal comes later as a second crate. Fatbins are prebuilt and committed so nobody but a
  kernel author needs nvcc.
- **gn-blocks** is written against the trait and tested block by block against tiny reference
  tensors generated once from diffusers.
- **gn-models** definitions are configuration and wiring: a few hundred lines each. A port that
  needs a new block puts it in `gn-blocks` with its own parity test.
- **gn-weights** reads official safetensors (mmap, hardened parser, no pickle ever), GGUF, and
  ComfyUI's repack naming through per-model key maps; indexes a folder into "what fits which
  slot".
- **gn-perception** binds ONNX Runtime (`ort`) with the CUDA/TensorRT providers for SAM 3.1 and
  future detectors; engine caches are per GPU + runtime version.

Around them: **gn-harness** (this repository's yardstick), **gn-engine** (the worker process:
channel server, residency, VRAM budget) and **gn-sdcpp** (the sd.cpp FFI behind the same worker
protocol).

## The editor's contract

PhotoshopEX talks to every engine through its `GenerativeBackend` trait
(`crates/genai/src/lib.rs`: `health`, `run`, `model_files`, `prepare`, `free`). The native
engine is one implementation (`NativeBackend`, the editor side of the worker channel). Routing
(`auto: native → sdcpp → comfy`, per-model overrides) lives in the editor.

## The worker process

The engine runs as `photocraft.exe --engine --channel <name>`: a named pipe on Windows, a Unix
domain socket elsewhere, length-prefixed messages, pixels copied first (shared memory later), a
process group on Unix. A CUDA fault ends the worker, never the editor.

## Platforms

Windows and Linux are first-class from day one; every phase closes on both. macOS arrives as a
`Kernels` implementation over Metal and the CoreML provider for perception.
