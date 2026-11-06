# Speed levers for DiT image generation on a consumer RTX 5090 (sm_120), October 2026

Scope: FLUX-class 12B (FLUX.1-dev, Krea 2) and Qwen-Image / Qwen-Image-Edit-2511 (20B) on a GeForce RTX 5090 (GB202, sm_120, 32 GB GDDR7). Every figure is dated and sourced; numbers measured on datacenter Blackwell (sm_100) or on workstation sm_120 parts (RTX PRO 5000/6000) are flagged as such. "Inference" sections are my own arithmetic or reading, not claims from a source.

Hardware baseline used throughout (third-party spec aggregators; NVIDIA's whitepaper was not retrieved): RTX 5090 = 32 GB GDDR7, 512-bit, 1,792 GB/s; dense tensor peaks ~419 TFLOPS FP16/BF16, ~838 TFLOPS FP8, ~1,676 TFLOPS FP4 (the "3,352 AI TOPS" marketing figure is the sparse FP4 rate) — [Flopper spec sheet](https://flopper.io/gpu/nvidia-geforce-rtx-5090-32gb); [Spheron](https://www.spheron.network/blog/nvidia-rtx-5090-specs/). Club386's review table lists FP8 1,676 / FP4 3,352, i.e. sparse numbers — [Club386](https://www.club386.com/nvidia-geforce-rtx-5090-review/).

---

## KQ1. Measured step times on RTX 5090: bf16 vs fp8 vs NVFP4 vs SVDQuant (FLUX / Krea 2 / Qwen-Image-Edit-2511)

### Takeaway
On a 5090 the consistent, repeatable picture is: FP8 ≈ 1.6× faster than BF16 per DiT step; NVFP4 (mixed, ComfyUI/cu130) ≈ 1.6–2.6× faster than FP8 depending on how many layers were left in FP8/BF16 for quality; the whole-prompt gain is smaller because text encoder, VAE and fixed overhead are untouched. For Qwen-Image-Edit-2511 specifically the only published 5090 numbers are a single community NVFP4 card (4 steps, 832×1280: BF16 9.80 s, FP8 6.09 s, NVFP4 3.45 s total). No independent benchmark of 2511 exists yet.

### Cited Findings

**Qwen-Image-Edit-2511 (20B DiT)**
- RTX 5090 (400 W), ComfyUI, torch 2.9.1+cu130, 832×1280, 4 steps, CFG 1.0, second run: NVFP4 2.52 it/s (≈0.40 s/step), total 3.45 s; FP8-mixed 1.02 s/it, total 6.09 s; BF16 1.62 s/it, total 9.80 s. Author says only "rough calibration, but the quality seems okay". Layer recipe: `attn.to_q/k/v/out`, `img_mlp.*` in NVFP4; `img_mod.1`, `txt_mlp.*`, `txt_mod.1` in FP8 e4m3fn. Undated card (model is Dec 2025+; torch 2.9.1 implies ~Dec 2025–Feb 2026) — [Bedovyy/Qwen-Image-Edit-2511-NVFP4](https://huggingface.co/Bedovyy/Qwen-Image-Edit-2511-NVFP4)
- SVDQuant W4A4 INT4 build of 2511 running on comfy-kitchen nodes: "1.66–1.68 s/it on RTX 5090", earlier version ~3 s/it before fusing the LoRA-up; claims "on par with Nunchaku"; resolution/steps not stated so this is not comparable to the card above. Undated (citation says 2025) — [LAXMAYDAY/qwen-image-edit-2511-int4-comfy](https://huggingface.co/LAXMAYDAY/qwen-image-edit-2511-int4-comfy)

**Qwen-Image-2512 (same 20B architecture, text-to-image), the best-controlled 5090 benchmark found**
- RTX 5090 32 GB (600 W power limit), ComfyUI 0.37.0, PyTorch 2.11.0+cu130, driver 580.105.08, 1328×1328, runs 1–4 averaged. 50 steps cfg 4: BF16 (40.9 GB, RAM offload) 75.5 s; official FP8 e4m3fn 76.6 s; NVFP4 42.5 s (1.80× FP8). Lightning 4-step: BF16 8.6 s, FP8 3.6 s, NVFP4 2.3 s (1.57× FP8). Lightning 8-step: 11.8 / 6.7 / 4.0 s (1.69× FP8). Peak VRAM (nvidia-smi, ComfyUI dynamic management, not minimums): 31.3 / 29.4 / 30.8 GB. Undated (torch 2.11 ⇒ mid-2026) — [MESHIVEAI/Qwen-Image-2512-NVFP4](https://huggingface.co/MESHIVEAI/Qwen-Image-2512-NVFP4)
- Quality on the same card (20 held-out prompts vs BF16): FP8 LPIPS 0.248 / SSIM 0.749; NVFP4 LPIPS 0.437 / SSIM 0.610. Text rendering: EN+ZH (35 images) BF16 35/35, FP8 35/35, NVFP4 34/35; Korean (10) BF16 4/10, FP8 2/10, NVFP4 0/10 — [MESHIVEAI/Qwen-Image-2512-NVFP4](https://huggingface.co/MESHIVEAI/Qwen-Image-2512-NVFP4)

**FLUX.1 / Krea 2 (12B class)**
- NVIDIA official (TensorRT 10.8 runtime, not PyTorch), 14 May 2025, RTX 5090 end-to-end pipeline: FLUX.1-dev 30 steps FP16 10,931 ms (low-VRAM mode), FP8 6,681 ms, FP4 3,853 ms (FP4 = 1.73× FP8); FLUX.1-schnell 4 steps FP16 4,427 ms, FP8 913 ms, FP4 591 ms (1.55× FP8). RTX 4090 FP8 reference: 10,620 ms / 3,385 ms (low-VRAM). FP4 fully-connected layers alone reach "up to 3.1× the performance of FP8". Pipeline VRAM: FP16 39.3 GB, BF16 35.7 GB, FP8 24.6 GB, FP4 21.67 GB (low-VRAM mode 23.9 / 23.9 / 14.9 / 11.1 GB). Transformer = 98 % of latency at 28 steps — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- Krea 2 (Comfy-Org repack, ComfyUI logs it as FLUX type, 28 blocks) on RTX 5090: Kijai (Comfy Org) reports NVFP4 "about 15 % faster than FP8" and explains "not all layers could use nvfp4 matmuls due to rather bad quality loss"; the first upload was re-uploaded because it "doesn't allow the fast nvfp4 matmuls". A user's mixed-precision build (8.8 GB, some layers BF16/MXFP8) ran 1.35 s/it vs 2.21 s/it for the Comfy-Org NVFP4 file (GPU not stated). RTX 4090 user: `--fast` FP8 matmul added 13–15 % (1.63→1.88 it/s, 1024 px, 8 steps). Posts dated Jun 25–Jul 17 (year not shown; Krea 2 shipped 2026) — [Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- Profiling of Krea 2 in ComfyUI (same thread, Jul 17): DiT sampling = 83.6 % of wall time; inside the DiT attention 41 %, GEMM 50 %, norms 8.8 % — [Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- FLUX SRPO NVFP4-mixed live demo on RTX 5090: ~5.7 s for 40 steps, ~14 GB VRAM, vs BF16 ~12–14 s and ~26 GB (author: "more than twice" the time). Relative it/s across formats (GPU mostly unnamed, "SimplePod"/RTX PRO 6000, 2048 px, heavy sampler): FLUX.1-dev GGUF Q8 3.54 it/s, NVFP4 +118 %, BF16 +28 %, FP8-scaled +19 %; FLUX.1 Kontext Q8 1.83 it/s, NVFP4 +93 %, BF16 +14 %, FP8 +9 %; Z-Image-Turbo 1536² Q8 2.26 it/s, NVFP4 4.23 it/s (+87 %). Wiki last edited 17 Jan 2026 — [FurkanGozukara wiki](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/BF16-vs-GGUF-FP8-Scaled-NVFP4-Speed-and-Quality-Compared-ComfyUI-CUDA-13-Gains-FLUX-2-Klein-9B)
- FLUX.2-dev NVFP4-mixed on RTX 5090 (ComfyUI 0.8.2, torch 2.9.0+cu130, SageAttention 2.2.0, driver 591.44, OC'd): 30 steps at 1024² ≈ 14.1 s (~2.14–2.21 it/s), 1408² ≈ 30 s, 2048² ≈ 74.5 s (2.25–2.29 s/it). Snapshot 10 Jan 2026 — [r/StableDiffusion thread by comfyanonymous (snapshot)](https://reddit.sentinel-team.org/posts/1q7ywqo/snapshots/2026-01-10T03%3A01%3A18.43366Z)
- Comfy Org blog (9 Jan 2026): Blackwell NVFP4 "roughly 2× the performance of fp8 or bf16/fp16 models" when everything works; benchmarked on RTX 5070 Ti and RTX PRO 6000, 1024², batch 1 (charts not in text) — [Comfy blog](https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia)

**SVDQuant / Nunchaku on 5090**
- MIT Han Lab (21 Feb 2025): SVDQuant INT4 and NVFP4 both ≈3× faster than BF16 FLUX.1 on RTX 4090 and 5090 with 16-bit attention retained; NVFP4 on 5090 ≈1.3× faster than INT4 on 4090; 3.6× model compression, 3.5× less memory. No absolute latencies — [hanlab.mit.edu](https://hanlab.mit.edu/blog/svdquant-nvfp4)
- Nunchaku README: NVFP4 "3.1× faster than both BF16 and NF4" on RTX 5090 (model not stated; FLUX context) — [nunchaku README (mirror)](https://github.com/nunchux-ai/nunchaku); [SVDQuant paper](https://arxiv.org/abs/2411.05007)

### Inferences
- Per-step ratios from the Bedovyy card: FP8/BF16 = 1.59×, NVFP4/FP8 = 2.57×, NVFP4/BF16 = 4.1× on the DiT step alone; but whole-prompt ratios are 1.61× and 1.77× because ~1.9–2.0 s of each run (text encoder, VAE, graph overhead) is unchanged: NVFP4 run = 3.45 s of which only ~1.6 s is sampling (4 × 0.40 s). For 4-step Lightning workflows the non-DiT part is already >50 % of wall time on NVFP4.
- The 2512 card's 50-step BF16 (75.5 s, offloaded) ≈ FP8 (76.6 s) result shows ComfyUI's FP8 path is not using fast FP8 matmul by default there (or the CFG-doubled run is compute-bound at bf16 upcast); the Lightning 4-step case (8.6 vs 3.6 s) exposes the offload penalty of a 40.9 GB BF16 model on a 32 GB card.
- Krea 2's modest 15 % NVFP4 gain versus 1.6–2.6× for Qwen shows the gain is proportional to the fraction of GEMM FLOPs actually running on FP4 tensor cores; ComfyUI's quality-driven mixed recipes keep modulation/text-MLP layers in FP8 and a few blocks in BF16, and attention (41 % of DiT time in Krea 2) is untouched.

### Gaps
- No independent, controlled bf16/fp8/NVFP4 benchmark of Qwen-Image-Edit-2511 on a 5090 exists; the single card above has no VRAM numbers and no ComfyUI version.
- No SVDQuant NVFP4 absolute step times for Qwen-Image-Edit-2511 on a 5090 (Nunchaku's Qwen releases stop at Edit-2509 as of v1.2.0, 12 Jan 2026; see KQ3).
- No official Comfy-Org 5090 chart data could be read (blog charts are images).

---

## KQ2. Does ComfyUI's NVFP4 path use real FP4 tensor cores on sm_120, and what fraction of theoretical is reached?

### Takeaway
Yes: sm_120 has native block-scaled FP4 MMA via `mma.sync ... kind::mxf4nvf4.block_scale` (register-based, no TMEM/tcgen05), and ComfyUI's NVFP4 GEMMs live in comfy-kitchen's CUDA/cuBLAS backend gated at SM ≥ 10.0 and requiring a cu130 PyTorch; without cu130 the fallback can be up to 2× slower than FP8. Best measured dense NVFP4 GEMM efficiency on an sm_120 part is ~81 % of peak (CUTLASS, large square GEMM, RTX PRO 5000); end-to-end DiT gains are far below that because attention, norms, activation quantisation and non-FP4 layers dominate.

### Cited Findings
- comfy-kitchen ("Fast kernel library for Diffusion inference", Apache-2.0): backends eager/cuda/triton/hip/ascend; NVFP4 layout = "Block quantization with 16-element blocks", gated "SM ≥ 10.0 (Blackwell)"; MXFP8 also SM ≥ 10.0; FP8 SM ≥ 8.9; install extra `comfy-kitchen[cublas]` described as "CUBLAS for NVFP4 (+Blackwell)"; default Linux CUDA build targets include `120f`; prebuilt wheels need CUDA runtime ≥ 13.0, PyTorch ≥ 2.7. No performance numbers — [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen); backend matrix lists `scaled_mm_nvfp4` on eager and CUDA only (no Triton/HIP) — [comfy-kitchen on PyPI](https://pypi.org/project/comfy-kitchen/)
- ComfyUI "only supports NVFP4 acceleration if you are running PyTorch built with CUDA 13.0 (cu130)"; otherwise sampling "up to 2× slower than fp8" (9 Jan 2026) — [Comfy blog](https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia)
- sm_120 FP4 instruction path: `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`; results flow through registers as on Ampere — [Theodore-Liu/sm120-fp4](https://github.com/Theodore-Liu/sm120-fp4); [backend.ai "Is DGX Spark actually Blackwell?" (Feb 2026)](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)
- SM120 has no `tcgen05.*`, no Tensor Memory (SM100: 256 KB/SM), shared memory 99 KiB/block vs 228 KiB, cluster size 1 only; accumulators must live in registers/SMEM ⇒ smaller GEMM tiles — [Blackwell GPU Wiki: SM100 vs SM120](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/). NVFP4 listed as native on both SM100 and SM120; SM100 has dedicated NVFP4→BF16 dequant in tcgen05, SM120 does it in software — [Blackwell GPU Wiki: number formats](https://0xsero.github.io/blackwell-gpu-wiki/fundamentals/number-formats/)
- Measured on **RTX PRO 5000 72 GB (GB203, sm_120, 110 SMs — not a 5090)**, CUDA 13.0, M=N=K=8192: NVFP4 block-scale (CUTLASS 4.5) 975.1 TFLOPS = 80.7 % efficiency; FP8 E4M3 (cuBLAS) 574.9 TFLOPS = 95.0 %; BF16 (cuBLAS) 280.4 TFLOPS = 96.7 %; cuBLAS does not support MXFP4 GEMM there. Page states it was "generated by AI without author review" and has internal inconsistencies (implied FP4 peak 1,208 vs listed 1,175) — [zartbot SM_120 characterization](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
- TensorRT FP4 fully-connected layers on RTX 5090: "up to 3.1× the performance of FP8" (14 May 2025) — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- Krea 2: the first Comfy-Org NVFP4 upload "doesn't allow the fast nvfp4 matmuls" and had to be re-uploaded; the mere file format does not guarantee the fast kernel path — [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- Consumer sm_120 FP4 peak per SM is "roughly comparable to Hopper's FP8" rather than the generational jump SM100 gets — [Blackwell GPU Wiki: tcgen05 and TMEM](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/)
- FP4 quality: NVFP4 = E2M1 values, E4M3 scale per 16-value block, FP32 per-tensor scale; ~4.5 bits/value vs MXFP4 ~4.25 — [NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference); [NVIDIA Research](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/)

### Inferences
- Scaling the RTX PRO 5000 ratios: a 5090 (170 SMs, ~1,676 TFLOPS dense FP4 peak) should reach roughly 1,300–1,400 TFLOPS in an ideal large NVFP4 GEMM, ~4.7× its ~280 TFLOPS BF16 cuBLAS rate. The observed 2.5–4× per-step gains on Qwen (Bedovyy card) are consistent with GEMMs being ~60–70 % of step time at bf16 and the rest (attention, norms, dynamic activation quantisation, FP8 layers) staying fixed.
- comfy-kitchen's `scaled_mm_nvfp4` is a cuBLAS(Lt) call, not a hand-written kernel, so "fraction of theoretical" is whatever cuBLASLt achieves on sm_120 at DiT shapes (M ≈ 4–8k tokens, K/N 3–12k); nobody has published that number for a 5090.
- The "up to 2× slower without cu130" warning means the fallback is a dequantise-to-bf16 eager path; the NVFP4 file alone is not a speed lever, the kernel stack (driver R580+, cu130 torch, comfy-kitchen CUDA wheel) is.

### Gaps
- No published measurement of ComfyUI's achieved NVFP4 TFLOPS or % of peak on an RTX 5090 (the only efficiency figure is CUTLASS on an RTX PRO 5000).
- Could not confirm from comfy-kitchen's README whether the CUDA path is cuBLASLt block-scaled FP4 or CUTLASS; the `[cublas]` extra strongly implies cuBLASLt.
- Conflicting sources on whether `wgmma` exists at reduced throughput on sm_120 (0xsero wiki) or fails to assemble (backend.ai); irrelevant for FP4 but flags that both wikis are community-maintained.

---

## KQ3. Nunchaku / SVDQuant: supported models, speedups, quality, licence, PyTorch-only?

### Takeaway
Nunchaku (Apache-2.0, v1.2.0 of 12 Jan 2026) ships 4-bit W4A4 SVDQuant for FLUX.1-dev/Krea/Kontext/tools, Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-2509 (+Lightning), Z-Image-Turbo, SANA, with ~3× over BF16 on 4090/5090 and NVFP4 quality close to BF16 (ImageReward 0.937 vs 0.953 on FLUX.1-dev). Qwen-Image-Edit-2511, Qwen-Image-2512/2.1, FLUX.2 and Krea 2 are not in its model list; it is a CUDA kernel library with PyTorch/diffusers-style Python bindings and a ComfyUI plugin, not a standalone runtime.

### Cited Findings
- Supported models and dates: FLUX.1-dev, FLUX.1-Krea-dev (2025-07-31), FLUX.1-Kontext (2025-06-29), FLUX.1-tools (2025-02-04), Qwen-Image (2025-08-15) + Lightning (2025-08-27), Qwen-Image-Edit (2025-09-09), Qwen-Image-Edit-2509 (2025-09-24/25), Z-Image-Turbo (2025-12-20), SANA (2025-01-23). Not listed: Qwen-Image-Edit-2511, Qwen-Image-2512/2.1, FLUX.2. Licence Apache-2.0. v1.2.0 (2026-01-12): 20–30 % Z-Image gain, LoRA with native ComfyUI nodes, INT4 for 20-series, ComfyUI 0.7 compat. Python backend added 2025-08-15; ComfyUI via separate ComfyUI-nunchaku plugin — [nunchaku README (mirror)](https://github.com/nunchux-ai/nunchaku)
- NVFP4 on RTX 5090 (news 2025-02-20): ~3× over BF16; "NVFP4 delivers superior image quality compared to INT4"; FLUX 3.6× smaller; 4 GiB minimum with per-layer CPU offload; Qwen-Image async offload ~3 GiB "with no performance loss" (v1.0.0, 2025-09-04) — [nunchaku README (mirror)](https://github.com/nunchux-ai/nunchaku)
- Quality, FLUX.1-dev, MJHQ prompts (21 Feb 2025): BF16 ImageReward 0.953; INT4 0.908 / LPIPS 0.322 / PSNR 18.5; INT4+SVDQuant 0.935 / 0.223 / 21.0; NVFP4 0.928 / 0.244 / 20.3; NVFP4+SVDQuant 0.937 / 0.208 / 21.4 — [hanlab.mit.edu](https://hanlab.mit.edu/blog/svdquant-nvfp4)
- NVIDIA ModelOpt FP4 variants on FLUX.1-dev (14 May 2025): BF16 ImageReward 1.118 / CLIP-IQA 0.926 / CLIP 30.15; FP4 PTQ 1.096 / 0.923 / 29.86; FP4 QAT 1.119 / 0.928 / 29.92; FP4 SVDQuant 1.108 / 0.927 / 30.07; SVDQuant "adds some runtime overhead" (low-rank branch) — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- SVDQuant paper: 3.1× speedup over W4A16 on 5090 — [arXiv 2411.05007](https://arxiv.org/abs/2411.05007)
- Nunchaku docs have a Qwen-Image-Edit usage page (no timings) — [nunchaku.tech docs](https://nunchaku.tech/docs/nunchaku/usage/qwen-image-edit.html)
- A May 2026 Forge-Neo tutorial describes running Qwen-Image-Edit-2511 "using Nunchaku" (secondary source; no repo named) — [digitalcreativeai.net](https://www.digitalcreativeai.net/en/post/how-speed-up-qwen-image-edit-2511-nunchaku-forge-neo)
- Independent SVDQuant W4A4 build of 2511 exists on comfy-kitchen's "kitchen-native" CUDA path, "kernels follow Nunchaku conventions", 1.66–1.68 s/it on 5090 — [LAXMAYDAY card](https://huggingface.co/LAXMAYDAY/qwen-image-edit-2511-int4-comfy)
- Nunchaku v0.2.0 (2025-04-05) added First-Block Cache; v0.3.0 (2025-06-01) Double FB Cache and multi-batch — [nunchaku README (mirror)](https://github.com/nunchux-ai/nunchaku)

### Inferences
- Nunchaku's quality edge over plain NVFP4 PTQ comes from the SVD low-rank branch absorbing outliers; that branch is extra FP16 GEMM work per layer, which is why NVIDIA notes runtime overhead and why Nunchaku keeps attention in 16-bit. Its ~3× figure is for FLUX with the whole DiT in W4A4; ComfyUI's mixed NVFP4 recipes trade some of that speed for quality by keeping sensitive layers in FP8/BF16.
- Nunchaku is PyTorch-bound (custom CUDA ops exposed as torch extensions, Python model wrappers). A non-Python runtime would have to port the kernels (Apache-2.0 permits) or re-implement W4A4 + low-rank fusion.
- The fetched HF organisation page for nunchaku-tech rendered "None public yet", almost certainly a JS-rendering artefact (the README links `huggingface.co/nunchaku-tech/...` repos); treat the model list from the README as authoritative.

### Gaps
- Whether nunchaku-tech has published an official Qwen-Image-Edit-2511 (or 2512 / 2.1, Krea 2, FLUX.2) SVDQuant checkpoint after v1.2.0 could not be confirmed from a primary source.
- No Nunchaku timing for any Qwen model on a 5090 in absolute seconds.

---

## KQ4. FP8 scaled matmul (`--fast fp8_matrix_mult`), attention kernels on sm_120, CUDA graphs / torch.compile

### Takeaway
`--fast fp8_matrix_mult` gives ~13–15 % on a 4090 for Krea 2 and underlies the ~1.6× FP8-vs-BF16 step gain on the 5090. For attention on sm_120: FlashAttention-3 is Hopper-only; FlashAttention-4's fast path needs tcgen05/TMEM that GB202 lacks (a downstream sm_120 FA4 build only matches FA2 ±10 %); SageAttention 2/2++ (INT8/FP8 QK) runs on the 5090 at "560 TOPS, 2.7× FA2", and SageAttention3 (FP4 attention, Apache-2.0, code released 27 Sep 2025) claims 1,038 TOPS, ~5× the fastest FA on a 5090, with an accuracy caveat. No one has published CUDA-graph/torch.compile ms-per-step for a DiT on a 5090; the only quantified launch-overhead figure is 5–10 µs per kernel.

### Cited Findings
- ComfyUI `--fast` options: `fp16_accumulation`, `fp8_matrix_mult`, `cublas_ops`, `autotune`, all "experimental… may crash ComfyUI or reduce image quality" — [ComfyUI command-line args](https://www.mintlify.com/Comfy-Org/ComfyUI/configuration/command-line-args). FP16 accumulation PR (Feb 2025): RTX 3090 ~4→5 it/s; ~10–15 % at batch 1, 25–33 % at batch ≥2, attributed to a CPU/RAM bottleneck at single images — [ComfyUI PR #6453](https://github.com/Comfy-Org/ComfyUI/pull/6453)
- Krea 2 on RTX 4090: `--fast` FP8 matmul 1.63→1.88 it/s (+13–15 %) at 1024 px, 8 steps — [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- SageAttention (Apache-2.0): 15 Feb 2025 "560T on RTX 5090, 2.7× faster than FlashAttention2"; SageAttention2++ code 1 Jul 2025 (pip 2.2.0); SageAttention3 (NeurIPS 2025 Spotlight) code released 27 Sep 2025 in `sageattention3_blackwell`, CUDA ≥ 12.8; README recommends SageAttention2 for precision-sensitive use; replace attention only inside the DiT for image/video models — [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)
- SageAttention3 paper: FP4 attention 1,038 TOPS on RTX 5090, ~5× the fastest FlashAttention on that card; "FlashAttention3 can only run on Hopper GPUs, so FlashAttention2 is already the fastest on RTX5090"; HunyuanVideo end-to-end 164 s vs 490 s with FA2 — [arXiv 2505.11594](https://arxiv.org/abs/2505.11594). Community wheels for sm120a: [sanjxz/sageattention-2.2.0-blackwell-sm120a](https://huggingface.co/sanjxz/sageattention-2.2.0-blackwell-sm120a)
- Later papers measuring on 5090: VC-Attention is within 5 % of SageAttention3's 4-bit kernel; another notes fully-FP4 attention "often fails to translate this throughput into end-to-end speedups due to non-matmul costs" — [VC-Attention](https://arxiv.org/pdf/2609.15810); [MpFA](https://arxiv.org/html/2609.33135v1)
- FlashAttention-4 on sm_120: gist investigation (6 Mar 2026, CUDA 13.1, nvidia-cutlass-dsl 4.4.1): interface asserts `arch // 10 in [9, 10, 11]`; ptxas rejects `tcgen05.alloc`, `tcgen05.mma`, `.cta_group::2` for `sm_120a`; SASS for sm_120a shows only `HMMA.16816.F32.BF16`, no `UTC*`/`UTMA*`; author concludes FA4 cannot run on GB202 (one person's analysis) — [solatticus gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739). Downstream FA4 build for SM120/121 (BSD-3) uses SM80-style `mma.sync` for BF16/FP16, "roughly the same compute as FA2 plus a small dispatch overhead"; on DGX Spark (sm_121a) it is 1.04–1.10× slower than FA2 at S=512–4096 and 1.71× slower at S=128; head_dim > 128 unsupported (99 KB SMEM); recommends FA2 for raw throughput — [SecondNatureComputing/flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120). FA issue #1987: "invalid argument" on RTX PRO 6000 (sm_120) with the official package — [Dao-AILab/flash-attention #1987](https://github.com/Dao-AILab/flash-attention/issues/1987)
- cuDNN-backed SDPA on sm_120 reaches 203.6 TFLOP/s, 97 % of the BF16 roofline for encoder shapes (Sep 2026, per that page's own test) — [keenable.ai](https://select.keenable.ai/r/sm120-attention-for-encoder-models/yWskIdfUzZDXx2FSsDhXIQ)
- Kernel launch overhead: "on fast GPUs such as the RTX 5090, the CPU-side kernel launch overhead (5–10 μs per kernel)" is a significant fraction; `torch.compile(mode="reduce-overhead")` captures CUDA graphs — [FastUSP, arXiv 2602.10940](https://arxiv.org/pdf/2602.10940)
- PyTorch "Diffusion, Fast" (A100 400 W, SDXL 30 steps, Nov 2024 revision): fp32 7.36 s → bf16 4.63 s → SDPA 3.31 s → torch.compile max-autotune (CUDA graphs) 2.54 s → fused QKV 2.52 s → dynamic int8 2.43 s; scheduler `sigmas` indexing on GPU causes a CPU–GPU sync that becomes visible once the denoiser is compiled — [PyTorch blog](https://pytorch.org/blog/accelerating-generative-ai-3/)
- torch.compile + CUDA graphs gave 2.14–2.16× on top of SDPA for multimodal generation models at small batch; gains shrink as batch grows; requires static shapes — [arXiv 2410.00215](https://arxiv.org/pdf/2410.00215). Regional compilation of repeated blocks gives the same runtime speedup with 8–10× less compile time — [diffusers docs](https://huggingface.co/docs/diffusers/en/optimization/fp16)
- Comfy-WaveSpeed: First Block Cache 1.5–3.0× "with acceptable accuracy loss"; FP8 + torch.compile unsupported pre-Ada — [Comfy-WaveSpeed](https://www.sourcepulse.org/projects/1916168)

### Inferences
- torch.compile's 3.31→2.54 s on SDXL (23 %) is the cleanest published "dispatch + fusion" number for a diffusion model; on a 5090, which finishes kernels ~3–5× faster than an A100 while the Python side is no faster, the unhidden launch fraction per step is larger, not smaller. ComfyUI does not run the DiT under CUDA graphs by default, so each step re-dispatches thousands of kernels.
- Attention is the ceiling once GEMMs go FP4: with attention at 41 % of DiT time (Krea 2 profile) and GEMMs at 50 %, a 4× GEMM speedup alone caps the DiT at ~1.9×; SageAttention2++ (2.7× FA2) is the matching lever on sm_120, SageAttention3 FP4 is faster still but trades accuracy.

### Gaps
- No measurement of CUDA-graph or torch.compile gains for FLUX/Qwen on an RTX 5090 in ms/step.
- No SageAttention3 end-to-end numbers for image DiTs (only HunyuanVideo), and no published quality deltas of FP4 attention on FLUX/Qwen.
- No official statement from Dao-AILab on sm_120 FA4 support; the "silicon-level" claim is a community inference.

---

## KQ5. Step distillation (Lightning 4/8-step) and caching (TeaCache, First-Block Cache)

### Takeaway
Lightning LoRAs are the single largest lever: 40–50 steps → 4 or 8 (10–12× fewer DiT passes, CFG off halves work again). For Qwen-Image-Edit-2511 lightx2v ships a 4-step V1.0 bf16 LoRA and a fused FP8 model; the 8-step was promised "later". Caching on Qwen-Image is marginal: FBC ≈1.37× at PSNR 32.7 vs baseline in one paper, TeaCache 0.97× (no gain) in the same paper but 1.9× on H200 per vLLM-Omni; caches are auto-disabled with CFG in some frameworks and add little at 4 steps.

### Cited Findings
- lightx2v Qwen-Image-Edit-2511-Lightning: 4-step V1.0 bf16 LoRA and a fused FP8 model; maintainer: 8-step "would come later", run the 4-step with 8-step inference meanwhile — [lightx2v repo](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning); [discussion #2](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/discussions/2); [ModelTC/LightX2V-Qwen-Image-Lightning](https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning)
- Qwen-Image 2.1 Lightning 4-step DMD LoRA (13 Oct 2026), T2I and edit, CFG disabled, default 40 steps → 4; side-by-sides vs 40-step base but no metrics — [ComfyUI Wiki news](https://comfyui-wiki.com/en/news/2026-10-13-qwen-image-2-1-lightning)
- Original Qwen-Image Lightning: 8-step V1.0 on 8 Aug 2025, 4-step V1.0 on 11 Aug 2025, Apache-2.0; recommended CFG 1.0 — [promptzone](https://www.promptzone.com/neha_sullivan/qwen-image-fast-ai-art-tool-for-comfyui-unveiled-og1); [sbcode.net](https://sbcode.net/genai/qwen-lighting-lora)
- Measured effect on 5090 (Qwen-Image-2512 NVFP4 card): 50-step cfg 4 = 42.5 s → Lightning 4-step 2.3 s (18.5×) → 8-step 4.0 s — [MESHIVEAI card](https://huggingface.co/MESHIVEAI/Qwen-Image-2512-NVFP4)
- Suggested ComfyUI settings for 2511: bf16 base CFG 4 / 40 steps; fp8 base CFG 4 / 20 steps; fp8 + Lightning CFG 1 / 4 steps — [stablediffusiontutorials (Dec 2025)](https://www.stablediffusiontutorials.com/2025/12/qwen-image-edit-2511.html)
- Caching on Qwen-Image, 50 steps, 1024² (LayerCache paper, Apr 2026): FBC τ=0.05 → 1.37× speedup, PSNR 32.66, SSIM 0.9475; TeaCache thresh 0.35 → 0.97×, PSNR 16.41 — [arXiv 2604.16492](https://arxiv.org/pdf/2604.16492). vLLM-Omni (H200, Dec 2025): Qwen-Image TeaCache rel_l1 0.2 → 1.91×, no quality metrics; docs claim 1.5–2.0× "minimal quality loss" — [vLLM blog](https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration); [vLLM-Omni docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/teacache/). SGLang: TeaCache auto-disabled for Flux/Qwen when CFG is on — [SGLang docs](https://docs.sglang.io/diffusion/performance/cache/index.html)
- Nunchaku ships First-Block Cache and Double FB Cache — [nunchaku README (mirror)](https://github.com/nunchux-ai/nunchaku)

### Inferences
- Caches need a run of similar consecutive steps to skip; at 4–8 Lightning steps there is almost nothing to skip, so caching and distillation are substitutes, not multipliers. For PhotoCraft's interactive fills (Lightning, CFG 1) caching is not worth implementing.
- The Lightning + NVFP4 combination moves the bottleneck off the DiT: in the Bedovyy 2511 run the DiT is ~1.6 s of a 3.45 s prompt.

### Gaps
- No published quality metrics (vs 40–50-step base) for the 2511 or 2.1 Lightning LoRAs; only side-by-side images.
- No measurement of the Lightning LoRA stacked on an NVFP4 weight set vs the fused FP8 Lightning model.

---

## KQ6. Memory-bandwidth math and "would NVFP4 make everything faster?"

### Takeaway
At one 1-MP image a 20B DiT step on a 5090 is compute-bound, not weight-bandwidth-bound: streaming 40 GB of bf16 weights takes ~22 ms at 1.79 TB/s, while the ~280 TFLOP of GEMM work per pass takes ~1 s at bf16 rates. NVFP4 helps because the FP4 tensor-core rate is ~4× FP8 and ~8× BF16, and because a 10–11 GB weight set avoids the 32-GB offload cliff that bf16 Qwen (40.9 GB) hits. It does not touch attention, norms, text encoder, VAE, LoRA application or Python dispatch, and it costs measurable quality (LPIPS 0.44 vs FP8 0.25 on Qwen; "noticeable" per several testers).

### Cited Findings
- RTX 5090 memory 1,792 GB/s; dense peaks ~419 BF16 / ~838 FP8 / ~1,676 FP4 TFLOPS — [Flopper](https://flopper.io/gpu/nvidia-geforce-rtx-5090-32gb); [Spheron](https://www.spheron.network/blog/nvidia-rtx-5090-specs/). Measured cuBLAS BF16 GEMM on an sm_120 part: 280 TFLOPS (97 %) — [zartbot](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
- Qwen-Image-2512 BF16 = 40.9 GB and needs RAM offload on 32 GB; FP8 20.4 GB; NVFP4 21.9 GB (file includes FP8/BF16 layers) — [MESHIVEAI card](https://huggingface.co/MESHIVEAI/Qwen-Image-2512-NVFP4). FLUX pipeline VRAM FP16 39.3 → FP8 24.6 → FP4 21.7 GB (TensorRT) — [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- Async offload + pinned memory (ComfyUI, default since Dec 2025): 10–50 % faster sampling only when weights must be offloaded; "not meaningful" if the model fits; PCIe 4.0 x8 gains "less impressive" — [Comfy blog](https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia)
- Krea 2 on 5090: NVFP4 only ~15 % over FP8 because quality forced many layers to stay FP8 — [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- Comfy-Org NVFP4 text encoder (MiniMax-H3 card): "does not require Blackwell GPU to use", i.e. the TE is dequantised, not FP4-accelerated — [Comfy-Org/MiniMax-H3 README (mirror)](https://gitserver.onethingai.com/ai-models/MiniMax-H3/src/branch/master/README.md)
- Quality reports: Qwen-Image-2512 NVFP4 LPIPS 0.437/SSIM 0.610 vs FP8 0.248/0.749; Korean text 0/10 vs 4/10 BF16 — [MESHIVEAI card](https://huggingface.co/MESHIVEAI/Qwen-Image-2512-NVFP4). FLUX.1-dev NVFP4 "noticeable quality degrade", FLUX.2-dev "some degradation but still perfectly usable", Kontext "still very good", SRPO-mixed "almost no difference" (visual, Jan 2026) — [FurkanGozukara wiki](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/BF16-vs-GGUF-FP8-Scaled-NVFP4-Speed-and-Quality-Compared-ComfyUI-CUDA-13-Gains-FLUX-2-Klein-9B). Qwen NVFP4 "faster, but the quality difference was noticeable" (u/MagiRaven, Jan 2026) — [reddit snapshot](https://reddit.sentinel-team.org/posts/1q7ywqo/snapshots/2026-01-10T03%3A01%3A18.43366Z). Krea 2: raw-model NVFP4 "completely useless" for LoRA training (user) — [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6). Counter-evidence with metrics: SVDQuant-NVFP4 FLUX.1-dev ImageReward 0.937 vs BF16 0.953 — [hanlab](https://hanlab.mit.edu/blog/svdquant-nvfp4); ModelOpt FP4 QAT 1.119 vs BF16 1.118 — [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus)
- RTX 40-series has no FP4 tensor cores; NVFP4 files there run at FP8 speed or up to 2× slower — [runaihome guide](https://runaihome.com/blog/comfyui-nvfp4-rtx-speed-guide-2026/); 40-series "works, but not in FP4" — [reddit snapshot](https://reddit.sentinel-team.org/posts/1q7ywqo/snapshots/2026-01-10T03%3A01%3A18.43366Z)

### Inferences (my arithmetic)
- Weight-streaming floor per DiT forward: 20B × 2 B = 40 GB / 1,792 GB/s ≈ 22 ms (bf16); 20 GB ≈ 11 ms (fp8); ~10 GB ≈ 6 ms (fp4). Compute per forward at 1328² (≈6,900 image tokens + text/reference tokens, say 8k): 2 × 20e9 × 8e3 ≈ 3.2e14 FLOP ≈ 1.1 s at 280 TFLOPS bf16, ~0.55 s at ~575 TFLOPS fp8, ~0.3 s at ~1,000 TFLOPS fp4. Ridge point for bf16 GEMM is ~156 FLOP/byte ⇒ a DiT step is weight-bandwidth-bound only below ~160 tokens (~0.04 MP); image generation is far above that, so the user's premise ("bandwidth-bound at 1 image") does not hold for the DiT; it does hold for the text encoder's decode-style passes and for anything that spills over PCIe (PCIe 4.0 x16 ≈ 25–30 GB/s ⇒ 8.9 GB of offloaded bf16 Qwen weights ≈ 0.3 s per pass, matching the 8.6 s vs 3.6 s Lightning gap on the 2512 card).
- Hence NVFP4's two real wins on a 5090: (1) FP4 tensor-core FLOPs for the GEMM fraction of the step (≈50 % of DiT time), (2) VRAM headroom that keeps a 20B model plus TE/VAE/reference latents resident at 1 MP+ without offload. Everything else is unchanged, which is why whole-prompt speedups are 1.6–1.8× when step speedups are 2.5×.
- For PhotoCraft's workloads (Lightning 4–8 steps, CFG 1, inpaint crops often < 1 MP) the fixed ~1.9–2.0 s of non-DiT time in the Bedovyy run is the larger target: Qwen2.5-VL text encoding of prompt + reference image, VAE encode/decode, model (re)load and Python graph execution. An in-process runtime can cut that; NVFP4 cannot.
- Quality: measured deltas range from "metric-indistinguishable" (SVDQuant/QAT FP4 on FLUX, NVIDIA/MIT) to clearly worse (ComfyUI PTQ NVFP4 on Qwen: LPIPS ~1.8× FP8, Korean text broken). The difference is the quantisation method (SVDQuant/QAT vs rough PTQ calibration), not the format. For a product default, FP8-scaled is the safe tier; NVFP4 should be an opt-in with the Comfy-Org/SVDQuant-grade checkpoint, not arbitrary community repacks.

### Gaps
- No source measures the Qwen2.5-VL text encoder or Qwen VAE time separately on a 5090; the ~1.9 s non-DiT figure is derived from one card's totals.
- No published NVFP4 vs FP8 quality study for Qwen-Image-Edit-2511 specifically (only the 2512 T2I card).
- No official NVIDIA statement that GB202 lacks hardware FP4→BF16 dequant; the claim is from a community wiki.

---

## KQ7. Remaining overhead an in-process (non-Python) runtime could remove

### Takeaway
No source quantifies ComfyUI's per-step Python overhead on a 5090 in ms. The best anchors are: 5–10 µs CPU cost per kernel launch (thousands of kernels per DiT step), torch.compile/CUDA-graphs removing ~23 % of SDXL step time on an A100 (more on a faster GPU), and the fact that in a 4-step NVFP4 Qwen-Edit run ~55 % of wall time (≈1.9 s of 3.45 s) is already outside the sampler. The realistic prize for a native runtime is that fixed ~2 s plus whatever fraction of the ~0.4 s/step is launch-bound, not the GEMMs themselves.

### Cited Findings
- 5–10 µs CPU-side launch overhead per kernel on fast GPUs like the RTX 5090; mitigated by CUDA graphs via `reduce-overhead` — [FastUSP](https://arxiv.org/pdf/2602.10940)
- SDXL on A100: compile + CUDA graphs 3.31 s → 2.54 s (30 steps); GPU-side `sigmas` indexing introduces a CPU–GPU sync per step — [PyTorch blog](https://pytorch.org/blog/accelerating-generative-ai-3/)
- ComfyUI FP16-accumulation gains were 10–15 % at batch 1 but 25–33 % at batch ≥2, attributed by the author to a CPU/RAM bottleneck at single images — [ComfyUI PR #6453](https://github.com/Comfy-Org/ComfyUI/pull/6453)
- Krea 2 profile: DiT sampling 83.6 % of wall time ⇒ 16.4 % is TE, VAE, scheduling, I/O — [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)
- Qwen-Edit-2511, 4 steps: totals 3.45 / 6.09 / 9.80 s vs sampling ≈ 1.59 / 4.08 / 6.48 s (NVFP4 / FP8 / BF16) — [Bedovyy card](https://huggingface.co/Bedovyy/Qwen-Image-Edit-2511-NVFP4)
- First-run penalties on 5090 (model load, warm-up): FLUX.2 NVFP4 1024² 39.4 s first vs 14.2 s steady — [reddit snapshot](https://reddit.sentinel-team.org/posts/1q7ywqo/snapshots/2026-01-10T03%3A01%3A18.43366Z)
- TensorRT (fully fused, no Python per step) FLUX.1-schnell 4 steps FP4 = 591 ms end-to-end on 5090 (May 2025) vs community ComfyUI PyTorch FLUX-class NVFP4 runs of several seconds — [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus); [Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)

### Inferences
- Order-of-magnitude: a 60-block Qwen DiT with ~25–40 kernels per block issues ~1,500–2,500 launches per forward ⇒ 10–25 ms of CPU launch time per step at 5–10 µs each, plus Python op dispatch (typically 20–50 µs per PyTorch op) ⇒ realistically 50–150 ms CPU-side per step. At bf16 (1.6 s/step) that is hidden behind the GPU; at NVFP4 Lightning (0.4 s/step) it is 10–35 % and partially exposed, and at smaller inpaint crops it becomes dominant. Cross-checked against the 23 % torch.compile gain on A100/SDXL, a 15–30 % per-step gain from graph capture on a 5090 Lightning run is a reasonable expectation, not a measurement.
- The bigger, better-evidenced win is the fixed ~1.9 s: TE (Qwen2.5-VL 7B over prompt + image), VAE encode of reference/mask region and decode, scheduler and ComfyUI graph execution, HTTP/WebSocket round-trips. TensorRT's 591 ms end-to-end schnell run shows what a fused native pipeline achieves when the TE and VAE are also compiled.

### Gaps
- No ms/step measurement of ComfyUI dispatch overhead on a 5090 for FLUX or Qwen; no CUDA-graph before/after for a DiT on sm_120.
- No breakdown of Qwen-Image-Edit TE vs VAE vs sampler time on a 5090.
