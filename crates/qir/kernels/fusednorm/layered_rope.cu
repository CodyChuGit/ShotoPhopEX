// Qwen-Image-Layered's RoPE (src/model/qwen_image_dit.rs, feature `fusednorm`), compiled at build
// time like qir's other kernels so the worker needs no NVRTC at run time.
//
//   qir_layered_rope  y = (a·cos ⊖ z·sin, a·sin ⊕ z·cos) on adjacent pairs of (B, H, S, 128) bf16
//
// ComfyUI rounds the first product to bf16 before the addcmul, so `a·cos` is rounded on its own
// and the rest is one f32 multiply-add rounded once. The `_rn` intrinsics keep nvcc from fusing
// the multiply and the add (an FMA rounds differently).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 rb(float x) { return __float2bfloat16_rn(x); }

// One thread a pair. Tables are (table_batches, 1, S, 64): shared by every head, per batch item
// when table_batches > 1.
__global__ void layered_rope_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ c,
                                    const __nv_bfloat16* __restrict__ s, uint32_t pairs, uint32_t seq, uint32_t heads, uint32_t table_batches) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    uint32_t p = i % (seq * 64u);
    if (table_batches > 1) p += (i / (heads * seq * 64u)) * seq * 64u;
    const float a = bf(x[2 * i]), z = bf(x[2 * i + 1]), cs = bf(c[p]), sn = bf(s[p]);
    y[2 * i] = rb(__fsub_rn(bf(rb(__fmul_rn(a, cs))), __fmul_rn(z, sn)));
    y[2 * i + 1] = rb(__fadd_rn(bf(rb(__fmul_rn(a, sn))), __fmul_rn(z, cs)));
}

}  // namespace

extern "C" int qir_layered_rope(void* y, const void* x, const void* c, const void* s, uint32_t pairs, uint32_t seq, uint32_t heads,
                                uint32_t table_batches, void* stream) {
    if (pairs == 0) return 0;
    const uint32_t threads = 256;
    const uint32_t blocks = (pairs + threads - 1) / threads;
    layered_rope_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(c),
        static_cast<const __nv_bfloat16*>(s), pairs, seq, heads, table_batches);
    return static_cast<int>(cudaGetLastError());
}
