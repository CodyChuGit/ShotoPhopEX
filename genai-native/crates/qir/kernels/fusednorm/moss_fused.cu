// MOSS-Image's SiTU feed-forward activation fused (src/moss/fused.rs, feature `fusednorm`):
//
//   qir_moss_situ  y[m, f] = 4·tanh(g/4)·sigmoid(g) · 25·tanh(h/25),  h = hg[m, f], g = hg[m, F + f]
//
// read straight from the merged `ffn_up_gate` output `hg (M, 2F)`; f32 math, one bf16 rounding.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

// Eight columns a thread (16-byte loads and stores).
__global__ void situ_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ hg, long rows, int F) {
    const int vecs = F / 8;
    const size_t total = static_cast<size_t>(rows) * vecs;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / vecs;
        const int v = static_cast<int>(i % vecs);
        const __nv_bfloat16* base = hg + row * 2 * static_cast<size_t>(F);
        const uint4 hr = reinterpret_cast<const uint4*>(base)[v];
        const uint4 gr = reinterpret_cast<const uint4*>(base + F)[v];
        const __nv_bfloat16* he = reinterpret_cast<const __nv_bfloat16*>(&hr);
        const __nv_bfloat16* ge = reinterpret_cast<const __nv_bfloat16*>(&gr);
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float h = __bfloat162float(he[j]);
            const float g = __bfloat162float(ge[j]);
            const float a = 4.f * tanhf(g * 0.25f) / (1.f + __expf(-g));
            o[j] = __float2bfloat16(a * 25.f * tanhf(h * 0.04f));
        }
        reinterpret_cast<uint4*>(y + row * static_cast<size_t>(F))[v] = *reinterpret_cast<const uint4*>(o);
    }
}

bool misaligned(const void* p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) != 0; }

}  // namespace

// Return codes: 0 ok, 1 a shape the kernel does not take, 2 a misaligned pointer, else CUDA's.
extern "C" int qir_moss_situ(void* y, const void* hg, long rows, int F, void* stream) {
    if (rows <= 0 || F <= 0 || F % 8 != 0) return 1;
    if (misaligned(y, 16) || misaligned(hg, 16)) return 2;
    size_t blocks = (static_cast<size_t>(rows) * (F / 8) + 255) / 256;
    if (blocks > 65535 * 8) blocks = 65535 * 8;
    situ_kernel<<<static_cast<unsigned>(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y),
                                                                                              static_cast<const __nv_bfloat16*>(hg), rows, F);
    return static_cast<int>(cudaGetLastError());
}
