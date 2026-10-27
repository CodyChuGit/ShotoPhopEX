// FP8 (e4m3) casts for src/fp8.rs, the way ComfyUI's comfy_kitchen does them:
//   dequantize: out = bf16(q) * bf16(scale)          (dequantize_per_tensor_fp8)
//   quantize:   out = e4m3(clamp(x, -448, 448))      (quantize_per_tensor_fp8 with scale 1)
// Both round to nearest even; the product of two bf16 values is exact in fp32, so the bf16
// multiply below rounds once, as torch's does. Eight elements a thread, a scalar tail.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void dequant_kernel(__nv_bfloat16* __restrict__ out, const __nv_fp8_e4m3* __restrict__ q, float scale, size_t n) {
    const __nv_bfloat16 s = __float2bfloat16(scale);
    const size_t vecs = n / 8;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < vecs; v += stride) {
        const uint2 raw = reinterpret_cast<const uint2*>(q)[v];
        const __nv_fp8_e4m3* in = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = __hmul(__float2bfloat16(float(in[j])), s);
        }
        reinterpret_cast<uint4*>(out)[v] = *reinterpret_cast<const uint4*>(o);
    }
    for (size_t i = vecs * 8 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = __hmul(__float2bfloat16(float(q[i])), s);
    }
}

__global__ void quant_kernel(__nv_fp8_e4m3* __restrict__ out, const __nv_bfloat16* __restrict__ x, size_t n) {
    const size_t vecs = n / 8;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < vecs; v += stride) {
        const uint4 raw = reinterpret_cast<const uint4*>(x)[v];
        const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(&raw);
        __align__(8) __nv_fp8_e4m3 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = __nv_fp8_e4m3(fminf(fmaxf(__bfloat162float(in[j]), -448.0f), 448.0f));
        }
        reinterpret_cast<uint2*>(out)[v] = *reinterpret_cast<const uint2*>(o);
    }
    for (size_t i = vecs * 8 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = __nv_fp8_e4m3(fminf(fmaxf(__bfloat162float(x[i]), -448.0f), 448.0f));
    }
}

unsigned grid_for(size_t n) {
    size_t blocks = (n / 8 + 255) / 256;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return (unsigned)blocks;
}

}  // namespace

// 0 = ok, 2 = misaligned pointer, else the CUDA error.
extern "C" int qir_fp8_dequant_bf16(void* out, const void* q, float scale, size_t n, void* stream) {
    if (((uintptr_t)out & 15) || ((uintptr_t)q & 7)) return 2;
    dequant_kernel<<<grid_for(n), 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16*)out, (const __nv_fp8_e4m3*)q, scale, n);
    return (int)cudaGetLastError();
}

extern "C" int qir_bf16_quant_fp8(void* out, const void* x, size_t n, void* stream) {
    if (((uintptr_t)out & 7) || ((uintptr_t)x & 15)) return 2;
    quant_kernel<<<grid_for(n), 256, 0, (cudaStream_t)stream>>>((__nv_fp8_e4m3*)out, (const __nv_bfloat16*)x, n);
    return (int)cudaGetLastError();
}
