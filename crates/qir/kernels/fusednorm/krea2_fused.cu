// Krea 2's small ops fused, for src/model/krea2.rs (feature `fusednorm`). Each replaces a chain
// of candle ops that made a full pass over the activations per op; the math is f32 and the
// result is rounded to bf16 once.
//
//   qir_k2_rms_mod   y = rms(x) · w [· (1 + scale_b) + shift_b]     rows of C, one block a row
//   qir_k2_head_rope y = rope(rms(x) · w)                          rows of 128, one warp a row
//   qir_k2_swiglu    y = silu(g) · u
//   qir_k2_sig_mul   y = a · sigmoid(g)
//   qir_k2_gate_add  y = x + v · gate_b                             gate per batch and column
//
// `_b`: the modulation vectors are per batch item (rows / rows_per_batch), `mod_ld` apart.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One block of 256 threads a row of C (C a multiple of 8, ≤ 8192): sum of squares, then the
// scaled, modulated row.
__global__ void rms_mod_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                               const __nv_bfloat16* __restrict__ scale, const __nv_bfloat16* __restrict__ shift, int C, int rows_per_batch,
                               long mod_ld, float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + static_cast<size_t>(row) * C;
    __nv_bfloat16* yr = y + static_cast<size_t>(row) * C;
    const int vecs = C / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
        const uint4 raw = reinterpret_cast<const uint4*>(xr)[v];
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = __bfloat162float(e[j]);
            ss += f * f;
        }
    }
    __shared__ float part[8];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) part[0] = t;
    }
    __syncthreads();
    const float r = rsqrtf(part[0] / static_cast<float>(C) + eps);
    const size_t m = static_cast<size_t>(row / rows_per_batch) * mod_ld;
    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
        const uint4 raw = reinterpret_cast<const uint4*>(xr)[v];
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = v * 8 + j;
            float f = __bfloat162float(e[j]) * r * w[c];
            if (scale != nullptr) f = f * (1.f + __bfloat162float(scale[m + c])) + __bfloat162float(shift[m + c]);
            o[j] = __float2bfloat16(f);
        }
        reinterpret_cast<uint4*>(yr)[v] = *reinterpret_cast<const uint4*>(o);
    }
}

// One warp a (token, head) row of 128: lane l holds elements 4l..4l+3, the RoPE pairs 2l and
// 2l+1 (interleaved, candle's rope_i). `cos`/`sin` are (S, 64) f32, or null for no rotation.
__global__ void head_rope_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                                 const float* __restrict__ cs, const float* __restrict__ sn, long rows, int heads, int S, float eps) {
    const long row = static_cast<long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint2 raw = reinterpret_cast<const uint2*>(x + row * 128)[lane];
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float f[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        f[j] = __bfloat162float(e[j]);
        ss += f[j] * f[j];
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) f[j] *= r * w[lane * 4 + j];
    if (cs != nullptr) {
        const long s = (row / heads) % S;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float c = cs[s * 64 + lane * 2 + p];
            const float si = sn[s * 64 + lane * 2 + p];
            const float a = f[2 * p], b = f[2 * p + 1];
            f[2 * p] = a * c - b * si;
            f[2 * p + 1] = a * si + b * c;
        }
    }
    __align__(8) __nv_bfloat16 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2bfloat16(f[j]);
    reinterpret_cast<uint2*>(y + row * 128)[lane] = *reinterpret_cast<const uint2*>(o);
}

__global__ void swiglu_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float a = __bfloat162float(g[i]);
        y[i] = __float2bfloat16(a / (1.f + __expf(-a)) * __bfloat162float(u[i]));
    }
}

__global__ void sig_mul_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ g, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        y[i] = __float2bfloat16(__bfloat162float(a[i]) / (1.f + __expf(-__bfloat162float(g[i]))));
    }
}

__global__ void gate_add_kernel(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ v,
                                const __nv_bfloat16* __restrict__ gate, size_t n, int C, long rows_per_batch, long mod_ld) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / C;
        const int c = static_cast<int>(i % C);
        const float gt = __bfloat162float(gate[static_cast<size_t>(row / rows_per_batch) * mod_ld + c]);
        y[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(v[i]) * gt);
    }
}

unsigned grid_for(size_t n) {
    size_t blocks = (n + 255) / 256;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535 * 8) blocks = 65535 * 8;
    return static_cast<unsigned>(blocks > 2147483647 ? 2147483647 : blocks);
}

bool misaligned(const void* p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) != 0; }

}  // namespace

// Return codes: 0 ok, 1 a shape the kernel does not take, 2 a misaligned pointer, else CUDA's.

extern "C" int qir_k2_rms_mod(void* y, const void* x, const float* w, const void* scale, const void* shift, int rows, int C, int rows_per_batch,
                              long mod_ld, float eps, void* stream) {
    if (C % 8 != 0 || C > 8192 || rows <= 0 || rows_per_batch <= 0) return 1;
    if (misaligned(y, 16) || misaligned(x, 16)) return 2;
    rms_mod_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), w,
                                                                       static_cast<const __nv_bfloat16*>(scale), static_cast<const __nv_bfloat16*>(shift), C,
                                                                       rows_per_batch, mod_ld, eps);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int qir_k2_head_rope(void* y, const void* x, const float* w, const float* cs, const float* sn, long rows, int heads, int S, float eps,
                                void* stream) {
    if (rows <= 0 || heads <= 0 || S <= 0) return 1;
    if (misaligned(y, 8) || misaligned(x, 8)) return 2;
    const unsigned blocks = static_cast<unsigned>((rows + 7) / 8);
    head_rope_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), w, cs, sn,
                                                                           rows, heads, S, eps);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int qir_k2_swiglu(void* y, const void* g, const void* u, size_t n, void* stream) {
    swiglu_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(g),
                                                                             static_cast<const __nv_bfloat16*>(u), n);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int qir_k2_sig_mul(void* y, const void* a, const void* g, size_t n, void* stream) {
    sig_mul_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(a),
                                                                              static_cast<const __nv_bfloat16*>(g), n);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int qir_k2_gate_add(void* y, const void* x, const void* v, const void* gate, size_t n, int C, long rows_per_batch, long mod_ld, void* stream) {
    if (C <= 0 || rows_per_batch <= 0) return 1;
    gate_add_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x),
                                                                               static_cast<const __nv_bfloat16*>(v), static_cast<const __nv_bfloat16*>(gate), n, C,
                                                                               rows_per_batch, mod_ld);
    return static_cast<int>(cudaGetLastError());
}
