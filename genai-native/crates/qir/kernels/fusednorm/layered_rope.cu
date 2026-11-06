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

// The head RMSNorm follows candle's 32-lane reduction and preserves the bf16
// normalization boundary before the learned weight. Write BHSD directly to
// avoid a transpose copy and a second pointwise pass over every Q/K tensor.
__global__ void layered_head_norm_kernel(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                         unsigned rows, unsigned seq, unsigned heads) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x & 31;
    float value[4];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        value[j] = __bfloat162float(x[row * 128 + lane + j * 32]);
        sum = fmaf(value[j], value[j], sum);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const float scale = rsqrtf(sum / 128.f + 1e-6f);
    const unsigned head = row % heads;
    const unsigned token = (row / heads) % seq;
    const unsigned batch = row / (heads * seq);
    const unsigned out_row = (batch * heads + head) * seq + token;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned col = lane + j * 32;
        const float normalized = __bfloat162float(__float2bfloat16_rn(scale * value[j]));
        y[out_row * 128 + col] = __float2bfloat16_rn(normalized * __bfloat162float(w[col]));
    }
}

extern "C" int qir_layered_head_norm(void* y, const void* x, const void* w, unsigned rows, unsigned seq, unsigned heads, void* stream) {
    layered_head_norm_kernel<<<(rows + 3) / 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w), rows, seq, heads);
    return static_cast<int>(cudaGetLastError());
}

// Keep candle's 1024-thread LayerNorm reduction and bf16 normalized value,
// then apply the bf16 (1 + scale) and single-round addcmul without temporaries.
template <bool Split = false>
__global__ void layered_modulate_kernel(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* params,
                                         unsigned seq, unsigned width, unsigned param_width, unsigned param_batches,
                                         unsigned shift_index, unsigned scale_index, unsigned split = 0) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    float sum = 0.f, square = 0.f;
    for (unsigned col = tid; col < width; col += 1024) {
        const float value = __bfloat162float(x[row * width + col]);
        sum += value;
        square = fmaf(value, value, square);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        square += __shfl_xor_sync(0xffffffffu, square, offset);
    }
    __shared__ float sums[32], squares[32];
    const unsigned lane = tid & 31;
    if (lane == 0) { sums[tid / 32] = sum; squares[tid / 32] = square; }
    __syncthreads();
    sum = sums[lane]; square = squares[lane];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        square += __shfl_xor_sync(0xffffffffu, square, offset);
    }
    const float mean = sum / width;
    const float variance = square / width - mean * mean;
    const float inv = rsqrtf(variance + 1e-6f);
    const unsigned batch = Split ? row / seq + ((row % seq >= split) ? param_batches / 2 : 0) : (param_batches == 1 ? 0 : row / seq);
    const __nv_bfloat16* p = params + batch * param_width;
    for (unsigned col = tid; col < width; col += 1024) {
        const float normalized = __bfloat162float(__float2bfloat16_rn((__bfloat162float(x[row * width + col]) - mean) * inv));
        const float scale = __bfloat162float(__float2bfloat16_rn(__bfloat162float(p[scale_index * width + col]) + 1.f));
        const float shift = __bfloat162float(p[shift_index * width + col]);
        out[row * width + col] = __float2bfloat16_rn(fmaf(normalized, scale, shift));
    }
}

extern "C" int qir_layered_modulate(void* y, const void* x, const void* params, unsigned rows, unsigned seq, unsigned width,
                                    unsigned param_width, unsigned param_batches, unsigned shift_index, unsigned scale_index, void* stream) {
    layered_modulate_kernel<<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(params),
        seq, width, param_width, param_batches, shift_index, scale_index);
    return static_cast<int>(cudaGetLastError());
}

// Only newly allocated GEMM outputs use this in-place epilogue. Preserve the
// bf16 GEMM and bias boundaries, and candle's bf16 arithmetic for optional GELU.
__global__ void layered_bias_kernel(__nv_bfloat16* x, const __nv_bfloat16* bias, size_t chunks, unsigned width, bool gelu) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= chunks) return;
    uint4 raw = *reinterpret_cast<const uint4*>(x + i * 8);
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(&raw);
    const unsigned col = (i * 8) % width;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat16 z = __float2bfloat16_rn(__bfloat162float(v[j]) + __bfloat162float(bias[col + j]));
        if (gelu) {
            const __nv_bfloat16 square = z * z;
            const __nv_bfloat16 cube = square * z;
            const __nv_bfloat16 alpha = z + __float2bfloat16_rn(0.044715f) * cube;
            const __nv_bfloat16 phase = __float2bfloat16_rn(0.7978845608028654f) * alpha;
            const __nv_bfloat16 tangent = __float2bfloat16_rn(tanhf(__bfloat162float(phase)));
            z = (__float2bfloat16_rn(0.5f) * z) * (__float2bfloat16_rn(1.f) + tangent);
        }
        v[j] = z;
    }
    *reinterpret_cast<uint4*>(x + i * 8) = raw;
}

extern "C" int qir_layered_bias(void* x, const void* bias, size_t elements, unsigned width, bool gelu, void* stream) {
    if (width % 8 != 0 || elements % 8 != 0) return 1;
    layered_bias_kernel<<<(elements / 8 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(bias), elements / 8, width, gelu);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int qir_edit2511_modulate(void* y, const void* x, const void* params, unsigned rows, unsigned seq, unsigned width,
                                    unsigned param_width, unsigned param_batches, unsigned shift_index, unsigned split, void* stream) {
    layered_modulate_kernel<true><<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(params),
        seq, width, param_width, param_batches, shift_index, shift_index + 1, split);
    return static_cast<int>(cudaGetLastError());
}

// Edit's residual rounds the product to bf16 before the addition.
__global__ void edit2511_residual_kernel(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* v,
                                       const __nv_bfloat16* p, size_t elements, unsigned seq, unsigned width,
                                       unsigned param_width, unsigned batches, unsigned gate, unsigned split) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const unsigned row = i / width;
    const unsigned batch = row / seq + ((row % seq >= split) ? batches : 0);
    const float g = bf(p[batch * param_width + gate * width + i % width]);
    y[i] = rb(__fadd_rn(bf(x[i]), bf(rb(__fmul_rn(bf(v[i]), g)))));
}

extern "C" int qir_edit2511_residual(void* y, const void* x, const void* v, const void* p, size_t elements,
                                    unsigned seq, unsigned width, unsigned param_width, unsigned batches,
                                    unsigned gate, unsigned split, void* stream) {
    edit2511_residual_kernel<<<(elements + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(v),
        static_cast<const __nv_bfloat16*>(p), elements, seq, width, param_width, batches, gate, split);
    return static_cast<int>(cudaGetLastError());
}
