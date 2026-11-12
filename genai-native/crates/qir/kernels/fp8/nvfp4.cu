// NVFP4 (e2m1 values, e4m3 scales for each 16-element block, one f32 global scale) for
// src/nvfp4.rs, laid out the way ComfyUI's `nvfp4` repacks (comfy_kitchen) store their weights
// and the way cuBLASLt's block-scaled FP4 GEMM reads them:
//   - data: two values a byte, row-major `(rows, cols/2)`, element 2i in the high nibble
//     (Comfy-Org's files; activations are packed alike, so the GEMM pairs the same products);
//   - block scales: e4m3, `(rows, cols/16)` padded to `(128, 4)` tiles and swizzled inside each
//     tile (cuBLAS's "1D block scaling factors layout": row r of a tile sits at
//     `(r % 32) * 16 + (r / 32) * 4`, then the column within the tile);
//   - value = e2m1 * block_scale * global_scale.
// Activations are quantized per call as comfy_kitchen's `quantize_nvfp4` does: the global scale
// is amax / (448 * 6), each block's scale is e4m3(block_amax / 6 / global), and each value is
// e2m1(x / (block_scale * global)) rounded to nearest even, saturating at ±6.
// Only plain CUDA arithmetic: the kernels build for every architecture; the GEMM that uses
// their output is cuBLASLt's and needs a Blackwell card.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float e2m1_value(uint8_t code) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[code & 7];
    return (code & 8) ? -v : v;
}

// Round to the nearest e2m1 value, ties to the even code, saturating at 6.
__device__ __forceinline__ uint8_t e2m1_code(float x) {
    const float a = fabsf(x);
    uint8_t c;
    if (!(a > 0.25f)) c = 0;  // also NaN -> 0
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return (x < 0.0f && c != 0) ? (uint8_t)(c | 8) : c;
}

// Where block-scale (row, col) lives in the swizzled, tile-padded buffer of `col_tiles` tiles a row.
__device__ __forceinline__ size_t swizzled(size_t row, size_t col, size_t col_tiles) {
    const size_t tile = (row / 128) * col_tiles + col / 4;
    const size_t r = row % 128;
    return tile * 512 + (r % 32) * 16 + (r / 32) * 4 + (col % 4);
}

__global__ void amax_kernel(unsigned int* __restrict__ out, const __nv_bfloat16* __restrict__ x, size_t n) {
    float m = 0.0f;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        m = fmaxf(m, fabsf(__bfloat162float(x[i])));
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float warp_max[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_max[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < (int)(blockDim.x >> 5) ? warp_max[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        // Non-negative floats order as their bit patterns.
        if (lane == 0) atomicMax(out, __float_as_uint(m));
    }
}

// One thread a 16-element block. `params[0]` holds the amax bits on entry; the first thread
// writes `params[1]` = alpha (activation global * weight global) and `params[2]` = 0 (beta).
__global__ void quant_kernel(uint8_t* __restrict__ q, uint8_t* __restrict__ scales, float* __restrict__ params,
                             const __nv_bfloat16* __restrict__ x, size_t rows, size_t cols, float weight_global) {
    const float amax = __uint_as_float(reinterpret_cast<const unsigned int*>(params)[0]);
    const float global = amax > 0.0f ? amax / (448.0f * 6.0f) : 1.0f;
    const size_t blocks_per_row = cols / 16;
    const size_t total = rows * blocks_per_row;
    const size_t col_tiles = (blocks_per_row + 3) / 4;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (first == 0) {
        params[1] = global * weight_global;
        params[2] = 0.0f;
    }
    for (size_t b = first; b < total; b += stride) {
        const size_t row = b / blocks_per_row, col = b % blocks_per_row;
        const uint4* src = reinterpret_cast<const uint4*>(x + row * cols + col * 16);
        __align__(16) __nv_bfloat16 v[16];
        reinterpret_cast<uint4*>(v)[0] = src[0];
        reinterpret_cast<uint4*>(v)[1] = src[1];
        float f[16];
        float bmax = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            f[j] = __bfloat162float(v[j]);
            bmax = fmaxf(bmax, fabsf(f[j]));
        }
        const __nv_fp8_e4m3 s8 = __nv_fp8_e4m3(fminf(bmax / 6.0f / global, 448.0f));
        const float s = float(s8);
        const float inv = s > 0.0f ? 1.0f / (s * global) : 0.0f;
        __align__(8) uint8_t packed[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            packed[j] = (uint8_t)((e2m1_code(f[2 * j] * inv) << 4) | e2m1_code(f[2 * j + 1] * inv));
        }
        reinterpret_cast<uint2*>(q)[(row * cols + col * 16) / 16] = *reinterpret_cast<const uint2*>(packed);
        scales[swizzled(row, col, col_tiles)] = *reinterpret_cast<const uint8_t*>(&s8);
    }
}

// Weights back to bf16: one thread a 16-element block.
__global__ void dequant_kernel(__nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ q, const uint8_t* __restrict__ scales,
                               size_t rows, size_t cols, float global, int linear_scales, int low_first) {
    const size_t blocks_per_row = cols / 16;
    const size_t total = rows * blocks_per_row;
    const size_t col_tiles = (blocks_per_row + 3) / 4;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x; b < total; b += stride) {
        const size_t row = b / blocks_per_row, col = b % blocks_per_row;
        const uint8_t sbits = scales[linear_scales ? row * blocks_per_row + col : swizzled(row, col, col_tiles)];
        const float s = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&sbits)) * global;
        const uint2 raw = reinterpret_cast<const uint2*>(q)[(row * cols + col * 16) / 16];
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
        __align__(16) __nv_bfloat16 o[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint8_t lo = bytes[j] & 15, hi = bytes[j] >> 4;
            o[2 * j] = __float2bfloat16(e2m1_value(low_first ? lo : hi) * s);
            o[2 * j + 1] = __float2bfloat16(e2m1_value(low_first ? hi : lo) * s);
        }
        uint4* dst = reinterpret_cast<uint4*>(out + row * cols + col * 16);
        dst[0] = reinterpret_cast<const uint4*>(o)[0];
        dst[1] = reinterpret_cast<const uint4*>(o)[1];
    }
}

unsigned grid_for(size_t work) {
    size_t blocks = (work + 255) / 256;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return (unsigned)blocks;
}

}  // namespace

// Bytes of the swizzled block-scale buffer for a `(rows, cols)` matrix.
extern "C" size_t qir_nvfp4_scale_bytes(size_t rows, size_t cols) {
    const size_t r = (rows + 127) / 128 * 128;
    const size_t c = (cols / 16 + 3) / 4 * 4;
    return r * c;
}

// bf16 `(rows, cols)` -> packed e2m1 `q` (rows * cols / 2 bytes), swizzled e4m3 `scales`
// (qir_nvfp4_scale_bytes, zeroed here so the padding reads as 0) and `params` (3 floats:
// scratch, alpha, beta) for the GEMM. 0 = ok, 2 = misaligned or a width not a multiple of 16.
extern "C" int qir_nvfp4_quant(void* q, void* scales, void* params, const void* x, size_t rows, size_t cols, float weight_global, void* stream) {
    if (((uintptr_t)x & 15) || ((uintptr_t)q & 7) || ((uintptr_t)params & 3) || (cols % 16)) return 2;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t e = cudaMemsetAsync(scales, 0, qir_nvfp4_scale_bytes(rows, cols), s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemsetAsync(params, 0, 3 * sizeof(float), s);
    if (e != cudaSuccess) return (int)e;
    const size_t n = rows * cols;
    const unsigned amax_grid = grid_for(n / 8 + 1) < 1024 ? grid_for(n / 8 + 1) : 1024;
    amax_kernel<<<amax_grid, 256, 0, s>>>((unsigned int*)params, (const __nv_bfloat16*)x, n);
    quant_kernel<<<grid_for(rows * (cols / 16)), 256, 0, s>>>((uint8_t*)q, (uint8_t*)scales, (float*)params, (const __nv_bfloat16*)x, rows, cols,
                                                             weight_global);
    return (int)cudaGetLastError();
}

// Packed e2m1 + e4m3 block scales + global -> bf16 `(rows, cols)`. `linear_scales` reads the
// scales as a plain `(rows, cols/16)` matrix instead of the swizzled tiles, `low_first` takes
// element 2i from the low nibble: both only for checking a file's layout.
extern "C" int qir_nvfp4_dequant(void* out, const void* q, const void* scales, size_t rows, size_t cols, float global, int linear_scales,
                                 int low_first, void* stream) {
    if (((uintptr_t)out & 15) || ((uintptr_t)q & 7) || (cols % 16)) return 2;
    dequant_kernel<<<grid_for(rows * (cols / 16)), 256, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const uint8_t*)q, (const uint8_t*)scales, rows, cols, global, linear_scales, low_first);
    return (int)cudaGetLastError();
}
