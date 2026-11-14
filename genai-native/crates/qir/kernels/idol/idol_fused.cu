// Idol (TripoSplat) fused kernels.
//
// qir_idol_qkv_prep: from the qkv projection (B*L rows of [q | k | v], each H heads of 64, f16)
// and RePo's per-token, per-head offsets (B*L*H*3, f16), writes q, k and v in bf16 as three
// (B*L, H, 64) planes for the flash kernel:
//   q' = rope(q, angle) / rms(q) * gamma_q,   k' likewise,   v' = v
// with angle = pi * delta[axis(pair)] * freq[pair] (pairs 0..n0 on axis 0, then n1 on axis 1,
// the rest on axis 2). RoPE rotates each pair, which keeps the pair's norm, so the RMS over the
// head is the same before and after the rotation: it is taken from the unrotated values.
// Everything is computed in f32 and rounded once to bf16.
// One warp per (row, head): lane i owns pair i (elements 2i, 2i+1).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

static __global__ void idol_qkv_prep_kernel(__nv_bfloat16* __restrict__ out, const __half* __restrict__ qkv,
                                            const __half* __restrict__ delta, const float* __restrict__ params,
                                            long long rows, int heads, int n0, int n1, float eps) {
    const int D = 64;
    const int lane = threadIdx.x & 31;
    const long long w = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= rows * heads) return;
    const long long r = w / heads;
    const int h = (int)(w - r * heads);
    const long long C = (long long)heads * D;
    const __half* base = qkv + r * 3 * C + (long long)h * D + 2 * lane;
    const float2 q = __half22float2(*reinterpret_cast<const __half2*>(base));
    const float2 k = __half22float2(*reinterpret_cast<const __half2*>(base + C));
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(base + 2 * C));

    float sq = q.x * q.x + q.y * q.y;
    float sk = k.x * k.x + k.y * k.y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
    const float iq = rsqrtf(sq / D + eps);
    const float ik = rsqrtf(sk / D + eps);

    const int axis = lane < n0 ? 0 : (lane < n0 + n1 ? 1 : 2);
    const float d = __half2float(delta[w * 3 + axis]);
    // As the reference: (delta * freq) in f32, times pi rounded to f32.
    const float ang = (d * params[lane]) * 3.14159265358979323846f;
    float s, c;
    sincosf(ang, &s, &c);

    const float* gq = params + 32 + h * D + 2 * lane;
    const float* gk = params + 32 + heads * D + h * D + 2 * lane;
    const long long plane = rows * C;
    const long long o = r * C + (long long)h * D + 2 * lane;
    *reinterpret_cast<__nv_bfloat162*>(out + o) =
        __floats2bfloat162_rn((c * q.x - s * q.y) * iq * gq[0], (s * q.x + c * q.y) * iq * gq[1]);
    *reinterpret_cast<__nv_bfloat162*>(out + plane + o) =
        __floats2bfloat162_rn((c * k.x - s * k.y) * ik * gk[0], (s * k.x + c * k.y) * ik * gk[1]);
    *reinterpret_cast<__nv_bfloat162*>(out + 2 * plane + o) = __floats2bfloat162_rn(v.x, v.y);
}

// params: 32 freqs, then gamma_q (H*64), then gamma_k (H*64), all f32. Returns a CUDA error code.
extern "C" int qir_idol_qkv_prep(void* out, const void* qkv, const void* delta, const void* params, long long rows,
                                 int heads, int n0, int n1, float eps, void* stream) {
    const int warps_per_block = 8;
    const long long warps = rows * heads;
    if (warps <= 0) return 0;
    const unsigned grid = (unsigned)((warps + warps_per_block - 1) / warps_per_block);
    idol_qkv_prep_kernel<<<grid, warps_per_block * 32, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __half*)qkv, (const __half*)delta, (const float*)params, rows, heads, n0, n1, eps);
    return (int)cudaGetLastError();
}
