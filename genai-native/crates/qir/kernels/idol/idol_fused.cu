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

// ---- AdaLN blocks: LayerNorm + modulation, gated residuals -----------------------------------
//
// Rows of C channels (C a multiple of 8, C/8 <= 1024 threads); row r belongs to batch r / L.
// The modulation m is (B, 6C) f16: chunk j of batch b is m[b*6C + j*C .. + C]. LayerNorm has no
// affine weights (eps given), as the modulated blocks. The arithmetic is f32, but every value the
// reference (ComfyUI's f16 ops, and the plain candle path) rounds to f16 is rounded here too: the
// normalised row, 1 + scale, the product, the gated update. Rounding once instead drifts the
// 50-step sample measurably away from ComfyUI's.

static __device__ __forceinline__ float r16(float x) {
    return __half2float(__float2half_rn(x));
}

struct Half8 {
    __half2 v[4];
};

static __device__ __forceinline__ void load8(const __half* p, float* f) {
    const Half8 h = *reinterpret_cast<const Half8*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h.v[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

static __device__ __forceinline__ void store8(__half* p, const float* f) {
    Half8 h;
#pragma unroll
    for (int i = 0; i < 4; ++i) h.v[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    *reinterpret_cast<Half8*>(p) = h;
}

// Sum over the block (blockDim.x threads, a multiple of 32).
static __device__ __forceinline__ float block_sum(float v, float* shared) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < warps; ++w) t += shared[w];
    return t;
}

// y = LN(v) * (1 + scale) + shift for this thread's 8 values; mod points at the batch's m row.
static __device__ __forceinline__ void ln_modulate(float* v, const __half* mod, int C, int sh, int sc, float eps, float* shared, float* y) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += v[i];
    const float mean = block_sum(s, shared) / C;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float d = v[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q, shared) / C + eps);
    float shift[8], scale[8];
    const int c0 = threadIdx.x * 8;
    load8(mod + (long long)sh * C + c0, shift);
    load8(mod + (long long)sc * C + c0, scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = r16(r16((v[i] - mean) * inv) * r16(1.f + scale[i])) + shift[i];
}

// h = LN(x) * (1 + m[sc]) + m[sh]
static __global__ void idol_ln_mod_kernel(__half* __restrict__ h, const __half* __restrict__ x, const __half* __restrict__ m,
                                          int L, int C, int sh, int sc, float eps) {
    __shared__ float shared[32];
    const long long r = blockIdx.x;
    const long long off = r * C + threadIdx.x * 8;
    float v[8], y[8];
    load8(x + off, v);
    ln_modulate(v, m + (r / L) * 6LL * C, C, sh, sc, eps, shared, y);
    store8(h + off, y);
}

// x2 = x + a * m[g];  h = LN(x2) * (1 + m[sc]) + m[sh]  (out: x2 plane, then h plane)
static __global__ void idol_res_ln_mod_kernel(__half* __restrict__ out, const __half* __restrict__ x, const __half* __restrict__ a,
                                              const __half* __restrict__ m, long long rows, int L, int C, int g, int sh, int sc, float eps) {
    __shared__ float shared[32];
    const long long r = blockIdx.x;
    const long long off = r * C + threadIdx.x * 8;
    const __half* mod = m + (r / L) * 6LL * C;
    float xv[8], av[8], gv[8], y[8];
    load8(x + off, xv);
    load8(a + off, av);
    load8(mod + (long long)g * C + threadIdx.x * 8, gv);
#pragma unroll
    for (int i = 0; i < 8; ++i) xv[i] += r16(av[i] * gv[i]);
    // The residual stream is kept in f16 as the reference does: normalise the rounded values.
    store8(out + off, xv);
    load8(out + off, xv);
    ln_modulate(xv, mod, C, sh, sc, eps, shared, y);
    store8(out + rows * C + off, y);
}

// x2 = x + f * m[g]
static __global__ void idol_gate_res_kernel(__half* __restrict__ out, const __half* __restrict__ x, const __half* __restrict__ f,
                                            const __half* __restrict__ m, int L, int C, int g) {
    const long long r = blockIdx.x;
    const long long off = r * C + threadIdx.x * 8;
    float xv[8], fv[8], gv[8];
    load8(x + off, xv);
    load8(f + off, fv);
    load8(m + (r / L) * 6LL * C + (long long)g * C + threadIdx.x * 8, gv);
#pragma unroll
    for (int i = 0; i < 8; ++i) xv[i] += r16(fv[i] * gv[i]);
    store8(out + off, xv);
}

extern "C" int qir_idol_ln_mod(void* h, const void* x, const void* m, long long rows, int L, int C, int sh, int sc, float eps, void* stream) {
    if (rows <= 0) return 0;
    idol_ln_mod_kernel<<<(unsigned)rows, C / 8, 0, (cudaStream_t)stream>>>((__half*)h, (const __half*)x, (const __half*)m, L, C, sh, sc, eps);
    return (int)cudaGetLastError();
}

extern "C" int qir_idol_res_ln_mod(void* out, const void* x, const void* a, const void* m, long long rows, int L, int C, int g, int sh, int sc,
                                   float eps, void* stream) {
    if (rows <= 0) return 0;
    idol_res_ln_mod_kernel<<<(unsigned)rows, C / 8, 0, (cudaStream_t)stream>>>((__half*)out, (const __half*)x, (const __half*)a,
                                                                                (const __half*)m, rows, L, C, g, sh, sc, eps);
    return (int)cudaGetLastError();
}

extern "C" int qir_idol_gate_res(void* out, const void* x, const void* f, const void* m, long long rows, int L, int C, int g, void* stream) {
    if (rows <= 0) return 0;
    idol_gate_res_kernel<<<(unsigned)rows, C / 8, 0, (cudaStream_t)stream>>>((__half*)out, (const __half*)x, (const __half*)f, (const __half*)m, L, C, g);
    return (int)cudaGetLastError();
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
