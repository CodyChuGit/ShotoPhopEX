// Exact attention for src/flash.rs: softmax(Q Kᵀ · scale) V in one pass (the FlashAttention-2
// algorithm), bf16 in and out, fp32 scores, softmax statistics and accumulators; the PV product
// in fp16 (probabilities and V rounded to fp16, 11 bits, where FlashAttention rounds to bf16). Head dim 128,
// non-causal, grouped KV heads (query head h reads KV head h / (H / KVH)).
//
// Layouts: q (B, H, S, 128), k and v (B, KVH, Skv, 128), each with its own batch, head and row
// strides in elements (a dense BHSD tensor, or the BSHD projections seen through a transpose;
// rows are dense and 16-byte aligned); o (B, S, H, 128) dense, the layout the DiT reshapes to
// (B, S, H·128) without a transpose.
//
// One CTA = 4 warps = 64 query rows (16 a warp) of one head; it walks the keys 64 at a time.
// Tensor cores through mma.sync m16n8k16 (bf16 → f32); operands from shared memory with
// ldmatrix (rows padded by 8 elements so the eight rows of a matrix fall in distinct banks).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int D = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int LD = D + 8;
constexpr int THREADS = 128;
constexpr int SMEM_BYTES = (BR + 2 * BC) * LD * 2;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const int n = valid ? 16 : 0;  // 0 bytes read: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(n));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The PV product in fp16 (f32 accumulate): probabilities in [0, 1] keep 11 bits instead of bf16's
// 8, closer to an f32 softmax; V is converted to fp16 in shared memory, clamped to its range.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// The 64-row tile this thread loaded (load_tile's chunks), bf16 -> fp16 in place.
__device__ __forceinline__ void tile_to_f16(__nv_bfloat16* tile, int tid) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = tid + i * THREADS;
        uint4* p = reinterpret_cast<uint4*>(tile + (c >> 4) * LD + (c & 15) * 8);
        uint4 raw = *p;
        __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&raw);
        __align__(16) __half o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = __float2half_rn(fminf(fmaxf(__bfloat162float(e[j]), -65504.f), 65504.f));
        *p = *reinterpret_cast<const uint4*>(o);
    }
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Rows row0.. of a (rows, 128) head slice, rows `rs` elements apart, into a [64][LD] tile; rows
// at or past `limit` read as zeros. 64 rows × 16 chunks of 16 bytes, eight chunks a thread.
__device__ __forceinline__ void load_tile(__nv_bfloat16* tile, const __nv_bfloat16* src, long rs, int row0, int limit, int tid) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = tid + i * THREADS;
        const int r = c >> 4;
        const int ch = c & 15;
        const int row = row0 + r;
        const bool ok = row < limit;
        cp_async16(tile + r * LD + ch * 8, src + static_cast<size_t>(ok ? row : 0) * rs + ch * 8, ok);
    }
}

__global__ void __launch_bounds__(THREADS) flash_fwd(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                     const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o, int S, int Skv,
                                                     int H, int KVH, long q_bs, long q_hs, long q_rs, long k_bs, long k_hs, long k_rs,
                                                     long v_bs, long v_hs, long v_rs, float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BR * LD;
    __nv_bfloat16* sV = sK + BC * LD;

    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int kvh = h / (H / KVH);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = blockIdx.x * BR;

    const __nv_bfloat16* qh = q + b * q_bs + h * q_hs;
    const __nv_bfloat16* kh = k + b * k_bs + kvh * k_hs;
    const __nv_bfloat16* vh = v + b * v_bs + kvh * v_hs;

    load_tile(sQ, qh, q_rs, q0, S, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // This warp's 16 query rows as A fragments, one per 16-wide slice of the head dim.
    uint32_t qf[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        ldsm_x4(qf[kk], sQ + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
    }

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    }
    // Each thread holds two rows (lane/4 and lane/4 + 8 of the warp's 16): their running max
    // (log2 domain, scaled) and this thread's share of the running sum.
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int blocks = (Skv + BC - 1) / BC;
    for (int kb = 0; kb < blocks; ++kb) {
        const int k0 = kb * BC;
        __syncthreads();  // everyone is done with the previous K and V tiles
        load_tile(sK, kh, k_rs, k0, Skv, tid);
        cp_commit();
        load_tile(sV, vh, v_rs, k0, Skv, tid);
        cp_commit();
        cp_wait<1>();  // K is in; V keeps coming while the scores are made
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                uint32_t bk[4];
                ldsm_x4(bk, sK + (j * 8 + (lane & 7)) * LD + kc * 32 + (lane >> 3) * 8);
                mma(s[j], qf[2 * kc], bk[0], bk[1]);
                mma(s[j], qf[2 * kc + 1], bk[2], bk[3]);
            }
        }
        if (k0 + BC > Skv) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (k0 + j * 8 + 2 * (lane & 3) + (e & 1) >= Skv) s[j][e] = -INFINITY;
                }
            }
        }

        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            bm0 = fmaxf(bm0, fmaxf(s[j][0], s[j][1]));
            bm1 = fmaxf(bm1, fmaxf(s[j][2], s[j][3]));
        }
        bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 1));
        bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 2));
        const float n0 = fmaxf(m0, bm0 * scale_log2);
        const float n1 = fmaxf(m1, bm1 * scale_log2);
        const float c0 = exp2f(m0 - n0);
        const float c1 = exp2f(m1 - n1);
        m0 = n0;
        m1 = n1;
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = exp2f(s[j][0] * scale_log2 - n0);
            s[j][1] = exp2f(s[j][1] * scale_log2 - n0);
            s[j][2] = exp2f(s[j][2] * scale_log2 - n1);
            s[j][3] = exp2f(s[j][3] * scale_log2 - n1);
            sum0 += s[j][0] + s[j][1];
            sum1 += s[j][2] + s[j][3];
        }
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j][0] *= c0;
            acc[j][1] *= c0;
            acc[j][2] *= c1;
            acc[j][3] *= c1;
        }

        cp_wait<0>();
        tile_to_f16(sV, tid);
        __syncthreads();
        // P (16 × 64, bf16) · V (64 × 128): the score accumulators of two neighbouring 8-key
        // tiles are exactly the A fragment of one 16-key slice.
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const uint32_t a[4] = {pack_f16(s[2 * ks][0], s[2 * ks][1]), pack_f16(s[2 * ks][2], s[2 * ks][3]), pack_f16(s[2 * ks + 1][0], s[2 * ks + 1][1]),
                                   pack_f16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
            for (int dj = 0; dj < 16; dj += 2) {
                uint32_t bv[4];
                ldsm_x4_trans(bv, sV + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + dj * 8 + (lane >> 4) * 8);
                mma_f16(acc[dj], a, bv[0], bv[1]);
                mma_f16(acc[dj + 1], a, bv[2], bv[3]);
            }
        }
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f;
    const float i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int r0 = q0 + warp * 16 + (lane >> 2);
    const int r1 = r0 + 8;
#pragma unroll
    for (int dj = 0; dj < 16; ++dj) {
        const int d = dj * 8 + 2 * (lane & 3);
        if (r0 < S) *reinterpret_cast<uint32_t*>(o + (static_cast<size_t>(b * S + r0) * H + h) * D + d) = pack(acc[dj][0] * i0, acc[dj][1] * i0);
        if (r1 < S) *reinterpret_cast<uint32_t*>(o + (static_cast<size_t>(b * S + r1) * H + h) * D + d) = pack(acc[dj][2] * i1, acc[dj][3] * i1);
    }
}

}  // namespace

// 0 = ok, 1 = a shape the kernel does not take, 2 = a misaligned pointer, else the CUDA error.
extern "C" int qir_flash_fwd(void* o, const void* q, const void* k, const void* v, int B, int H, int KVH, int S, int Skv, int head_dim,
                             const long* strides, float scale, void* stream) {
    if (head_dim != D || B <= 0 || S <= 0 || Skv <= 0 || KVH <= 0 || H % KVH != 0) return 1;
    if ((reinterpret_cast<uintptr_t>(o) | reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) | reinterpret_cast<uintptr_t>(v)) & 15) return 2;
    for (int i = 0; i < 9; ++i) {
        if (strides[i] % 8 != 0) return 2;  // every row must start on 16 bytes
    }
    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(flash_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        if (e != cudaSuccess) return static_cast<int>(e);
        configured = true;
    }
    const dim3 grid((S + BR - 1) / BR, H, B);
    flash_fwd<<<grid, THREADS, SMEM_BYTES, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v),
        static_cast<__nv_bfloat16*>(o), S, Skv, H, KVH, strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
        strides[8], scale * 1.4426950408889634f);
    return static_cast<int>(cudaGetLastError());
}
