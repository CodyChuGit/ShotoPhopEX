// ConvRot weight materialization. Quantizer conventions follow Comfy Kitchen
// (Apache-2.0, Comfy-Org/comfy-kitchen, backends/cuda/ops/int8_linear.cu).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

template<class T> __device__ T butterfly(int digit, T a, T b, T c, T d) {
    if (digit == 3) a = -a;
    if (digit == 2) b = -b;
    if (digit == 1) c = -c;
    if (digit == 0) d = -d;
    return (((a + b) + c) + d) * T(0.5);
}

__global__ void unrotate_weight(__nv_bfloat16* dst, const int8_t* src, const float* scales, int k) {
    __shared__ float values[2][256];
    const int lane = threadIdx.x, group = blockIdx.x, row = group / (k / 256);
    values[0][lane] = float(src[group * 256 + 255 - lane]) * scales[row];
    __syncthreads();
    for (int stage = 0, stride = 1; stage < 4; ++stage, stride *= 4) {
        const int digit = (lane / stride) % 4, first = lane - digit * stride;
        float* v = values[stage % 2];
        values[(stage + 1) % 2][lane] = butterfly(digit, v[first], v[first + stride], v[first + 2 * stride], v[first + 3 * stride]);
        __syncthreads();
    }
    dst[group * 256 + lane] = __float2bfloat16_rn(values[0][lane]);
}

__global__ void rotate_weight(float* dst, const __nv_bfloat16* src, bool round_rot) {
    __shared__ double values[2][256];
    const int lane = threadIdx.x, group = blockIdx.x;
    values[0][lane] = double(__bfloat162float(src[group * 256 + lane]));
    __syncthreads();
    for (int stage = 0, stride = 1; stage < 4; ++stage, stride *= 4) {
        const int digit = (lane / stride) % 4, first = lane - digit * stride;
        double* v = values[stage % 2];
        values[(stage + 1) % 2][lane] = butterfly(digit, v[first], v[first + stride], v[first + 2 * stride], v[first + 3 * stride]);
        __syncthreads();
    }
    dst[group * 256 + lane] = round_rot ? __bfloat162float(__double2bfloat16(values[0][lane])) : float(values[0][lane]);
}

__device__ float random_bf16(uint64_t index, uint64_t seed) {
    uint64_t sum = index + seed;
    uint32_t value = uint32_t(sum) ^ uint32_t(sum >> 32);
    value = value * 747796405u + 2891336453u;
    uint32_t mixed = ((value >> ((value >> 28) + 4)) ^ value) * 277803737u;
    mixed ^= mixed >> 22;
    return __bfloat162float(__float2bfloat16_rn(float(mixed >> 8) * (1.f / 16777216.f)));
}

__global__ void quantize_weight(uint8_t* dst, const float* rotated, float* scales, int k, uint64_t seed) {
    __shared__ float maximum[256];
    const int row = blockIdx.x, lane = threadIdx.x;
    float amax = 0.f;
    for (int col = lane; col < k; col += 256) amax = fmaxf(amax, fabsf(rotated[row * k + col]));
    maximum[lane] = amax;
    __syncthreads();
    for (int step = 128; step > 0; step /= 2) {
        if (lane < step) maximum[lane] = fmaxf(maximum[lane], maximum[lane + step]);
        __syncthreads();
    }
    const float scale = fmaxf(fminf(maximum[0], 3.38953139e38f) * (1.f / 127.f), 1.e-30f);
    if (lane == 0) scales[row] = scale;
    const float divisor = __bfloat162float(__float2bfloat16_rn(scale));
    for (int col = lane; col < k; col += 256) {
        const uint64_t index = uint64_t(row) * k + col;
        float value = __bfloat162float(__float2bfloat16_rn(rotated[index]));
        value = __bfloat162float(__float2bfloat16_rn(value / divisor));
        value = __bfloat162float(__float2bfloat16_rn(value + random_bf16(index, seed)));
        const float q = fminf(127.f, fmaxf(-128.f, floorf(value)));
        dst[row * k + (col / 256) * 256 + 255 - col % 256] = uint8_t(int8_t(q));
    }
}

extern "C" int qir_lora_weight_dequant(void* out, const uint8_t* q, const float* scales, int n, int k, void* stream) {
    unrotate_weight<<<n * (k / 256), 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16*)out, (const int8_t*)q, scales, k);
    return int(cudaGetLastError());
}
extern "C" int qir_lora_weight_rotate(float* out, const void* weight, int n, int k, int round_rot, void* stream) {
    rotate_weight<<<n * (k / 256), 256, 0, (cudaStream_t)stream>>>(out, (const __nv_bfloat16*)weight, round_rot != 0);
    return int(cudaGetLastError());
}
extern "C" int qir_lora_weight_quant(uint8_t* out, const float* rotated, float* scales, int n, int k, uint64_t seed, void* stream) {
    quantize_weight<<<n, 256, 0, (cudaStream_t)stream>>>(out, rotated, scales, k, seed);
    return int(cudaGetLastError());
}
