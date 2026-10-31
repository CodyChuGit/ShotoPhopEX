// Implicit-GEMM 2-D convolution for the Wan VAE (Krea 2, Split, 2511): CUTLASS's Ampere-style
// tensor-op Conv2dFprop on NHWC bf16 activations and KRSC bf16 weights, f32 accumulation, the
// bias added in the epilogue (a stride-0 C operand), bf16 out. Stride 1, square kernel, padding
// k/2; C and K multiples of 8 (16-byte vectors). Runs on sm_80 and later, sm_120 included.
// Replaces candle's im2col + GEMM, whose column buffer for a 3x3 conv at full resolution is
// 9x the activation.

#include <cuda_runtime.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace {

using Element = cutlass::bfloat16_t;
using Acc = float;
using Layout = cutlass::layout::TensorNHWC;

template <class TB, class Warp, int Stages>
using FpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    Element, Layout, Element, Layout, Element, Layout, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, Acc, Acc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, Stages, cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

// Wide channels: 128x128 tiles; narrow outputs (K < 128): 128x64.
using WideConv = cutlass::conv::device::ImplicitGemmConvolution<
    FpropKernel<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, 3>>;
using NarrowConv = cutlass::conv::device::ImplicitGemmConvolution<
    FpropKernel<cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>, 4>>;

template <class Conv>
int run(const void* x, const void* w, const void* bias, void* y, int n, int h, int wd, int c, int k, int r, cudaStream_t s) {
  const int pad = r / 2;
  cutlass::conv::Conv2dProblemSize problem({n, h, wd, c}, {k, r, r, c}, {pad, pad, pad, pad}, {1, 1}, {1, 1}, {n, h, wd, k},
                                           cutlass::conv::Mode::kCrossCorrelation, 1);
  auto* xp = reinterpret_cast<Element*>(const_cast<void*>(x));
  auto* wp = reinterpret_cast<Element*>(const_cast<void*>(w));
  auto* bp = reinterpret_cast<Element*>(const_cast<void*>(bias));
  auto* yp = reinterpret_cast<Element*>(y);
  typename Conv::Arguments args(problem, {xp, Layout::packed({n, h, wd, c})}, {wp, Layout::packed({k, r, r, c})},
                                {bp, Layout(Layout::Stride(0))}, {yp, Layout::packed({n, h, wd, k})}, {Acc(1), Acc(1)});
  Conv conv;
  if (Conv::get_workspace_size(args) != 0) return 3;
  if (conv.can_implement(args) != cutlass::Status::kSuccess) return 4;
  if (conv.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return 5;
  if (conv(s) != cutlass::Status::kSuccess) return 6;
  return (int)cudaGetLastError();
}

}  // namespace

// y (N,H,W,K) = conv(x (N,H,W,C), w (K,R,R,C)) + bias (K), all bf16, stride 1, padding R/2.
// 0 = ok, 1 = unsupported shape (C or K not a multiple of 8, even R), 2 = misaligned pointer,
// 3..6 = CUTLASS refused it, else a CUDA error.
extern "C" int qir_vae_conv2d_nhwc(const void* x, const void* w, const void* bias, void* y, int n, int h, int wd, int c, int k,
                                   int r, void* stream) {
  if (c % 8 != 0 || k % 8 != 0 || r % 2 == 0 || n <= 0 || h <= 0 || wd <= 0) return 1;
  if (((size_t)x | (size_t)w | (size_t)bias | (size_t)y) & 15) return 2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  return k >= 128 ? run<WideConv>(x, w, bias, y, n, h, wd, c, k, r, s) : run<NarrowConv>(x, w, bias, y, n, h, wd, c, k, r, s);
}
