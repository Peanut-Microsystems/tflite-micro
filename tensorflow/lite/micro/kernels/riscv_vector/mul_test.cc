#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

// -----------------------------------------------------------------------------
// Declarations of the three kernel implementations in mul_impl.c
// (make sure these names match the ones in your C file)
// -----------------------------------------------------------------------------
extern "C" {

void MulScalar(const int8_t* input1, const int8_t* input2, int8_t* output,
               int size, const TfLiteArithmeticParams* params);

void MulAutoVectorized(const int8_t* input1, const int8_t* input2,
                       int8_t* output, int size,
                       const TfLiteArithmeticParams* params);

void MulRvv(const int8_t* input1, const int8_t* input2, int8_t* output,
            int size, const TfLiteArithmeticParams* params);

}  // extern "C"

// -----------------------------------------------------------------------------
// Simple cycle counter using rdcycle
// -----------------------------------------------------------------------------
static inline uint64_t ReadTicks() {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

// -----------------------------------------------------------------------------
// Test data and helpers
// -----------------------------------------------------------------------------
namespace {

constexpr int kSize = 256;

// Global aligned buffers so all implementations share the same memory.
alignas(16) int8_t g_input1[kSize];
alignas(16) int8_t g_input2[kSize];
alignas(16) int8_t g_output_scalar[kSize];
alignas(16) int8_t g_output_auto[kSize];
alignas(16) int8_t g_output_rvv[kSize];

void InitInputs() {
  // Deterministic but “random-ish” patterns.
  for (int i = 0; i < kSize; ++i) {
    g_input1[i] = static_cast<int8_t>((i * 3) % 7 - 3);  // roughly [-3, 3]
    g_input2[i] = static_cast<int8_t>((i * 5) % 9 - 4);  // roughly [-4, 4]
  }
}

void InitParams(TfLiteArithmeticParams* params) {
  // Zero-initialize everything first.
  *params = TfLiteArithmeticParams{};
  // Then set the fields we care about.
  params->input1_offset = 0;
  params->input2_offset = 0;
  params->output_offset = 0;
  params->quantized_activation_min = -128;
  params->quantized_activation_max = 127;
  // Other fields (broadcast_category, etc.) can stay at 0.
}

bool ArraysEqual(const int8_t* a, const int8_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      std::printf("  MISMATCH at index %d: %d vs %d\n", i,
                  static_cast<int>(a[i]), static_cast<int>(b[i]));
      return false;
    }
  }
  return true;
}

// Benchmark helper: run fn() `iterations` times and return average cycles.
using KernelFn = void (*)(const int8_t* input1, const int8_t* input2,
                          int8_t* output, int size,
                          const TfLiteArithmeticParams* params);

uint64_t BenchmarkKernel(const char* name, KernelFn fn,
                         int8_t* output, const TfLiteArithmeticParams* params,
                         int iterations) {
  // Warm-up
  fn(g_input1, g_input2, output, kSize, params);

  uint64_t start = ReadTicks();
  for (int i = 0; i < iterations; ++i) {
    fn(g_input1, g_input2, output, kSize, params);
  }
  uint64_t end = ReadTicks();
  uint64_t total = end - start;
  uint64_t avg = total / static_cast<uint64_t>(iterations);

  std::printf("%s: total ticks = %" PRIu64 ", avg ticks = %" PRIu64 "\n",
              name, total, avg);
  return avg;
}

}  // namespace

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {
  std::printf("Int8 MUL benchmark (N = %d)\n", kSize);

  InitInputs();

  TfLiteArithmeticParams params;
  InitParams(&params);

  // ---------------------------------------------------------------------------
  // Correctness check
  // ---------------------------------------------------------------------------
  // 1) Scalar implementation (reference)
  MulScalar(g_input1, g_input2, g_output_scalar, kSize, &params);

  // 2) Auto-vectorized
  MulAutoVectorized(g_input1, g_input2, g_output_auto, kSize, &params);
  if (!ArraysEqual(g_output_scalar, g_output_auto, kSize)) {
    std::printf("ERROR: Auto-vectorized output does NOT match scalar!\n");
    return 1;
  }

  // 3) RVV implementation
  MulRvv(g_input1, g_input2, g_output_rvv, kSize, &params);
  if (!ArraysEqual(g_output_scalar, g_output_rvv, kSize)) {
    std::printf("ERROR: RVV output does NOT match scalar!\n");
    return 1;
  }

  std::printf("All implementations produce identical results.\n\n");

  // ---------------------------------------------------------------------------
  // Benchmark section
  // ---------------------------------------------------------------------------
  constexpr int kIterations = 1000;

  uint64_t scalar_ticks =
      BenchmarkKernel("Scalar         ", MulScalar,
                      g_output_scalar, &params, kIterations);

  uint64_t auto_ticks =
      BenchmarkKernel("Auto-vectorized", MulAutoVectorized,
                      g_output_auto, &params, kIterations);

  uint64_t rvv_ticks =
      BenchmarkKernel("RVV intrinsics ", MulRvv,
                      g_output_rvv, &params, kIterations);

  std::printf("\nSpeedup (vs scalar):\n");
  std::printf("  Auto-vectorized: %.3fx\n",
              static_cast<double>(scalar_ticks) /
              static_cast<double>(auto_ticks));
  std::printf("  RVV intrinsics : %.3fx\n",
              static_cast<double>(scalar_ticks) /
              static_cast<double>(rvv_ticks));

  return 0;
}
