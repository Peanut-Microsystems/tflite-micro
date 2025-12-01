#include <cstdio>
#include <cinttypes>
#include <cstdint>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

extern "C" {
// Declare the functions implemented in mul_impl.c
void MulScalar(const int8_t* input1, const int8_t* input2, int8_t* output,
               int size, const TfLiteArithmeticParams& params);

void MulAutoVectorized(const int8_t* input1, const int8_t* input2,
                       int8_t* output, int size,
                       const TfLiteArithmeticParams& params);

void MulInt8Rvv(const int8_t* input1, const int8_t* input2, int8_t* output,
                int size, const TfLiteArithmeticParams& params);
}

// -------------------------------------------------------
// rdcycle() to measure performance
// -------------------------------------------------------
static inline uint64_t ReadTicks() {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

// -------------------------------------------------------
// Test parameters
// -------------------------------------------------------
namespace tflite {
namespace testing {

constexpr int kSize = 256;

static int8_t input1[kSize];
static int8_t input2[kSize];
static int8_t output[kSize];
static int8_t golden[kSize];

// Fill test data
void FillTestData() {
  for (int i = 0; i < kSize; i++) {
    input1[i] = (i * 3) % 11 - 5;
    input2[i] = (i * 7) % 13 - 6;
    golden[i] = input1[i] * input2[i];
  }
}

}  // namespace testing
}  // namespace tflite

// -------------------------------------------------------
// MAIN TEST ENTRY
// -------------------------------------------------------
int main() {
  using namespace tflite::testing;

  FillTestData();

  TfLiteArithmeticParams params{};
  params.input1_offset = 0;
  params.input2_offset = 0;
  params.output_offset = 0;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  // ---------------------------------------------------
  // Scalar (reference) version
  // ---------------------------------------------------
  uint64_t start_scalar = ReadTicks();
  MulScalar(input1, input2, output, kSize, params);
  uint64_t end_scalar = ReadTicks();

  bool scalar_ok = true;
  for (int i = 0; i < kSize; i++) {
    if (output[i] != golden[i]) {
      scalar_ok = false;
      break;
    }
  }

  printf("Scalar result: %s\n", scalar_ok ? "PASS" : "FAIL");
  printf("Scalar ticks: %" PRIu64 "\n", end_scalar - start_scalar);

  // ---------------------------------------------------
  // Auto-Vectorized version
  // ---------------------------------------------------
  uint64_t start_auto = ReadTicks();
  MulAutoVectorized(input1, input2, output, kSize, params);
  uint64_t end_auto = ReadTicks();

  bool auto_ok = true;
  for (int i = 0; i < kSize; i++) {
    if (output[i] != golden[i]) {
      auto_ok = false;
      break;
    }
  }

  printf("Auto-vectorized result: %s\n", auto_ok ? "PASS" : "FAIL");
  printf("Auto-vectorized ticks: %" PRIu64 "\n", end_auto - start_auto);

  // ---------------------------------------------------
  // RISCV Vector Intrinsics RVV version
  // ---------------------------------------------------
  uint64_t start_rvv = ReadTicks();
  MulInt8Rvv(input1, input2, output, kSize, params);
  uint64_t end_rvv = ReadTicks();

  bool rvv_ok = true;
  for (int i = 0; i < kSize; i++) {
    if (output[i] != golden[i]) {
      rvv_ok = false;
      break;
    }
  }

  printf("RVV result: %s\n", rvv_ok ? "PASS" : "FAIL");
  printf("RVV ticks: %" PRIu64 "\n", end_rvv - start_rvv);

  return 0;
}
