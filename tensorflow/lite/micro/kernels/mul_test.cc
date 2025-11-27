/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"

#if defined(TFLM_USE_RISCV_VECTOR_MUL)
#include "tensorflow/lite/micro/kernels/riscv_vector/mul_rvv.h"
#endif


static inline uint64_t ReadTicks() {
  return 0;  // TEMP: no rdcycle, just a dummy
}


namespace tflite {
namespace testing {
namespace {

const int flat_size_simple = 4;
const float scale_simple = 0.01;
int dims_simple[] = {4, 1, 2, 2, 1};
const float input1_simple[] = {-0.8, 0.2, 0.9, 0.7};
const float input2_simple[] = {0.6, 0.4, 0.9, 0.8};
const float golden_simple[] = {-0.48, 0.08, 0.81, 0.56};
const float golden_simple_relu[] = {0.0, 0.08, 0.81, 0.56};

const int flat_size_broadcast = 6;
const float input_scale_broadcast = 0.05f;
const float output_scale_broadcast = 0.01f;
int dims_broadcast[] = {4, 1, 3, 1, 2};
int dims_scalar_broadcast[] = {1, 1};
const float input1_broadcast[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
const float input2_broadcast[] = {0.1};
const float golden_broadcast[] = {-0.2, 0.02, 0.07, 0.08, 0.11, 0.2};
const float golden_broadcast_relu[] = {0, 0.02, 0.07, 0.08, 0.11, 0.2};

template <typename T>
void ValidateMulGoldens(TfLiteTensor* tensors, int tensors_size,
                        TfLiteFusedActivation activation, const T* golden,
                        int output_len, float tolerance, T* output) {
  TfLiteMulParams builtin_data = {
      .activation = activation,
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_MUL();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_len; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestMulFloat(int* input1_dims_data, const float* input1_data,
                  int* input2_dims_data, const float* input2_data,
                  int* output_dims_data, const float* golden,
                  float* output_data, TfLiteFusedActivation activation) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateMulGoldens(tensors, tensors_size, activation, golden,
                     output_dims_count, 1e-5, output_data);
}

template <typename T>
void TestMulQuantized(int* input1_dims_data, const float* input1_data,
                      T* input1_quantized, int* input2_dims_data,
                      const float* input2_data, T* input2_quantized,
                      const float input_scale, const int input_zero_point,
                      int* output_dims_data, const float* golden,
                      T* golden_quantized, const float output_scale,
                      const int output_zero_point, T* output_data,
                      TfLiteFusedActivation activation) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);

  ValidateMulGoldens(tensors, tensors_size, activation, golden_quantized,
                     output_dims_count, 1.0f, output_data);
}

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleFloatNoActivationShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      tflite::testing::dims_simple, tflite::testing::input2_simple,
      tflite::testing::dims_simple, tflite::testing::golden_simple, output_data,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleFloatReluShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      tflite::testing::dims_simple, tflite::testing::input2_simple,
      tflite::testing::dims_simple, tflite::testing::golden_simple_relu,
      output_data, kTfLiteActRelu);
}

TF_LITE_MICRO_TEST(SimpleInt8NoActivationShouldMatchGolden) {
  int8_t input1_quantized[tflite::testing::flat_size_simple];
  int8_t input2_quantized[tflite::testing::flat_size_simple];
  int8_t golden_quantized[tflite::testing::flat_size_simple];
  int8_t output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      input1_quantized, tflite::testing::dims_simple,
      tflite::testing::input2_simple, input2_quantized,
      tflite::testing::scale_simple, 0, tflite::testing::dims_simple,
      tflite::testing::golden_simple, golden_quantized,
      tflite::testing::scale_simple, 0, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleInt16NoActivationShouldMatchGolden) {
  int16_t input1_quantized[tflite::testing::flat_size_simple];
  int16_t input2_quantized[tflite::testing::flat_size_simple];
  int16_t golden_quantized[tflite::testing::flat_size_simple];
  int16_t output_data[tflite::testing::flat_size_simple];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      input1_quantized, tflite::testing::dims_simple,
      tflite::testing::input2_simple, input2_quantized,
      tflite::testing::scale_simple, 0, tflite::testing::dims_simple,
      tflite::testing::golden_simple, golden_quantized,
      tflite::testing::scale_simple, 0, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastFloatNoActivationShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      tflite::testing::dims_scalar_broadcast, tflite::testing::input2_broadcast,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastFloatReluShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulFloat(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      tflite::testing::dims_scalar_broadcast, tflite::testing::input2_broadcast,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast_relu,
      output_data, kTfLiteActRelu);
}

TF_LITE_MICRO_TEST(BroadcastInt8NoActivationShouldMatchGolden) {
  int8_t input1_quantized[tflite::testing::flat_size_broadcast];
  int8_t input2_quantized[tflite::testing::flat_size_broadcast];
  int8_t golden_quantized[tflite::testing::flat_size_broadcast];
  int8_t output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      input1_quantized, tflite::testing::dims_scalar_broadcast,
      tflite::testing::input2_broadcast, input2_quantized,
      tflite::testing::input_scale_broadcast, 0,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      golden_quantized, tflite::testing::output_scale_broadcast, 0, output_data,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(BroadcastInt16NoActivationShouldMatchGolden) {
  int16_t input1_quantized[tflite::testing::flat_size_broadcast];
  int16_t input2_quantized[tflite::testing::flat_size_broadcast];
  int16_t golden_quantized[tflite::testing::flat_size_broadcast];
  int16_t output_data[tflite::testing::flat_size_broadcast];

  tflite::testing::TestMulQuantized(
      tflite::testing::dims_broadcast, tflite::testing::input1_broadcast,
      input1_quantized, tflite::testing::dims_scalar_broadcast,
      tflite::testing::input2_broadcast, input2_quantized,
      tflite::testing::input_scale_broadcast, 0,
      tflite::testing::dims_broadcast, tflite::testing::golden_broadcast,
      golden_quantized, tflite::testing::output_scale_broadcast, 0, output_data,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleInt32NoActivationShouldMatchGolden) {
  int32_t input1_quantized[tflite::testing::flat_size_simple];
  int32_t input2_quantized[tflite::testing::flat_size_simple];
  int32_t golden_quantized[tflite::testing::flat_size_simple];
  int32_t output_data[tflite::testing::flat_size_simple];

  // Int32 mul ignores quantization parameters with TFLite and TFLM. Use
  // TestMulQuantized method to convert float arrays to int32 arrays, but use
  // quantization parameters of 0.01 for both inputs and 0.0001 for output,
  // since input scales are multiplied together to get output scale when there
  // is no rescaling inside the op.
  tflite::testing::TestMulQuantized(
      tflite::testing::dims_simple, tflite::testing::input1_simple,
      input1_quantized, tflite::testing::dims_simple,
      tflite::testing::input2_simple, input2_quantized, 0.01, 0,
      tflite::testing::dims_simple, tflite::testing::golden_simple,
      golden_quantized, 0.0001, 0, output_data, kTfLiteActNone);
}

// -------------------------------------------------------------
// Benchmark for int8 Mul (direct call to mul_common.cc dispatcher)
// -------------------------------------------------------------
TF_LITE_MICRO_TEST(BenchmarkInt8MulDirect) {
  // Size of the test vectors
  constexpr int kSize = 16;
  int8_t input1[kSize];
  int8_t input2[kSize];
  int8_t output[kSize];

  // Fill inputs with deterministic data
  for (int i = 0; i < kSize; i++) {
    input1[i] = (i * 3) % 7 - 3;
    input2[i] = (i * 5) % 9 - 4;
  }

  // Set up quantized Mul params
  tflite::ArithmeticParams params;
  params.input1_offset = 0;
  params.input2_offset = 0;
  params.output_offset = 0;
  // 0.5 in Q31 fixed point
  params.output_multiplier = 1073741824;
  params.output_shift = -1;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  // 1D shape with kSize elements
  tflite::RuntimeShape shape({1, kSize});
  tflite::RuntimeShape out_shape({1, kSize});

  constexpr int kIters = 20000;

  uint64_t start = ReadTicks();
  for (int i = 0; i < kIters; ++i) {
#if defined(TFLM_USE_RISCV_VECTOR_MUL)
    // Your RVV-optimized path
    tflite::riscv_vector::MulInt8Rvv(params, shape, input1, shape, input2,
                                     out_shape, output);
#else
    // Reference scalar implementation
    tflite::reference_integer_ops::Mul(params, shape, input1, shape, input2,
                                       out_shape, output);
#endif
  }
  uint64_t end = ReadTicks();
  uint64_t cycles = end - start;

  printf("BenchmarkInt8MulDirect: %llu cycles, %f cycles/iter\n",
         (unsigned long long)cycles,
         (double)cycles / kIters);
}


TF_LITE_MICRO_TESTS_END
