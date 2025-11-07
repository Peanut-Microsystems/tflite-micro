
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_ADD_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_ADD_RVV_H_

#include <cstdint>
#include <cstddef>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"


using namespace tflite;

void AddRVV8(const ArithmeticParams& params,
          const RuntimeShape& input1_shape, const int8_t* input1_data,
          const RuntimeShape& input2_shape, const int8_t* input2_data,
          const RuntimeShape& output_shape, int8_t* output_data);

#endif

