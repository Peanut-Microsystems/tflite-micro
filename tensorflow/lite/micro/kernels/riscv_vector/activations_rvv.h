/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_ACTIVATIONS_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_ACTIVATIONS_RVV_H_

#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/activations.h"

namespace tflite {

// RISC-V Vector optimized RELU for quantized int8
void ReluQuantizedInt8RVV(const ReluOpData& data,
                          const RuntimeShape& input_shape,
                          const RuntimeShape& output_shape,
                          const int8_t* input_data, int8_t* output_data);

// RISC-V Vector optimized RELU for quantized int16
void ReluQuantizedInt16RVV(const ReluOpData& data,
                           const RuntimeShape& input_shape,
                           const RuntimeShape& output_shape,
                           const int16_t* input_data, int16_t* output_data);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_ACTIVATIONS_RVV_H_
