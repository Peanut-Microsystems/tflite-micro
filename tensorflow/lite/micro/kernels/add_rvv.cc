/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/add.h"

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "riscv_vector.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace riscv_rvv{
void AddRVV8(const ArithmeticParams& params,
          const RuntimeShape& input1_shape, const int8_t* input1_data,
          const RuntimeShape& input2_shape, const int8_t* input2_data,
          const RuntimeShape& output_shape, int8_t* output_data){
            //flatten data into 1D array for processing 
            //MatchingElementsSize = total number of elements that are proccessed at once.
          int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
            
            //flat_size used to drive the RVV loop, processing all elements in chunks determined by v1

            for(int i = 0; i < flat_size;){
                size_t vlen = vsetvl_e8m1(flat_size - i); //set vector length dynamically to handle remaining elements
                vint8m1_t v1 = vle8_v_i8m1(input1_data + i, vlen);
                vint8m1_t v2 = vle8_v_i8m1(input2_data + i, vlen);
                vint8m1_t v_out = vadd_vv_i8m1(v1, v2, vlen); //add two vectors
                vse8_v_i8m1(output_data + i, vout, v1);
                i += vlen; //move to next chunk of elements
            }

          
          }
}
}

