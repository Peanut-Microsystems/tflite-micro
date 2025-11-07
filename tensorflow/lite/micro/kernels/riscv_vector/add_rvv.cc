#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "add_rvv.h"

using namespace tflite;

void AddRVV8(const ArithmeticParams& params,
          const RuntimeShape& input1_shape, const int8_t* input1_data,
          const RuntimeShape& input2_shape, const int8_t* input2_data,
          const RuntimeShape& output_shape, int8_t* output_data){
            //flatten data into 1D array for processing 
            //MatchingElementsSize = total number of elements that are proccessed at once.
          int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
            
            //flat_size used to drive the RVV loop, processing all elements in chunks determined by v1

            for(int i = 0; i < flat_size;){
                size_t vlen = __riscv_vsetvl_e8m1(flat_size - i); //set vector length dynamically to handle remaining elements
                vint8m1_t v1 = __riscv_vle8_v_i8m1(input1_data + i, vlen);
                vint8m1_t v2 = __riscv_vle8_v_i8m1(input2_data + i, vlen);
                vint8m1_t v_out = __riscv_vadd_vv_i8m1(v1, v2, vlen); //add two vectors
                __riscv_vse8_v_i8m1(output_data + i, v_out, vlen);
                i += vlen; //move to next chunk of elements
            }

          
}

