#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdio>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "add_rvv.h"

using namespace tflite;


// void AddRVV32(const ArithmeticParams& params,
//               const RuntimeShape& input1_shape, const int32_t* input1_data,
//               const RuntimeShape& input2_shape, const int32_t* input2_data,
//               const RuntimeShape& output_shape, int32_t* output_data) {

//   const int batch = output_shape.Dims(0);
//   const int height = output_shape.Dims(1);
//   const int width = output_shape.Dims(2);
//   const int channels = output_shape.Dims(3);

//   for (int b = 0; b < batch; ++b) {
//     const int b1 = (input1_shape.Dims(0) == 1) ? 0 : b;
//     const int b2 = (input2_shape.Dims(0) == 1) ? 0 : b;

//     for (int h = 0; h < height; ++h) {
//       const int h1 = (input1_shape.Dims(1) == 1) ? 0 : h;
//       const int h2 = (input2_shape.Dims(1) == 1) ? 0 : h;

//       for (int w = 0; w < width; ++w) {
//         const int w1 = (input1_shape.Dims(2) == 1) ? 0 : w;
//         const int w2 = (input2_shape.Dims(2) == 1) ? 0 : w;

//         int i = 0;
//         while (i < channels) {
//           size_t vl = __riscv_vsetvl_e32m4(channels - i);

//           // Compute base offsets
//           int offset1 = ((b1 * input1_shape.Dims(1) + h1) * input1_shape.Dims(2) + w1) * input1_shape.Dims(3) + i;
//           int offset2 = ((b2 * input2_shape.Dims(1) + h2) * input2_shape.Dims(2) + w2) * input2_shape.Dims(3) + i;
//           int offset_out = ((b * height + h) * width + w) * channels + i;

//           // Load vectors
//           vint32m4_t v1 = __riscv_vle32_v_i32m4(input1_data + offset1, vl);
//           vint32m4_t v2 = __riscv_vle32_v_i32m4(input2_data + offset2, vl);

//           // Handle broadcasting in channels dimension
//           if (input1_shape.Dims(3) == 1) {
//             v1 = __riscv_vmv_v_x_i32m4(input1_data[offset1], vl);  // scalar broadcast
//           }
//           if (input2_shape.Dims(3) == 1) {
//             v2 = __riscv_vmv_v_x_i32m4(input2_data[offset2], vl);  // scalar broadcast
//           }

//           // Vector add
//           vint32m4_t vout = __riscv_vadd_vv_i32m4(v1, v2, vl);

//           // Store output
//           __riscv_vse32_v_i32m4(output_data + offset_out, vout, vl);

//           i += vl;
//         }
//       }
//     }
//   }
// }

  // namespace tflite


void AddRVV32(const ArithmeticParams& params,
          const RuntimeShape& input1_shape, const int32_t* input1_data,
          const RuntimeShape& input2_shape, const int32_t* input2_data,
          const RuntimeShape& output_shape, int32_t* output_data){

          const int flat_size = MatchingFlatSize(input1_shape, input2_shape);
          int i =0;
          while(i < flat_size){
            size_t vl = __riscv_vsetvl_e32m4(flat_size - i);
            vint32m4_t v1 = __riscv_vle32_v_i32m4(input1_data + i, vl);
            vint32m4_t v2 = __riscv_vle32_v_i32m4(input2_data + i, vl);
            vint32m4_t vout = __riscv_vadd_vv_i32m4(v1, v2, vl);
      
            __riscv_vse32_v_i32m4(output_data + i, vout, vl);

            i += vl;

          }
        }
  

void AddRVV8(const ArithmeticParams& params,
          const RuntimeShape& input1_shape, const int8_t* input1_data,
          const RuntimeShape& input2_shape, const int8_t* input2_data,
          const RuntimeShape& output_shape, int8_t* output_data){
            //flatten data into 1D array for processing 
            //MatchingElementsSize = total number of elements that are proccessed at once.

            size_t flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
            
            //flat_size used to drive the RVV loop, processing all elements in chunks determined by v1
           
            size_t i = 0;
            while(i < flat_size){ 
                size_t vlen = __riscv_vsetvl_e8m1(flat_size - i); 
                //set vector length dynamically to handle remaining elements
                vint8m1_t v1 = __riscv_vle8_v_i8m1(input1_data + i, vlen);
                vint8m1_t v2 = __riscv_vle8_v_i8m1(input2_data + i, vlen);
                vint8m1_t v_out = __riscv_vadd_vv_i8m1(v1, v2, vlen); //add two vectors
                __riscv_vse8_v_i8m1(output_data + i, v_out, vlen);
                i += vlen; //move to next chunk of elements
            }        
}



