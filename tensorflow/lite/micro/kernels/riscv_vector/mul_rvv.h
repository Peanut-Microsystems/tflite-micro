#ifndef TFLITE_MICRO_KERNELS_RISCV_VECTOR_MUL_RVV_H_
#define TFLITE_MICRO_KERNELS_RISCV_VECTOR_MUL_RVV_H_

#include <stdint.h>
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace riscv_vector {

void MulInt8Rvv(const ArithmeticParams& params,
                const RuntimeShape& input1_shape,
                const int8_t* input1_data,
                const RuntimeShape& input2_shape,
                const int8_t* input2_data,
                const RuntimeShape& output_shape,
                int8_t* output_data);

}  // namespace riscv_vector
}  // namespace tflite

#endif  // TFLITE_MICRO_KERNELS_RISCV_VECTOR_MUL_RVV_H_
