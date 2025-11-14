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
#include "requantize_rvv.h"


namespace tflite {

void AddRVV8(const ArithmeticParams& params,
             const RuntimeShape& input1_shape, const int8_t* input1_data,
             const RuntimeShape& input2_shape, const int8_t* input2_data,
             const RuntimeShape& output_shape, int8_t* output_data) {

  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  int i = 0;
  while (i < flat_size) {
    // Determine vector length for remaining elements (in bytes for e8)
    size_t vl = __riscv_vsetvl_e8m1(flat_size - i);
    // Load chunk as int8 vector

    //Widen and dequantize inputs
    vint8m1_t v1_i8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
    vint8m1_t v2_i8 = __riscv_vle8_v_i8m1(input2_data + i, vl);

    vint16m2_t v1_i16 = __riscv_vsext_vf2_i16m2(v1_i8, vl);
    vint16m2_t v2_i16 = __riscv_vsext_vf2_i16m2(v2_i8, vl);

    vint32m4_t v1_i32 = __riscv_vsext_vf2_i32m4(v1_i16, vl);
    vint32m4_t v2_i32 = __riscv_vsext_vf2_i32m4(v2_i16, vl);


    //Apply per-input scaling
    v1_i32 = __riscv_vadd_vx_i32m4(v1_i32, params.input1_offset, vl);
    v2_i32 = __riscv_vadd_vx_i32m4(v2_i32, params.input2_offset, vl);

    if(params.left_shift > 0)
    {
      v1_i32 = __riscv_vsll_vx_i32m4(v1_i32, params.left_shift, vl);
      v2_i32 = __riscv_vsll_vx_i32m4(v2_i32, params.left_shift, vl);
    }

    // Add vectors
    vint32m4_t v_sum = __riscv_vadd_vv_i32m4(v1_i32, v2_i32, vl);

// Requantize the sum to the output scale using your new function
    vint32m4_t v_res = RequantizeVectorPerTensorS32(
      v_sum,
      params.output_multiplier,
      params.output_shift,
      params.output_offset,
      params.quantized_activation_min,
      params.quantized_activation_max,
      vl);

    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_out8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);
    

    __riscv_vse8_v_i8m1(output_data + i, v_out8, vl);

    i += vl;
    }
  

}  // namespace riscv_rvv
}


