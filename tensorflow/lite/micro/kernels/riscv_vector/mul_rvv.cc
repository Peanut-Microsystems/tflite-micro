#include <riscv_vector.h>
#include "tensorflow/lite/micro/kernels/riscv_vector/mul_rvv.h"
#include "tensorflow/lite/micro/kernels/riscv_vector/requantize_rvv.h"

namespace tflite {
namespace riscv_vector {

void MulInt8Rvv(const ArithmeticParams& params,
                const RuntimeShape& input1_shape,
                const int8_t* input1_data,
                const RuntimeShape& input2_shape,
                const int8_t* input2_data,
                const RuntimeShape& output_shape,
                int8_t* output_data) {
  // Non-broadcast case: all shapes match
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  // Convert TFLM's output_shift into an effective right shift
  const int effective_right_shift = 31 - params.output_shift;

  int i = 0;
  // For RV32 with Zve32x_zvl128b, max vl for e32 is 4, for e8 is 16.
  // 16 int32 values is enough for one vector.
  int32_t tmp[16];

  while (i < flat_size) {
    size_t vl = __riscv_vsetvl_e8m1(flat_size - i);

    // 1. Load int8 inputs
    vint8m1_t v_in1 = __riscv_vle8_v_i8m1(input1_data + i, vl);
    vint8m1_t v_in2 = __riscv_vle8_v_i8m1(input2_data + i, vl);

    // 2. Widen to 16-bit and add zero-point offsets
    vint16m2_t v_in1_16 =
        __riscv_vwadd_vx_i16m2(v_in1, params.input1_offset, vl);
    vint16m2_t v_in2_16 =
        __riscv_vwadd_vx_i16m2(v_in2, params.input2_offset, vl);

    // 3. Multiply: int16 × int16 -> int32
    vint32m4_t v_mul = __riscv_vwmul_vv_i32m4(v_in1_16, v_in2_16, vl);

    // 4. Requantize, add output offset, clamp to activation range
    //    The helper in requantize_rvv.h does all of that.
    vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
        v_mul,
        params.output_multiplier,
        effective_right_shift,
        params.output_offset,
        params.quantized_activation_min,
        params.quantized_activation_max,
        vl);

    // 5. Narrow to int8_t.
    //    To avoid dealing with vnclip_* intrinsic policies, we store the
    //    32-bit vector to a small temporary array and cast to int8 scalars.
    __riscv_vse32_v_i32m4(tmp, v_res32, vl);
    for (size_t j = 0; j < vl; ++j) {
      // Values are already clamped to [activation_min, activation_max],
      // so this cast is safe for int8 range.
      output_data[i + j] = static_cast<int8_t>(tmp[j]);
    }

    i += vl;
  }
}

}  // namespace riscv_vector
}  // namespace tflite
