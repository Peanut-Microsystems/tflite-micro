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

#include "tensorflow/lite/micro/kernels/riscv_vector/activations_rvv.h"

#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"

namespace tflite {

// RISC-V Vector optimized RELU for float32
// Note: zve32x only supports integer vectors, so we use scalar for float32
void ReluFloatRVV(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  // Scalar implementation for float32 (zve32x doesn't support float vectors)
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float lower = 0.0f;
    const float clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

// RISC-V Vector optimized RELU for quantized int8
void ReluQuantizedInt8RVV(const ReluOpData& data,
                          const RuntimeShape& input_shape,
                          const RuntimeShape& output_shape,
                          const int8_t* input_data, int8_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const int32_t input_offset = data.params.input_offset;
  const int32_t output_offset = data.params.output_offset;
  const int32_t output_multiplier = data.params.output_multiplier;
  const int output_shift = data.params.output_shift;
  const int32_t output_activation_min = data.params.quantized_activation_min;
  const int32_t output_activation_max = data.params.quantized_activation_max;

  int i = 0;
  while (i < flat_size) {
    size_t vl = __riscv_vsetvl_e32m4(flat_size - i);
    
    // Load int8 input and widen to int32
    vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_data + i, vl);
    vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
    vint32m4_t v_input_s32 = __riscv_vsext_vf2_i32m4(v_input_s16, vl);
    
    // Apply input offset
    v_input_s32 = __riscv_vsub_vx_i32m4(v_input_s32, input_offset, vl);
    
    // Multiply by quantization multiplier
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_input_s32, output_multiplier, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_input_s32, output_multiplier, vl);
    
    // Apply shift (simplified for common case)
    vint32m4_t v_result;
    const int effective_shift = 31 - output_shift;
    
    if (effective_shift > 0 && effective_shift < 32) {
      // Calculate rounding
      int32_t rounding = (1 << (effective_shift - 1));
      vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
      vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, rounding, vl);
      vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding, vl);
      vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, 0, vl);
      v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
      
      // Perform 64-bit shift
      vuint32m4_t v_lo_part = __riscv_vsrl_vx_u32m4(v_sum_lo_u, effective_shift, vl);
      vint32m4_t v_hi_part = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_shift, vl);
      v_result = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4(v_lo_part, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_part), vl));
    } else {
      v_result = v_prod_lo;
    }
    
    // Add output offset
    v_result = __riscv_vadd_vx_i32m4(v_result, output_offset, vl);
    
    // Clamp to activation range
    v_result = __riscv_vmax_vx_i32m4(v_result, output_activation_min, vl);
    v_result = __riscv_vmin_vx_i32m4(v_result, output_activation_max, vl);
    
    // Narrow back to int8
    vint16m2_t v_result_s16 = __riscv_vnclip_wx_i16m2(v_result, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_output_s8 = __riscv_vnclip_wx_i8m1(v_result_s16, 0, __RISCV_VXRM_RNU, vl);
    
    // Store result
    __riscv_vse8_v_i8m1(output_data + i, v_output_s8, vl);
    
    i += vl;
  }
}

// RISC-V Vector optimized RELU for quantized int16
void ReluQuantizedInt16RVV(const ReluOpData& data,
                           const RuntimeShape& input_shape,
                           const RuntimeShape& output_shape,
                           const int16_t* input_data, int16_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const int32_t input_offset = data.params.input_offset;
  const int32_t output_offset = data.params.output_offset;
  const int32_t output_multiplier = data.params.output_multiplier;
  const int output_shift = data.params.output_shift;
  const int32_t output_activation_min = data.params.quantized_activation_min;
  const int32_t output_activation_max = data.params.quantized_activation_max;

  int i = 0;
  while (i < flat_size) {
    size_t vl = __riscv_vsetvl_e32m4(flat_size - i);
    
    // Load int16 input and widen to int32
    vint16m2_t v_input_s16 = __riscv_vle16_v_i16m2(input_data + i, vl);
    vint32m4_t v_input_s32 = __riscv_vsext_vf2_i32m4(v_input_s16, vl);
    
    // Apply input offset
    v_input_s32 = __riscv_vsub_vx_i32m4(v_input_s32, input_offset, vl);
    
    // Multiply by quantization multiplier
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_input_s32, output_multiplier, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_input_s32, output_multiplier, vl);
    
    // Apply shift
    vint32m4_t v_result;
    const int effective_shift = 31 - output_shift;
    
    if (effective_shift > 0 && effective_shift < 32) {
      int32_t rounding = (1 << (effective_shift - 1));
      vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
      vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, rounding, vl);
      vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding, vl);
      vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, 0, vl);
      v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
      
      vuint32m4_t v_lo_part = __riscv_vsrl_vx_u32m4(v_sum_lo_u, effective_shift, vl);
      vint32m4_t v_hi_part = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_shift, vl);
      v_result = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4(v_lo_part, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_part), vl));
    } else {
      v_result = v_prod_lo;
    }
    
    // Add output offset
    v_result = __riscv_vadd_vx_i32m4(v_result, output_offset, vl);
    
    // Clamp to activation range
    v_result = __riscv_vmax_vx_i32m4(v_result, output_activation_min, vl);
    v_result = __riscv_vmin_vx_i32m4(v_result, output_activation_max, vl);
    
    // Narrow back to int16
    vint16m2_t v_output_s16 = __riscv_vnclip_wx_i16m2(v_result, 0, __RISCV_VXRM_RNU, vl);
    
    // Store result
    __riscv_vse16_v_i16m2(output_data + i, v_output_s16, vl);
    
    i += vl;
  }
}

// RISC-V Vector optimized RELU6 for float32
// Note: zve32x only supports integer vectors, so we use scalar for float32
void Relu6FloatRVV(const RuntimeShape& input_shape, const float* input_data,
                   const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  // Scalar implementation for float32 (zve32x doesn't support float vectors)
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6.0f;
    const float lower = 0.0f;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

// RISC-V Vector optimized RELU6 for quantized int8
void Relu6QuantizedInt8RVV(int8_t lower, int8_t upper,
                           const RuntimeShape& input_shape,
                           const int8_t* input_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  int i = 0;
  while (i < flat_size) {
    size_t vl = __riscv_vsetvl_e8m2(flat_size - i);
    
    // Load input vector
    vint8m2_t v_input = __riscv_vle8_v_i8m2(input_data + i, vl);
    
    // Apply clamping: clamp(x, lower, upper)
    vint8m2_t v_output = __riscv_vmax_vx_i8m2(v_input, lower, vl);
    v_output = __riscv_vmin_vx_i8m2(v_output, upper, vl);
    
    // Store result
    __riscv_vse8_v_i8m2(output_data + i, v_output, vl);
    
    i += vl;
  }
}

// RISC-V Vector optimized RELU6 for quantized int16
void Relu6QuantizedInt16RVV(int16_t lower, int16_t upper,
                            const RuntimeShape& input_shape,
                            const int16_t* input_data,
                            const RuntimeShape& output_shape,
                            int16_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  
  int i = 0;
  while (i < flat_size) {
    size_t vl = __riscv_vsetvl_e16m4(flat_size - i);
    
    // Load input vector
    vint16m4_t v_input = __riscv_vle16_v_i16m4(input_data + i, vl);
    
    // Apply clamping: clamp(x, lower, upper)
    vint16m4_t v_output = __riscv_vmax_vx_i16m4(v_input, lower, vl);
    v_output = __riscv_vmin_vx_i16m4(v_output, upper, vl);
    
    // Store result
    __riscv_vse16_v_i16m4(output_data + i, v_output, vl);
    
    i += vl;
  }
}

}  // namespace tflite
