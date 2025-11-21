#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdio>

#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "add_rvv.h"
#include "requantize_rvv.h"

using namespace tflite;

vint32m4_t VectorMultiplyByQuantizedMultiplierSmallerThanOne(vint32m4_t v_acc, 
    const int32_t multiplier, const int shift, size_t vl){

      //COMPUTE 64 bit product
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_acc, multiplier, vl);
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_acc, multiplier, vl);

    //rounding constant
    const int effective_right_shift = 31 - shift;
    const int64_t rounding_val = 
    (effective_right_shift > 0) ? (INT64_C(1) << (effective_right_shift - 1)) : 0;
    const int32_t rounding_lo = static_cast<int32_t>(rounding_val);
    const int32_t rounding_hi = static_cast<int32_t>((rounding_val >> 32));

    // Add rounding to low part (32-bit add with carry)
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo); //make unsigned
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, rounding_lo, vl); //adds low 32 bits of rounding to low 32 bits of product
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding_lo, vl); //checks if prev addition caused carry. 
    // v_carry is mask of lanes where carry can happen
    vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, rounding_hi, vl); //add 32 bits of rounding to high 32 bits of product
    v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl); //add 1 only where v_carry is true - handle overflow of v_sum_lo_u + rounding_lo
    vint32m4_t v_rounded_lo = __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_u); //convert low part back to signed integers
    
    // Perform 64b arithmetic right shift using 32b vector shifts
    vint32m4_t v_res32;
    //no right shift needed
    if (effective_right_shift == 0) {
      v_res32 = v_rounded_lo;
    } 
    else if (effective_right_shift > 0 && effective_right_shift < 32) {
      vuint32m4_t v_lo_usrl = __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(v_rounded_lo),
                                                    effective_right_shift, vl); //logical right shift of low 32 bits 
      vint32m4_t v_hi_sll = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_right_shift, vl); //shift high 32 bits left to align with bits that go in 32 lower 32 bit result
      v_res32 = __riscv_vreinterpret_v_u32m4_i32m4(
          __riscv_vor_vv_u32m4(v_lo_usrl, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_sll), vl)); //combine lo and hi
    } else { //
      int shift_hi = std::min(31, effective_right_shift - 32); //discard low 32 bits
      v_res32 = __riscv_vsra_vx_i32m4(v_rounded_hi, shift_hi, vl); //right shift to preserve sign 
    }

    return v_res32;
  }

void AddElementwiseRVV8(int size, const ArithmeticParams& params, 
                       const int8_t* input1_data, const int8_t* input2_data,
                       int8_t* output_data){
  int i = 0;
  while (i < size) {
    // Determine vector length for remaining elements (in bytes for e8)
    size_t vl = __riscv_vsetvl_e8m1(size - i);
    // Load chunk as int8 vector

    //Widen inputs
    vint8m1_t v1_i8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
    vint8m1_t v2_i8 = __riscv_vle8_v_i8m1(input2_data + i, vl);

    vint32m4_t v1 = __riscv_vsext_vf4_i32m4(v1_i8, vl);
    vint32m4_t v2 = __riscv_vsext_vf4_i32m4(v2_i8, vl);

    //Dequantize

    //Add input offsets
    v1 = __riscv_vadd_vx_i32m4(v1, params.input1_offset, vl);
    v2 = __riscv_vadd_vx_i32m4(v2, params.input2_offset, vl);

    //Left shit by left shift amount
    if(params.left_shift > 0){
      v1 = __riscv_vsll_vx_i32m4(v1, params.left_shift, vl);
      v2 = __riscv_vsll_vx_i32m4(v2, params.left_shift, vl);
    }

    vint32m4_t v1_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v1, params.input1_multiplier, params.input1_shift, vl);
    vint32m4_t v2_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v2, params.input2_multiplier, params.input2_shift, vl);

    //add scaled inputs
    vint32m4_t v_sum = __riscv_vadd_vv_i32m4(v1_scaled, v2_scaled, vl);

    const int effective_right_shift = 31 - params.output_shift;
    //requantize
    vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
        v_sum, params.output_multiplier, effective_right_shift,
        params.output_offset, params.quantized_activation_min,
        params.quantized_activation_max, vl);

    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_out8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m1(output_data + i, v_out8, vl);

    i += vl;
  }


}


void AddRVV8(const ArithmeticParams& params,
             const RuntimeShape& input1_shape, const int8_t* input1_data,
             const RuntimeShape& input2_shape, const int8_t* input2_data,
             const RuntimeShape& output_shape, int8_t* output_data) {

  const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);
  AddElementwiseRVV8(flat_size, params, input1_data, input2_data, output_data);
  //int i = 0;
  // while (i < flat_size) {
  //   // Determine vector length for remaining elements (in bytes for e8)
  //   size_t vl = __riscv_vsetvl_e8m1(flat_size - i);
  //   // Load chunk as int8 vector

  //   //Widen inputs
  //   vint8m1_t v1_i8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
  //   vint8m1_t v2_i8 = __riscv_vle8_v_i8m1(input2_data + i, vl);

  //   vint16m2_t v1_i16 = __riscv_vsext_vf2_i16m2(v1_i8, vl);
  //   vint16m2_t v2_i16 = __riscv_vsext_vf2_i16m2(v2_i8, vl);

  //   vint32m4_t v1 = __riscv_vsext_vf2_i32m4(v1_i16, vl);
  //   vint32m4_t v2 = __riscv_vsext_vf2_i32m4(v2_i16, vl);

  //   //Dequantize

  //   //Add input offsets
  //   v1 = __riscv_vadd_vx_i32m4(v1, params.input1_offset, vl);
  //   v2 = __riscv_vadd_vx_i32m4(v2, params.input2_offset, vl);

  //   //Left shit by left shift amount
  //   if(params.left_shift > 0){
  //     v1 = __riscv_vsll_vx_i32m4(v1, params.left_shift, vl);
  //     v2 = __riscv_vsll_vx_i32m4(v2, params.left_shift, vl);
  //   }

  //   vint32m4_t v1_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v1, params.input1_multiplier, params.input1_shift, vl);
  //   vint32m4_t v2_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v2, params.input2_multiplier, params.input2_shift, vl);

  //   //add scaled inputs
  //   vint32m4_t v_sum = __riscv_vadd_vv_i32m4(v1_scaled, v2_scaled, vl);

  //   const int effective_right_shift = 31 - params.output_shift;
  //   //requantize
  //   vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
  //       v_sum, params.output_multiplier, effective_right_shift,
  //       params.output_offset, params.quantized_activation_min,
  //       params.quantized_activation_max, vl);

  //   vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
  //   vint8m1_t v_out8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

  //   __riscv_vse8_v_i8m1(output_data + i, v_out8, vl);

  //   i += vl;
  //}
}

static void AddBroadcast1_Contiguous2_RVV8(size_t size, const ArithmeticParams& params, 
                       const int8_t* scalar_data, const int8_t* input2_data,
                       int8_t* output_data){
  
  size_t i = 0;
  int8_t s8 = *scalar_data;
  int32_t s32 = params.input1_offset + s8;
  s32 <<= params.left_shift;
  s32 = MultiplyByQuantizedMultiplierSmallerThanOneExp(s32, params.input1_multiplier, params.input1_shift);

  while (i < size) {
    // Determine vector length for remaining elements (in bytes for e8)
    size_t vl = __riscv_vsetvl_e8m1(size - i);
   
    vint32m4_t v1 = __riscv_vmv_v_x_i32m4(s32, vl);

    //Widen inputs
    vint8m1_t v2_i8 = __riscv_vle8_v_i8m1(input2_data + i, vl);
    vint32m4_t v2 = __riscv_vsext_vf4_i32m4(v2_i8, vl);

    //Dequantize
    //Add input offsets
    v2 = __riscv_vadd_vx_i32m4(v2, params.input2_offset, vl);

    //Left shit by left shift amount
    if(params.left_shift > 0){
      v2 = __riscv_vsll_vx_i32m4(v2, params.left_shift, vl);
    }

    vint32m4_t v2_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v2, params.input2_multiplier, params.input2_shift, vl);

    //add scaled inputs
    vint32m4_t v_sum = __riscv_vadd_vv_i32m4(v1, v2_scaled, vl);

    const int effective_right_shift = 31 - params.output_shift;
    //requantize
    vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
        v_sum, params.output_multiplier, effective_right_shift,
        params.output_offset, params.quantized_activation_min,
        params.quantized_activation_max, vl);

    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_out8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m1(output_data + i, v_out8, vl);

    i += vl;
    }
  }
  //input2 scalar and needs to be broadcasted
  static void AddContiguous1_Broadcast2_RVV8(size_t size, const ArithmeticParams& params, 
                       const int8_t* input1_data, const int8_t* scalar_data,
                       int8_t* output_data){
  
  size_t i = 0;
  int8_t s8 = *scalar_data;
  int32_t s32 = params.input2_offset + static_cast<int32_t>(s8);
  s32 = s32 * (1 << params.left_shift);
  s32 = MultiplyByQuantizedMultiplierSmallerThanOneExp(s32, params.input2_multiplier, params.input2_shift);

  while (i < size) {
    // Determine vector length for remaining elements (in bytes for e8)
    size_t vl = __riscv_vsetvl_e8m1(size - i);
   
    vint32m4_t v2 = __riscv_vmv_v_x_i32m4(s32, vl);

    //Widen inputs
    vint8m1_t v1_i8 = __riscv_vle8_v_i8m1(input1_data + i, vl);
    vint32m4_t v1 = __riscv_vsext_vf4_i32m4(v1_i8, vl);

    //Dequantize
    //Add input offsets
    v1 = __riscv_vadd_vx_i32m4(v1, params.input1_offset, vl);

    //Left shit by left shift amount
    if(params.left_shift > 0){
      v1 = __riscv_vsll_vx_i32m4(v1, params.left_shift, vl);
    }

    vint32m4_t v1_scaled = VectorMultiplyByQuantizedMultiplierSmallerThanOne(v1, params.input1_multiplier, params.input1_shift, vl);

    //add scaled inputs
    vint32m4_t v_sum = __riscv_vadd_vv_i32m4(v1_scaled, v2, vl);

    const int effective_right_shift = 31 - params.output_shift;
    //requantize
    vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
        v_sum, params.output_multiplier, effective_right_shift,
        params.output_offset, params.quantized_activation_min,
        params.quantized_activation_max, vl);

    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t v_out8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m1(output_data + i, v_out8, vl);

    i += vl;
    }
  }



static void BroadcastAddRecursiveRVV8(
    const ArithmeticParams& params,
    int dim,
    size_t* in1_offset, size_t* in2_offset, size_t* out_offset,
    const size_t* stride1, const size_t* stride2,
    const size_t* shape,
    const int8_t* input1, const int8_t* input2,
    int8_t* output) {
  
  if(dim > 0){
    for (size_t i = 0; i < shape[dim]; i++) {

      size_t in1_tmp = *in1_offset;
      size_t in2_tmp = *in2_offset;

      BroadcastAddRecursiveRVV8(params, dim - 1,
                               &in1_tmp, &in2_tmp, out_offset,
                               stride1, stride2, shape,
                               input1, input2, output);
      
      *in1_offset += stride1[dim];
      *in2_offset += stride2[dim];
    }
  } else {
      size_t size = shape[0];
      //dim = 0
      if (stride1[0] == 1 && stride2[0] == 1) {
      AddElementwiseRVV8(static_cast<int>(size), params,
                         input1 + *in1_offset, input2 + *in2_offset,
                         output + *out_offset);
      // Advance local tmp offsets (caller used copies when recursing)
      *out_offset += size;
      *in1_offset += size * stride1[0];  // equals size
      *in2_offset += size * stride2[0];  // equals size
      }
      else if(stride1[0] == 0 && stride2[0] == 1)
      {
        AddBroadcast1_Contiguous2_RVV8(static_cast<int>(size), params,
                                   input1 + *in1_offset,
                                   input2 + *in2_offset,
                                   output + *out_offset);
        *out_offset += size;
        *in1_offset += size * stride1[0];  // equals size
        *in2_offset += size * stride2[0]; 
      }
      else if(stride1[0] == 1 && stride2[0] == 0)
      {
        AddContiguous1_Broadcast2_RVV8(static_cast<int>(size), params,
                                   input1 + *in1_offset,
                                   input2 + *in2_offset,
                                   output + *out_offset);
        *out_offset += size;
        *in1_offset += size * stride1[0];  // equals size
        *in2_offset += size * stride2[0];
      }
     else {
      // General (safe) path: handle broadcasting/non-contiguous strides elementwise.
      // This is correct for stride==0 (broadcast) or stride>1 (non-contiguous).
      for (size_t i = 0; i < size; ++i) {
        // Load scalar inputs according to current local offsets
        const int32_t input1_val = params.input1_offset +
            static_cast<int32_t>(input1[*in1_offset]);
        const int32_t input2_val = params.input2_offset +
            static_cast<int32_t>(input2[*in2_offset]);

        const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
        const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);

        const int32_t scaled_input1_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input1_val, params.input1_multiplier,
                params.input1_shift);
        const int32_t scaled_input2_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input2_val, params.input2_multiplier,
                params.input2_shift);

        const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
        const int32_t raw_output =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                raw_sum, params.output_multiplier, params.output_shift) +
            params.output_offset;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        output[*out_offset] = static_cast<int8_t>(clamped_output);

        // advance the local offsets by the compressed stride for dim 0
        *out_offset += 1;
        *in1_offset += stride1[0];
        *in2_offset += stride2[0];
      }
     }
    }

      //size_t size = shape[0];

      // AddElementwiseRVV8(size, params, input1 + *in1_offset, input2 + *in2_offset, output + *out_offset);

      // *out_offset += static_cast<size_t>(size);
      // *in1_offset += static_cast<size_t>(size) * stride1[dim];
      // *in2_offset += static_cast<size_t>(size) * stride2[dim];   
}

void BroadcastAdd4DRVV8(const ArithmeticParams& params,
                       const RuntimeShape& input1_shape,
                       const int8_t* input1_data,
                       const RuntimeShape& input2_shape,
                       const int8_t* input2_data,
                       const RuntimeShape& output_shape,
                       int8_t* output_data) {

  constexpr int ND = 6;

  size_t stride1[ND], stride2[ND], shape[ND];
  bool ok = ReduceDimensionsForBroadcast<ND>(
      input1_shape, input2_shape,
      stride1, stride2, shape);

  if (!ok) return;

  size_t o1 = 0, o2 = 0, oo = 0;

  BroadcastAddRecursiveRVV8(params,
                           ND - 1,
                           &o1, &o2, &oo,
                           stride1, stride2, shape,
                           input1_data, input2_data,
                           output_data);
}


