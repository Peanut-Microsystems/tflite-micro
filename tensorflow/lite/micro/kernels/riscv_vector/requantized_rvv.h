#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_REQUANTIZE_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_REQUANTIZE_RVV_H_

#include <stdint.h>
#include <stddef.h>
#include <algorithm>

#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace riscv_vector {

// Per-tensor requantize: 32-bit accumulator -> 32-bit, with clamp and offset.
inline vint32m4_t RequantizeVectorPerTensorS32(
    vint32m4_t v_acc, const int32_t multiplier, const int effective_right_shift,
    const int32_t output_offset, const int32_t activation_min,
    const int32_t activation_max, const size_t vl) {
  // 64-bit rounding value
  const int64_t rounding_val =
      (effective_right_shift > 0)
          ? (INT64_C(1) << (effective_right_shift - 1))
          : 0;
  const int32_t rounding_lo = static_cast<int32_t>(rounding_val);
  const int32_t rounding_hi = static_cast<int32_t>(rounding_val >> 32);

  // 32x32 -> 64 multiply (hi/lo parts)
  vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_acc, multiplier, vl);
  vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_acc, multiplier, vl);

  // add 64-bit rounding (as two 32-bit ops + carry)
  vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
  vuint32m4_t v_sum_lo_u =
      __riscv_vadd_vx_u32m4(v_prod_lo_u, rounding_lo, vl);
  vbool8_t v_carry =
      __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding_lo, vl);
  vint32m4_t v_rounded_hi =
      __riscv_vadd_vx_i32m4(v_prod_hi, rounding_hi, vl);
  v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
  vint32m4_t v_rounded_lo =
      __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_u);

  // emulate 64-bit arithmetic right shift
  vint32m4_t v_res32;
  if (effective_right_shift == 0) {
    v_res32 = v_rounded_lo;
  } else if (effective_right_shift > 0 && effective_right_shift < 32) {
    vuint32m4_t v_lo_usrl =
        __riscv_vsrl_vx_u32m4(
            __riscv_vreinterpret_v_i32m4_u32m4(v_rounded_lo),
            effective_right_shift, vl);
    vint32m4_t v_hi_sll =
        __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_right_shift, vl);
    v_res32 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vor_vv_u32m4(
            v_lo_usrl,
            __riscv_vreinterpret_v_i32m4_u32m4(v_hi_sll), vl));
  } else {
    const int shift_hi = std::min(31, effective_right_shift - 32);
    v_res32 = __riscv_vsra_vx_i32m4(v_rounded_hi, shift_hi, vl);
  }

  // add offset and clamp
  v_res32 = __riscv_vadd_vx_i32m4(v_res32, output_offset, vl);
  v_res32 = __riscv_vmax_vx_i32m4(v_res32, activation_min, vl);
  v_res32 = __riscv_vmin_vx_i32m4(v_res32, activation_max, vl);

  return v_res32;
}

// Per-channel requantize: 32-bit accumulator -> 32-bit, with clamp.
inline vint32m4_t RequantizeVectorPerChannelS32(
    vint32m4_t v_acc, vint32m4_t v_multiplier, vint32m4_t v_shift,
    const int32_t output_offset, const int32_t activation_min,
    const int32_t activation_max, const size_t vl) {
  // 32x32 -> 64 multiply
  vint32m4_t v_prod_hi = __riscv_vmulh_vv_i32m4(v_acc, v_multiplier, vl);
  vint32m4_t v_prod_lo = __riscv_vmul_vv_i32m4(v_acc, v_multiplier, vl);

  // effective shift = 31 - shift (TFLM convention)
  vint32m4_t v_effective_shift =
      __riscv_vrsub_vx_i32m4(v_shift, 31, vl);

  // masks for lanes: right-shift vs left-shift
  vbool8_t v_mask_right_shift =
      __riscv_vmsgt_vx_i32m4_b8(v_effective_shift, 0, vl);
  vbool8_t v_mask_left_shift =
      __riscv_vmnot_m_b8(v_mask_right_shift, vl);

  // ----- right-shift path -----
  vint32m4_t v_res_right;
  {
    // rounding value: 1 << (shift-1)
    vint32m4_t v_shift_minus_1 =
        __riscv_vsub_vx_i32m4_m(
            v_mask_right_shift, v_effective_shift, 1, vl);
    vuint32m4_t v_shift_minus_1_u =
        __riscv_vreinterpret_v_i32m4_u32m4(v_shift_minus_1);
    vbool8_t v_mask_round_lt_32 =
        __riscv_vmsltu_vx_u32m4_b8_m(
            v_mask_right_shift, v_shift_minus_1_u, 32, vl);
    vbool8_t v_mask_round_ge_32 =
        __riscv_vmandn_mm_b8(
            v_mask_right_shift, v_mask_round_lt_32, vl);

    vuint32m4_t v_one_u = __riscv_vmv_v_x_u32m4(1, vl);
    vuint32m4_t v_zero_u = __riscv_vmv_v_x_u32m4(0, vl);

    vuint32m4_t v_rounding_lo_u =
        __riscv_vmerge_vvm_u32m4(
            v_zero_u,
            __riscv_vsll_vv_u32m4_m(
                v_mask_round_lt_32, v_one_u, v_shift_minus_1_u, vl),
            v_mask_round_lt_32, vl);
    vuint32m4_t v_rounding_hi_u =
        __riscv_vmerge_vvm_u32m4(
            v_zero_u,
            __riscv_vsll_vv_u32m4_m(
                v_mask_round_ge_32, v_one_u,
                __riscv_vsub_vx_u32m4_m(
                    v_mask_round_ge_32, v_shift_minus_1_u, 32, vl),
                vl),
            v_mask_round_ge_32, vl);

    vuint32m4_t v_prod_lo_u =
        __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
    vuint32m4_t v_sum_lo_u =
        __riscv_vadd_vv_u32m4_m(
            v_mask_right_shift, v_prod_lo_u, v_rounding_lo_u, vl);
    vbool8_t v_carry =
        __riscv_vmsltu_vv_u32m4_b8_m(
            v_mask_right_shift, v_sum_lo_u, v_prod_lo_u, vl);

    vint32m4_t v_rounded_hi =
        __riscv_vadd_vv_i32m4_m(
            v_mask_right_shift, v_prod_hi,
            __riscv_vreinterpret_v_u32m4_i32m4(v_rounding_hi_u), vl);
    v_rounded_hi =
        __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);

    // emulate 64-bit >> for two cases: shift<32 and shift>=32
    vbool8_t v_mask_shift_lt_32 =
        __riscv_vmslt_vx_i32m4_b8_m(
            v_mask_right_shift, v_effective_shift, 32, vl);
    vbool8_t v_mask_shift_ge_32 =
        __riscv_vmandn_mm_b8(
            v_mask_right_shift, v_mask_shift_lt_32, vl);

    vuint32m4_t v_shift_u =
        __riscv_vreinterpret_v_i32m4_u32m4(v_effective_shift);

    vuint32m4_t v_lo_part =
        __riscv_vsrl_vv_u32m4_m(
            v_mask_shift_lt_32, v_sum_lo_u, v_shift_u, vl);
    vuint32m4_t v_hi_part =
        __riscv_vsll_vv_u32m4_m(
            v_mask_shift_lt_32,
            __riscv_vreinterpret_v_i32m4_u32m4(v_rounded_hi),
            __riscv_vrsub_vx_u32m4_m(
                v_mask_shift_lt_32, v_shift_u, 32, vl),
            vl);

    vint32m4_t v_res_lt_32 =
        __riscv_vreinterpret_v_u32m4_i32m4(
            __riscv_vor_vv_u32m4_m(
                v_mask_shift_lt_32, v_lo_part, v_hi_part, vl));

    vint32m4_t v_res_ge_32 =
        __riscv_vsra_vv_i32m4_m(
            v_mask_shift_ge_32, v_rounded_hi,
            __riscv_vreinterpret_v_i32m4_u32m4(
                __riscv_vsub_vx_i32m4_m(
                    v_mask_shift_ge_32, v_effective_shift, 32, vl)),
            vl);

    v_res_right = __riscv_vmerge_vvm_i32m4(
        v_res_ge_32, v_res_lt_32, v_mask_shift_lt_32, vl);
  }

  // ----- left-shift path -----
  vint32m4_t v_res_left;
  {
    vint32m4_t v_left_shift_amount =
        __riscv_vneg_v_i32m4_m(
            v_mask_left_shift, v_effective_shift, vl);
    v_res_left =
        __riscv_vsll_vv_i32m4_m(
            v_mask_left_shift, v_prod_lo,
            __riscv_vreinterpret_v_i32m4_u32m4(
                v_left_shift_amount),
            vl);
  }

  // merge both paths
  vint32m4_t v_res32 =
      __riscv_vmerge_vvm_i32m4(
          v_res_left, v_res_right, v_mask_right_shift, vl);

  // add offset and clamp
  v_res32 = __riscv_vadd_vx_i32m4(v_res32, output_offset, vl);
  v_res32 = __riscv_vmax_vx_i32m4(v_res32, activation_min, vl);
  v_res32 = __riscv_vmin_vx_i32m4(v_res32, activation_max, vl);

  return v_res32;
}

}  // namespace riscv_vector
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_REQUANTIZE_RVV_H_
