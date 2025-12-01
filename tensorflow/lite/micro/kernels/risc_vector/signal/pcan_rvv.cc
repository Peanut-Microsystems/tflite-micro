#include <riscv_vector>
#include "pcan_rvv.h"


using namespace tflite;

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut) {
  if (x <= 2) {
    return lut[x];
  }

  const int16_t interval = MostSignificantBit32(x);
  lut += 4 * interval - 6;

  const int16_t frac =
      ((interval < 11) ? (x << (11 - interval)) : (x >> (interval - 11))) &
      0x3FF;

  int32_t result = ((int32_t)lut[2] * frac) >> 5;
  result += (int32_t)((uint32_t)lut[1] << 5);
  result *= frac;
  result = (result + (1 << 14)) >> 15;
  result += lut[0];
  return (int16_t)result;
}

void ApplyPcanAutoGainControlFixed_RVV(const int16_t* gain_lut, int32_t snr_shift,
                                       const uint32_t* noise_estimate,
                                       uint32_t* filterbank_output,
                                       int num_channels) {
  // 1) Scalar precompute of gains[] (keeps WideDynamicFunction unchanged)
  //    We store gains as uint32_t to simplify subsequent vector arithmetic.
  std::vector<uint32_t> gains(num_channels);
  for (int i = 0; i < num_channels; ++i) {
    // WideDynamicFunction returns int16_t (lut result); cast to uint32 for multiply
    gains[i] = static_cast<uint32_t>(WideDynamicFunction(noise_estimate[i], gain_lut));
  }

  // 2) Vectorized multiply (32x32->64) and shift, then PcanShrink (vectorized)
  int i = 0;
  while (i < num_channels) {
    // choose an LMUL variant that fits your register pressure; use e.g. e32m2
    size_t vl = __riscv_vsetvl_e32m2(num_channels - i);

    // load uint32 filterbank_output and gains
    vuint32m2_t v_fb = __riscv_vle32_v_u32m2(&filterbank_output[i], vl);
    vuint32m2_t v_gain = __riscv_vle32_v_u32m2(&gains[i], vl);

    // low 32-bit product (mod 2^32)
    vuint32m2_t prod_lo = __riscv_vmul_vv_u32m2(v_fb, v_gain, vl);

    // high 32-bit product (signed/unsigned mix)
    // Use vmulhsu: high part of signed*unsigned. We need unsigned*unsigned high; if unavailable, reinterpret types.
    // Reinterpret gains as signed 32 and use vmulhsu with fb as unsigned to get the high part:
    vint32m2_t v_gain_i32 = __riscv_vreinterpret_v_u32m2_i32m2(v_gain);
    vint32m2_t v_prod_hi_i = __riscv_vmulhsu_vv_i32m2(v_gain_i32, v_fb, vl);
    vuint32m2_t prod_hi = __riscv_vreinterpret_v_i32m2_u32m2(v_prod_hi_i);

    // Now each lane has (prod_hi << 32) | prod_lo as the 64-bit product.
    // We need to right-shift this 64-bit value by snr_shift to create 'snr'.
    // Implement 64-bit logical right shift using hi/lo combination:
    // if shift >= 32:
    //    result = prod_hi >> (shift - 32)
    // else:
    //    result = (prod_hi << (32 - shift)) | (prod_lo >> shift)

    // build snr vector (32-bit result) with masks
    const int s = snr_shift;
    if (s >= 32) {
      int r = s - 32;
      // logical shift right on 32-bit hi parts
      vuint32m2_t snr = __riscv_vlsrl_vx_u32m2(prod_hi, r, vl); // if vlsrl exists; otherwise use vsrl
      // store into temp vector for PcanShrink
      // NOTE: name or exact intrinsic may vary by toolchain
      // Use snr below
    } else {
      int r = s;
      // high_shift = prod_hi << (32 - r)
      vint32m2_t high_shift_i = __riscv_vsll_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(prod_hi), 32 - r, vl);
      vuint32m2_t high_shift = __riscv_vreinterpret_v_i32m2_u32m2(high_shift_i);
      // low_shift = prod_lo >> r
      vuint32m2_t low_shift = __riscv_vsrl_vx_u32m2(prod_lo, r, vl);
      vuint32m2_t snr = __riscv_vor_vv_u32m2(high_shift, low_shift, vl);
      // use snr below
    }

    // Now apply PcanShrink(snr) vectorized
    // PcanShrink: if snr < (2 << kPcanSnrBits) : return (snr*snr) >> const
    // else: return (snr >> shift_diff) - (1 << kPcanOutputBits)
    // We'll implement both branches and then select via mask.

    // mask = snr < threshold
    vbool16_t mask = __riscv_vmsltu_vx_u32m2_b16(snr, (2u << kPcanSnrBits), vl);

    // branch1: x2 = (snr * snr) >> (2 + 2*kPcanSnrBits - kPcanOutputBits)
    // we need 64-bit product for square -> do same low/high extraction as above for multiplication of snr by snr
    vuint32m2_t sq_lo = __riscv_vmul_vv_u32m2(snr, snr, vl);
    vint32m2_t sq_hi_i = __riscv_vmulhsu_vv_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(snr), snr, vl);
    vuint32m2_t sq_hi = __riscv_vreinterpret_v_i32m2_u32m2(sq_hi_i);

    // Build shifted result depending on shift amount (similar combine technique)
    // Let SHR = (2 + 2*kPcanSnrBits - kPcanOutputBits)
    const int SHR = 2 + 2 * kPcanSnrBits - kPcanOutputBits;
    vuint32m2_t branch1_result;
    if (SHR >= 32) {
      // result = sq_hi >> (SHR - 32)
      branch1_result = __riscv_vsrl_vx_u32m2(sq_hi, SHR - 32, vl);
    } else {
      // combine (sq_hi << (32 - SHR)) | (sq_lo >> SHR)
      vint32m2_t tmp_hi_i = __riscv_vsll_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(sq_hi), 32 - SHR, vl);
      vuint32m2_t tmp_hi = __riscv_vreinterpret_v_i32m2_u32m2(tmp_hi_i);
      vuint32m2_t tmp_lo = __riscv_vsrl_vx_u32m2(sq_lo, SHR, vl);
      branch1_result = __riscv_vor_vv_u32m2(tmp_hi, tmp_lo, vl);
    }

    // branch2: (snr >> (kPcanSnrBits - kPcanOutputBits)) - (1 << kPcanOutputBits)
    const int SHR2 = kPcanSnrBits - kPcanOutputBits;
    vuint32m2_t branch2_shift = __riscv_vsrl_vx_u32m2(snr, SHR2, vl);
    vuint32m2_t branch2_result = __riscv_vsub_vx_u32m2(branch2_shift, (1u << kPcanOutputBits), vl);

    // select final = mask ? branch1_result : branch2_result
    vuint32m2_t final_val = __riscv_vmerge_vvm_u32m2(mask, branch1_result, branch2_result, vl);

    // store back
    __riscv_vse32_v_u32m2(&filterbank_output[i], final_val, vl);

    i += vl;
  }
}
