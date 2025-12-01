#ifndef SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H
#define SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H
#include <cstdint>

#include "msb.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

using namespace tflite;

#define kPcanSnrBits 12
#define kPcanOutputBits 6

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

void ApplyPcanAutoGainControlFixed_RVV(const int16_t* gain_lut, int32_t snr_shift,
                                   const uint32_t* noise_estimate,
                                   uint32_t* filterbank_output,
                                   int num_channels);

#endif 