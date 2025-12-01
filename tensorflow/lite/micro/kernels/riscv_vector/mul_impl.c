#include <stddef.h>        // for size_t
#include <stdint.h>        // for int8_t, int16_t, etc.
#include <riscv_vector.h>  // for RVV intrinsics

void MulScalar(
    int8_t* input1, int8_t* input2, int8_t* output, int size)
{
    for (int i = 0; i < size; i++) {
        int32_t x = input1[i];
        int32_t y = input2[i];
        int32_t r = x * y;       // simple scalar multiply
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        output[i] = (int8_t)r;
    }
}

void MulRVV(
    int8_t* input1, int8_t* input2, int8_t* output, int size)
{
    size_t n = (size_t)size;

    for (size_t i = 0; i < n; ) {
        // Set vector length for 8-bit elements
        size_t vl = __riscv_vsetvl_e8m1(n - i);

        // Load 8-bit vectors
        vint8m1_t v1 = __riscv_vle8_v_i8m1(&input1[i], vl);
        vint8m1_t v2 = __riscv_vle8_v_i8m1(&input2[i], vl);

        // Widening multiply: int8 x int8 -> int16
        vint16m2_t vprod = __riscv_vwmul_vv_i16m2(v1, v2, vl);

        // Store 16-bit products into a temporary buffer
        int16_t tmp[vl];  // VLA sized by current vl
        __riscv_vse16_v_i16m2(tmp, vprod, vl);

        // Scalar saturation back to int8 range
        for (size_t j = 0; j < vl; ++j) {
            int32_t r = tmp[j];
            if (r > 127)  r = 127;
            if (r < -128) r = -128;
            output[i + j] = (int8_t)r;
        }

        i += vl;
    }
}

