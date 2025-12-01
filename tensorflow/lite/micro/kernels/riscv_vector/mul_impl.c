#include <stddef.h>        // for size_t
#include <stdint.h>        // for int8_t, etc.
#include <riscv_vector.h>  // for vint8m1_t and intrinsics

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
    // Simple int8 vector multiply for benchmarking.
    // Our input ranges are small, so the result fits in int8.
    for (int i = 0; i < size; ) {
        size_t vl = __riscv_vsetvl_e8m1(size - i);

        // load int8 vectors
        vint8m1_t v1 = __riscv_vle8_v_i8m1(&input1[i], vl);
        vint8m1_t v2 = __riscv_vle8_v_i8m1(&input2[i], vl);

        // vector multiply in int8
        vint8m1_t vout = __riscv_vmul_vv_i8m1(v1, v2, vl);

        // store back to memory
        __riscv_vse8_v_i8m1(&output[i], vout, vl);

        i += vl;
    }
}


