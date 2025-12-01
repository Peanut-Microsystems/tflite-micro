#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

using namespace tflite;

void AvgPool8bitRVV(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape,
                        int8_t* output_data)
{
    //pooling params
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;
    const int filter_height = params.filter_height;
    const int filter_width = params.filter_width;
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;
    const int8_t output_activation_min = params.quantized_activation_min;
    const int8_t output_activation_max = params.quantized_activation_max;

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    //tensor strides
    const int input_y_stride = input_width * depth;
    const int input_b_stride = input_height * input_y_stride;
    const int output_y_stride = output_width * depth;
    const int output_b_stride = output_height * output_y_stride;
    for (int batch = 0; batch < batches; ++batch) {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;

        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            for (int out_x = 0; out_x < output_width; ++out_x)
            {
                // Vectorized loop over channels (depth)
                size_t current_channel = 0;
                while (current_channel < static_cast<size_t>(depth))
                {
                    // Set vector length. For `zvl128b`, VLEN=128. With SEW=8 (int8_t),
                    // VLMAX is 16 * LMUL. Using LMUL=4 provides a good balance, allowing                    // up to 64 channels to be processed per iteration.
                    size_t vl = __riscv_vsetvl_e8m4(depth - current_channel);

                    //Use 32-bit accumulator for sums
                    vint32m4_t v_sum_s32 = __riscv_vmv_v_x_i32m4(0, vl);
                    int num_valid_elements = 0;

                    for (int f_y = 0; f_y < filter_height; ++f_y) 
                    {
                            const int in_y = out_y * stride_height + f_y - pad_height;
                            if (in_y < 0 || in_y >= input_height) continue;

                            for (int f_x = 0; f_x < filter_width; ++f_x) 
                            {
                                const int in_x = out_x * stride_width + f_x - pad_width;
                                if (in_x < 0 || in_x >= input_width) continue;

                                const int8_t* input_ptr =
                                    input_batch_base + in_y * input_y_stride + in_x * depth + current_channel;

                                // Load int8 vector and widen to int32
                                vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_ptr, vl);
                                vint32m4_t v_input_s32 = __riscv_vsext_vf4_i32m4(v_input_s8, vl);

                                // Accumulate
                                v_sum_s32 = __riscv_vadd_vv_i32m4(v_sum_s32, v_input_s32, vl);
                                num_valid_elements++;
                            }
                    }
                    
                    //divide by number of valid elements
                    if (num_valid_elements > 0) 
                    {
                        //vectorized division
                        vint32m4_t v_count = __riscv_vmv_v_x_i32m4(num_valid_elements, vl);
                        v_sum_s32 = __riscv_vdiv_vv_i32m4(v_sum_s32, v_count, vl);
                    }
                    
                    //narrow and clamp output
                    vint16m2_t v_sum_s16 = __riscv_vnclip_wx_i16m2(v_sum_s32, 0, __RISCV_VXRM_RNU, vl);
                    vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_sum_s16, 0, __RISCV_VXRM_RNU, vl);

                    v_out_s8 = __riscv_vmax_vx_i8m1(v_out_s8, output_activation_min, vl);
                    v_out_s8 = __riscv_vmin_vx_i8m1(v_out_s8, output_activation_max, vl);
                    
                    //Store final vector of average values
                    int8_t* output_ptr = output_batch_base + out_y * output_y_stride + out_x * depth + current_channel;
                    __riscv_vse8_v_i8m1(output_ptr, v_out_s8, vl);
                    
                    //Advance to next block of the channel
                    current_channel += vl;

                }
            }
        }
    }
}
void AvgPool16bitRVV(const PoolParams& params,
                     const RuntimeShape& input_shape,
                     const int16_t* input_data,
                     const RuntimeShape& output_shape,
                     int16_t* output_data) {
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int filter_height = params.filter_height;
  const int filter_width = params.filter_width;
  const int pad_height = params.padding_values.height;
  const int pad_width = params.padding_values.width;

  // Activation bounds must be promoted to int16 for vector compares.
  const int16_t output_activation_min = static_cast<int16_t>(params.quantized_activation_min);
  const int16_t output_activation_max = static_cast<int16_t>(params.quantized_activation_max);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int input_y_stride = input_width * depth;
  const int input_b_stride = input_height * input_y_stride;
  const int output_y_stride = output_width * depth;
  const int output_b_stride = output_height * output_y_stride;


  for (int batch = 0; batch < batches; ++batch) {
    const int16_t* input_batch_base = input_data + batch * input_b_stride;
    int16_t* output_batch_base = output_data + batch * output_b_stride;

    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        size_t current_channel = 0;
        while (current_channel < static_cast<size_t>(depth)) {
          // Choose SEW=16, LMUL=1 for input load. This yields a vint16m1_t.
          size_t vl = __riscv_vsetvl_e16m1(depth - current_channel);

          // Accumulator must be a widened type: i16m1 -> i32m2 via vsext_vf2 -> use vint32m2_t
          vint32m2_t v_sum_s32 = __riscv_vmv_v_x_i32m2(0, vl);

          int num_valid_elements = 0;

          for (int f_y = 0; f_y < filter_height; ++f_y) {
            const int in_y = out_y * stride_height + f_y - pad_height;
            if (in_y < 0 || in_y >= input_height) continue;

            for (int f_x = 0; f_x < filter_width; ++f_x) {
              const int in_x = out_x * stride_width + f_x - pad_width;
              if (in_x < 0 || in_x >= input_width) continue;

              const int16_t* input_ptr =
                  input_batch_base + in_y * input_y_stride + in_x * depth + current_channel;

              // load int16 vector (i16m1)
              vint16m1_t v_input_s16 = __riscv_vle16_v_i16m1(input_ptr, vl);

              // widen to i32m2 (vf2)
              vint32m2_t v_input_s32 = __riscv_vsext_vf2_i32m2(v_input_s16, vl);

              // accumulate
              v_sum_s32 = __riscv_vadd_vv_i32m2(v_sum_s32, v_input_s32, vl);
              ++num_valid_elements;
            }
          }

          // If no valid elements (shouldn't happen normally) keep zeros.
          if (num_valid_elements > 0) {

            vint32m2_t v_count = __riscv_vmv_v_x_i32m2(num_valid_elements, vl);
            v_sum_s32 = __riscv_vdiv_vv_i32m2(v_sum_s32, v_count, vl);

          }

          // Narrow 32->16 with saturation (vnclip with shift 0)
          vint16m1_t v_out_s16 = __riscv_vnclip_wx_i16m1(v_sum_s32, 0, __RISCV_VXRM_RNU, vl);

          // Apply activation/clamp (use i16 vector ops)
          v_out_s16 = __riscv_vmax_vx_i16m1(v_out_s16, output_activation_min, vl);
          v_out_s16 = __riscv_vmin_vx_i16m1(v_out_s16, output_activation_max, vl);

          // store results
          int16_t* output_ptr = output_batch_base + out_y * output_y_stride + out_x * depth + current_channel;
          __riscv_vse16_v_i16m1(output_ptr, v_out_s16, vl);

          current_channel += vl;
        }
      }
    }
  }
}

void MaxPool8BitRVV(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data)
{
    // Extract pooling parameters
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;
    const int filter_height = params.filter_height;
    const int filter_width = params.filter_width;
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;
    const int8_t output_activation_min = params.quantized_activation_min;
    const int8_t output_activation_max = params.quantized_activation_max;

    // Extract shape dimensions
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    // Calculate tensor strides for direct pointer arithmetic
    const int input_y_stride = input_width * depth;
    const int input_b_stride = input_height * input_y_stride;
    const int output_y_stride = output_width * depth;
    const int output_b_stride = output_height * output_y_stride;

    // Loop over batches
    for (int batch = 0; batch < batches; ++batch) {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;

        // Loop over output spatial dimensions (y, x)
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            for (int out_x = 0; out_x < output_width; ++out_x)
            {
                // Vectorized loop over channels (depth)
                size_t current_channel = 0;
                while (current_channel < static_cast<size_t>(depth))
                {
                    // Set vector length. For `zvl128b`, VLEN=128. With SEW=8 (int8_t),
                    // VLMAX is 16 * LMUL. Using LMUL=4 provides a good balance, allowing
                    // up to 64 channels to be processed per iteration.
                    size_t vl = __riscv_vsetvl_e8m4(depth - current_channel);

                    // Initialize the accumulator vector with the smallest possible int8_t value.
                    vint8m4_t v_max_s8 = __riscv_vmv_v_x_i8m4(std::numeric_limits<int8_t>::lowest(), vl);

                    // Loop over the filter window dimensions (y, x)
                    for (int f_y = 0; f_y < filter_height; ++f_y)
                    {
                        for (int f_x = 0; f_x < filter_width; ++f_x)
                        {
                            // Calculate corresponding input coordinates for this filter tap
                            const int in_y = (out_y * stride_height) + f_y - pad_height;
                            const int in_x = (out_x * stride_width) + f_x - pad_width;

                            // Handle padding by checking if the input coordinates are valid
                            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
                            {
                                // If valid, calculate the pointer to the input vector
                                const int8_t* input_ptr = input_batch_base +
                                                        (in_y * input_y_stride) +
                                                        (in_x * depth) +
                                                        current_channel;

                                // Load a vector of input values (unit-stride access)
                                vint8m4_t v_input_s8 = __riscv_vle8_v_i8m4(input_ptr, vl);
                                
                                // Perform the vector max operation
                                v_max_s8 = __riscv_vmax_vv_i8m4(v_max_s8, v_input_s8, vl);
                            }
                        }
                    }

                    // After iterating through the filter window, apply activation clamping
                    v_max_s8 = __riscv_vmax_vx_i8m4(v_max_s8, output_activation_min, vl);
                    v_max_s8 = __riscv_vmin_vx_i8m4(v_max_s8, output_activation_max, vl);
                    
                    // Calculate the output pointer
                    int8_t* output_ptr = output_batch_base +
                                        (out_y * output_y_stride) +
                                        (out_x * depth) +
                                        current_channel;
                    
                    // Store the final vector of maximum values (unit-stride access)
                    __riscv_vse8_v_i8m4(output_ptr, v_max_s8, vl);

                    // Advance to the next block of channels
                    current_channel += vl;
                }
            }
        }
    }
}

void MaxPool16BitRVV(const PoolParams& params, const RuntimeShape& input_shape,
                    const int16_t* input_data, const RuntimeShape& output_shape,
                    int16_t* output_data)
{
    // Extract pooling parameters
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;
    const int filter_height = params.filter_height;
    const int filter_width = params.filter_width;
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;
    const int16_t output_activation_min = params.quantized_activation_min;
    const int16_t output_activation_max = params.quantized_activation_max;

    // Extract shape dimensions
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    // Calculate tensor strides for direct pointer arithmetic
    const int input_y_stride = input_width * depth;
    const int input_b_stride = input_height * input_y_stride;
    const int output_y_stride = output_width * depth;
    const int output_b_stride = output_height * output_y_stride;

    // Loop over batches
    for (int batch = 0; batch < batches; ++batch)
    {
        const int16_t* input_batch_base = input_data + batch * input_b_stride;
        int16_t* output_batch_base = output_data + batch * output_b_stride;

        // Loop over output spatial dimensions (y, x)
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            for (int out_x = 0; out_x < output_width; ++out_x)
            {
                // Vectorized loop over channels (depth)
                size_t current_channel = 0;
                while (current_channel < static_cast<size_t>(depth))
                {
                    // Set vector length. SEW is now 16 bits. With VLEN=128, VLMAX is 8 * LMUL.
                    // LMUL=4 still provides a good balance, processing up to 32 channels.
                    size_t vl = __riscv_vsetvl_e16m4(depth - current_channel);

                    // Initialize the accumulator vector with the smallest possible int16_t value.
                    vint16m4_t v_max_s16 = __riscv_vmv_v_x_i16m4(std::numeric_limits<int16_t>::lowest(), vl);

                    // Loop over the filter window dimensions (y, x)
                    for (int f_y = 0; f_y < filter_height; ++f_y)
                    {
                        for (int f_x = 0; f_x < filter_width; ++f_x)
                        {
                            // Calculate corresponding input coordinates for this filter tap
                            const int in_y = (out_y * stride_height) + f_y - pad_height;
                            const int in_x = (out_x * stride_width) + f_x - pad_width;

                            // Handle padding by checking if the input coordinates are valid
                            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width)
                            {
                                // If valid, calculate the pointer to the input vector
                                const int16_t* input_ptr = input_batch_base +
                                                        (in_y * input_y_stride) +
                                                        (in_x * depth) +
                                                        current_channel;

                                // Load a vector of input values (unit-stride access)
                                vint16m4_t v_input_s16 = __riscv_vle16_v_i16m4(input_ptr, vl);
                                
                                // Perform the vector max operation
                                v_max_s16 = __riscv_vmax_vv_i16m4(v_max_s16, v_input_s16, vl);
                            }
                        }
                    }

                    // After iterating through the filter window, apply activation clamping
                    v_max_s16 = __riscv_vmax_vx_i16m4(v_max_s16, output_activation_min, vl);
                    v_max_s16 = __riscv_vmin_vx_i16m4(v_max_s16, output_activation_max, vl);
                    
                    // Calculate the output pointer
                    int16_t* output_ptr = output_batch_base +
                                            (out_y * output_y_stride) +
                                            (out_x * depth) +
                                            current_channel;
                    
                    // Store the final vector of maximum values (unit-stride access)
                    __riscv_vse16_v_i16m4(output_ptr, v_max_s16, vl);

                    // Advance to the next block of channels
                    current_channel += vl;
                }
            }
        }
    }
}