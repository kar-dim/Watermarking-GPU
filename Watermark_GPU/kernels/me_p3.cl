__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __local float Rx_local[4096], //64 local threads, 64 values each
    __local float rx_local[512]) //64 local threads, 8 values each
{
    const int x = get_global_id(0), y = get_global_id(1);
    const int width = get_image_width(image), height = get_image_height(image);
    const int local_id = get_local_id(1);
    const int padded_height = get_global_size(1);

    //clear local memory
    for (int i = 0; i < 8; i++)
        rx_local[(local_id * 8) + i] = 0.0f;
    for (int i = 0; i < 64; i++)
        Rx_local[(local_id * 64) + i] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    //fix for OpenCL 1.2 where global size % local size should be 0, and local size is padded, a bound check is needed
    if (y < height) {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        int k = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[k++] = read_imagef(image, sampler, (int2)(j, i)).x;
        const float cur_value = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 64 local Rx and 8 local rx values
        int counter = 0;
        for (int i = 0; i < 8; i++) {
            rx_local[(local_id * 8) + i] = x_[i] * cur_value;
            for (int j = 0; j < 8; j++) {
                Rx_local[(local_id * 64) + counter] = x_[i] * x_[j];
                counter++;
            }
        }

        //first local thread of the group will do the reduction sums into the global memory
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id == 0) {
            const int output_index = (x * padded_height) + y;
            float reduction_sum_Rx = 0.0f;
            float reduction_sum_rx = 0.0f;
            for (int i = 0; i < 64; i++) {
                reduction_sum_Rx = 0.0f;
                reduction_sum_rx = 0.0f;
                for (int j = 0; j < 4096; j += 64)
                    reduction_sum_Rx += Rx_local[i + j];
                for (int j = 0; j < 512; j += 64)
                    reduction_sum_rx += rx_local[i + j];
                Rx[i + output_index] = reduction_sum_Rx;
                rx[i + output_index] = reduction_sum_rx;

            }
        }
    }
}