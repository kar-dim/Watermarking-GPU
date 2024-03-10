__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __local float Rx_local[4096], //64 local threads, 64 values each
    __local float rx_local[512], //64 local threads, 8 values each
    __local float Rx_local_final[64], //final reduction Rx value of each thread
    __local float rx_local_final[64]) //final reduction rx value of each thread
{
    const int x = get_global_id(1), y = get_global_id(0);
    const int width = get_image_width(image), height = get_image_height(image);
    const int local_id = get_local_id(1);
    const int padded_cols = get_global_size(1);
    const int output_index = (y * padded_cols) + x;

    //clear local memory
    Rx_local_final[local_id] = 0.0f;
    rx_local_final[local_id] = 0.0f;
    for (int i = 0; i < 8; i++)
        rx_local[(local_id * 8) + i] = 0.0f;
    for (int i = 0; i < 64; i++)
        Rx_local[(local_id * 64) + i] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    //fix for OpenCL 1.2 where global size % local size should be 0, and local size is padded, a bound check is needed
    if (x < width) {
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
    }

    //first local thread of the group will do the reduction sums into the global memory
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        float reduction_sum_Rx, reduction_sum_rx;
        for (int i = 0; i < 64; i++) {
            reduction_sum_Rx = 0.0f;
            reduction_sum_rx = 0.0f;
            for (int j = 0; j < 4096; j += 64)
                reduction_sum_Rx += Rx_local[i + j];
            for (int j = 0; j < 512; j += 64)
                reduction_sum_rx += rx_local[i + j];
            Rx_local_final[i] = reduction_sum_Rx;
            rx_local_final[i] = reduction_sum_rx;
        }
    }
    //wait for the first thread to finish, then each thread will write from the final reduction sums to global memory in parallel
    barrier(CLK_LOCAL_MEM_FENCE);
    Rx[output_index] = Rx_local_final[local_id];
    rx[output_index] = rx_local_final[local_id];
}