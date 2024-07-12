__attribute__((reqd_work_group_size(1, 64, 1)))
__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __constant int* Rx_mappings,
    __local float Rx_local[64][36], //64 local threads, 36 values each
    __local float rx_local[512]) //64 local threads, 8 values each
{
    const int x = get_global_id(1), y = get_global_id(0);
    const int width = get_image_width(image);
    const int padded_width = get_global_size(1);
    const int local_id = get_local_id(1);
    const int rx_stride = local_id * 8;
    const int Rx_stride = local_id * 64;
    const int output_index = (y * padded_width) + x;
    const bool is_padded = padded_width > width;

    //fix for OpenCL 1.2 where global size % local size should be 0, and local size is padded, a bound check is needed
    if (x < width) {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        int counter = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[counter++] = read_imagef(image, sampler, (int2)(j, i)).x;
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 64 local Rx and 8 local rx values
        counter = 0;
        for (int i = 0; i < 8; i++) {
            rx_local[rx_stride + i] = x_[i] * current_pixel;
            Rx_local[local_id][counter] = x_[i] * x_[i];
            counter++;
            for (int j = i + 1; j < 8; j++) {
                Rx_local[local_id][counter] = x_[i] * x_[j];
                counter++;
            }
        }
    }
    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    barrier(CLK_LOCAL_MEM_FENCE);
    const int limit = (is_padded && padded_width - x <= 64) ? 64 - (padded_width - width) : 64;
    float reduction_sum_Rx = 0.0f, reduction_sum_rx = 0.0f;
    for (int j = 0; j < limit; j++)
        reduction_sum_Rx += Rx_local[j][Rx_mappings[local_id]];
    for (int j = 0; j < (512 * limit) / 64; j += 64)
        reduction_sum_rx += rx_local[local_id + j];
    Rx[output_index] = reduction_sum_Rx;
    rx[output_index] = reduction_sum_rx;
}