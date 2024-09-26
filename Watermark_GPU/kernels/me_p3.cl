__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __constant int* Rx_mappings,
    __local float Rx_local[64][36], //64 local threads, 36 values each
    __local float rx_local[8][64],
    __local float rx_partial[8][8]) //helper scratch memory for rx calculation
{
    const int x = get_global_id(1), y = get_global_id(0);
    const int width = get_image_width(image);
    const int padded_width = get_global_size(1);
    const int local_id = get_local_id(1);
    const int output_index = (y * padded_width) + x;
    const bool is_padded = padded_width > width;

    //initialize rx shared memory with coalesced access
    for (int i = 0; i < 8; i++)
        rx_local[i][local_id] = 0.0f;
    rx_partial[local_id % 8][local_id / 8] = 0.0f;

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
            rx_local[i][local_id] = x_[i] * current_pixel;
            for (int j = i; j < 8; j++)
                Rx_local[local_id][counter++] = x_[i] * x_[j]; 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    const int limit = (is_padded && padded_width - x <= 64) ? 64 - (padded_width - width) : 64;
    float reduction_sum_Rx = 0.0f, reduction_sum_rx = 0.0f;
    for (int j = 0; j < limit; j++)
        reduction_sum_Rx += Rx_local[j][Rx_mappings[local_id]];

    //optimized summation for rx: normally we would sum 64 values per line/thread for a total of 8 sums
    //but this introduces heavy uncoalesced shared loads and bank conflicts, so we assign each of the 64 threads
    //to partially sum 8 horizontal values, and then the first 8 threads will fully sum the partial sums
    for (int i = 0; i < 8; i++)
        reduction_sum_rx += rx_local[local_id / 8][((local_id % 8) * 8) + i];
    rx_partial[local_id % 8][local_id / 8] = reduction_sum_rx;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float row_sum = 0.0f;
    if (local_id < 8) {
        for (int i = 0; i < 8; i++)
            row_sum += rx_partial[i][local_id];
        rx[(output_index / 8) + local_id] = row_sum;
    }
    Rx[output_index] = reduction_sum_Rx;
}