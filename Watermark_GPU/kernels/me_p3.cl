__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __global float* neighb,
    __local float Rx_local[4096]
    )
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const int width = get_image_width(image), height = get_image_height(image);
    const int x = get_global_id(0), y = get_global_id(1);
    const int local_id = get_local_id(1);

    int k = 0;
    float x_[9];
    for (int j = x - 1; j <= x + 1; j++)
        for (int i = y - 1; i <= y + 1; i++)
            x_[k++] = read_imagef(image, sampler, (int2)(j, i)).x;
    const float cur_value = x_[4];

    //shift neighborhood values, so that consecutive values are neighbors only
    for (int i = 4; i < 8; i++)
        x_[i] = x_[i + 1];

    //initialize local memory sums
#pragma unroll
    for (int i = 0; i < 64; i++) {
        Rx_local[(local_id * 64) + i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //TODO debug
    int counter = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Rx_local[(local_id * 64) + counter] += x_[i] * x_[j];
            counter++;
        }
    }

    int base_index = (x * height * 8) + (y * 8);
    for (int i = 0; i < 8; i++) {
        const int output_index = i + base_index;
        rx[output_index] = x_[i] * cur_value;
        neighb[output_index] = x_[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //first local thread of the group will do the reduction sums into the global memory
    if (local_id == 0) {
        const int output_index = (x * height) + y;
        float chunk64sum = 0;
        for (int i = 0; i <64; i++) {
            chunk64sum = 0;
            for (int j = 0; j < 4096; j+=64) {
                chunk64sum += Rx_local[i + j];
            }
            Rx[i + output_index] = chunk64sum;
        }
    }
}