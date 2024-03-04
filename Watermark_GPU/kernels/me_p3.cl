__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __global float* neighb,
    __local float Rx_local[4096]
    )
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
        | CLK_ADDRESS_CLAMP
        | CLK_FILTER_NEAREST;

    const uint width = get_image_width(image);
    const uint height = get_image_height(image);
    const int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
    const int local_id = get_local_id(1);

    uint k = 0;
    float x_[9];
    for (int j = pixelcoord.x - 1; j <= pixelcoord.x + 1; j++) {
        for (int i = pixelcoord.y - 1; i <= pixelcoord.y + 1; i++) {
            x_[k++] = read_imagef(image, sampler, (int2)(j, i)).x;
        }
    }
    
    const float cur_value = x_[4];

    for (int i = 4; i < 8; i++)
        x_[i] = x_[i + 1];

    int counter0 = 0;
    for (int i = 0; i < 64; i++) {
        Rx_local[(local_id * 64) + i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    counter0 = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            //printf("local_id: %d, counter0: %d, Rx_local[] index: %d\n", local_id, counter0, local_id * counter0);
            Rx_local[(local_id * counter0) + counter0] += x_[i] * x_[j];
            counter0++;
        }
    }

    const uint y_minus_pad_mul_8 = (pixelcoord.y) * 8;
    const uint y_minus_pad_mul_64 = y_minus_pad_mul_8 * 8;
    const uint x_minus_pad_mul_8_mul_real_height = (pixelcoord.x) * (height) * 8;
    const uint x_minus_pad_mul_64_mul_real_height = x_minus_pad_mul_8_mul_real_height * 8;

    //uint counter = 0;
    for (int i = 0; i < 8; i++) {
        const uint base_index = i + x_minus_pad_mul_8_mul_real_height + y_minus_pad_mul_8;
        rx[base_index] = x_[i] * cur_value;
        neighb[base_index] = x_[i];
        //for (int j = 0; j < 8; j++) {
            //Rx[counter + x_minus_pad_mul_64_mul_real_height + y_minus_pad_mul_64] = x_[i] * x_[j];
            //counter++;
        //}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        float chunk64sum = 0;
        for (int i = 0; i <64; i++) {
            chunk64sum = 0;
            for (int j = 0; j < 4096; j+=64) {
                chunk64sum += Rx_local[i + j];
            }
            Rx[i + ((pixelcoord.x) * (height)) + (pixelcoord.y)] = chunk64sum;
        }
    }
}