__kernel void me(__read_only image2d_t padded,
    __global float* Rx,
    __global float* rx,
    __global float* neighb
    )
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
        | CLK_ADDRESS_NONE
        | CLK_FILTER_NEAREST;

    const uint width = get_image_width(padded);
    const uint height = get_image_height(padded);

    const uint2 pixelcoord = (uint2) (get_global_id(0), get_global_id(1));

    if (pixelcoord.y <= height - 2 && pixelcoord.y >= 1 && pixelcoord.x <= width - 2 && pixelcoord.x >= 1) {
        uint k = 0;
        float x_[9];
        for (int j = pixelcoord.x - 1; j <= pixelcoord.x + 1; j++) {
            for (int i = pixelcoord.y - 1; i <= pixelcoord.y + 1; i++) {
                x_[k++] = read_imagef(padded, sampler, (int2)(j, i)).x;
            }
        }
        const float cur_value = x_[4];

        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        const uint y_minus_pad_mul_8 = (pixelcoord.y - 1) * 8;
        const uint y_minus_pad_mul_64 = y_minus_pad_mul_8 * 8;
        const uint x_minus_pad_mul_8_mul_real_height = (pixelcoord.x - 1) * (height - 2) * 8;
        const uint x_minus_pad_mul_64_mul_real_height = x_minus_pad_mul_8_mul_real_height * 8;

        uint counter = 0;
        for (int i = 0; i < 8; i++) {
            float cur_x = x_[i];
            int base_index = i + x_minus_pad_mul_8_mul_real_height + y_minus_pad_mul_8;
            rx[base_index] = cur_x * cur_value;
            neighb[base_index] = cur_x;
            for (int j = 0; j < 8; j++) {
                Rx[counter + x_minus_pad_mul_64_mul_real_height + y_minus_pad_mul_64] = cur_x * x_[j];
                counter++;
            }
        }
    }
}