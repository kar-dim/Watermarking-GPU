__kernel void calculate_neighbors_p3(
    __read_only image2d_t image, 
    __global float* x_)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //image2d is transposed, so we read the opposite dimensions
    const int width = get_image_height(image), height = get_image_width(image);
    const int outputIndex = (x * height + y);

    if (x < width && y < height) 
    {
        // Write 8 neighboring pixels into global memory (coalesced writes)
        x_[0 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y - 1, x - 1)).x;
        x_[1 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y - 1, x)).x;
        x_[2 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y - 1, x + 1)).x;
        x_[3 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y, x - 1)).x;
        x_[4 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y, x + 1)).x;
        x_[5 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y + 1, x - 1)).x;
        x_[6 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y + 1, x)).x;
        x_[7 * width * height + outputIndex] = read_imagef(image, sampler, (int2)(y + 1, x + 1)).x;
    }
}