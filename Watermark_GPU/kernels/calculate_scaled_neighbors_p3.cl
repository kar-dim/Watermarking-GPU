__kernel void calculate_scaled_neighbors_p3(
    __read_only image2d_t image, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //image2d is transposed, so we read the opposite dimensions
    const int width = get_image_height(image), height = get_image_width(image);
    const int outputIndex = (x * height + y);
    //calculate the dot product of the coefficients and the neighborhood for this pixel
    if (x < width && y < height) 
    {
        float dot = 0.0f;
        dot += coeffs[0] * read_imagef(image, sampler, (int2)(y - 1, x - 1)).x;
        dot += coeffs[1] * read_imagef(image, sampler, (int2)(y - 1, x)).x;
        dot += coeffs[2] * read_imagef(image, sampler, (int2)(y - 1, x + 1)).x;
        dot += coeffs[3] * read_imagef(image, sampler, (int2)(y, x - 1)).x;
        dot += coeffs[4] * read_imagef(image, sampler, (int2)(y, x + 1)).x;
        dot += coeffs[5] * read_imagef(image, sampler, (int2)(y + 1, x - 1)).x;
        dot += coeffs[6] * read_imagef(image, sampler, (int2)(y + 1, x)).x;
        dot += coeffs[7] * read_imagef(image, sampler, (int2)(y + 1, x + 1)).x;
		x_[outputIndex] = dot;
    }
}