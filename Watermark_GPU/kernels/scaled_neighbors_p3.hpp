#pragma once
#include <string>
inline const std::string scaled_neighbors_p3 = R"CLC(

__kernel void scaled_neighbors_p3(
    __read_only image2d_t image, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs,
    __local float region[16 + 2][16 + 2]) //hold the 18 x 18 region for this 16 x 16 block
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int localX = get_local_id(0);
	const int localY = get_local_id(1);
    const int shX = localY + 1;
    const int shY = localX + 1;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //image2d is transposed, so we read the opposite dimensions
    const int width = get_image_height(image), height = get_image_width(image);

    //load current pixel value
    region[shY][shX] = read_imagef(image, sampler, (int2)(y, x)).x;

    // Load the padded regions (only edge threads)
    if (localX == 0)
        region[shY - 1][shX] = read_imagef(image, sampler, (int2)(y - 1, x)).x;
    if (localX == 15)
        region[shY + 1][shX] = read_imagef(image, sampler, (int2)(y + 1, x)).x;
    if (localY == 0)
        region[shY][shX - 1] = read_imagef(image, sampler, (int2)(y, x - 1)).x;
    if (localY == 15)
        region[shY][shX + 1] = read_imagef(image, sampler, (int2)(y, x + 1)).x;

    // Load the corners of the padded region (only edge threads)
    if (localX == 0 && localY == 0)
        region[shY - 1][shX - 1] = read_imagef(image, sampler, (int2)(y - 1, x - 1)).x;
    if (localX == 15 && localY == 15)
        region[shY + 1][shX + 1] = read_imagef(image, sampler, (int2)(y + 1, x + 1)).x;
    if (localX == 0 && localY == 15)
        region[shY - 1][shX + 1] = read_imagef(image, sampler, (int2)(y - 1, x + 1)).x;
    if (localX == 15 && localY == 0)
        region[shY + 1][shX - 1] = read_imagef(image, sampler, (int2)(y + 1, x - 1)).x;

    barrier(CLK_LOCAL_MEM_FENCE);

    //calculate the dot product of the coefficients and the neighborhood for this pixel
    if (x < width && y < height) 
    {
        float dot = 0.0f;
        dot += coeffs[0] * region[shY - 1][shX - 1];
        dot += coeffs[1] * region[shY - 1][shX];
        dot += coeffs[2] * region[shY - 1][shX + 1];
        dot += coeffs[3] * region[shY][shX - 1];
        dot += coeffs[4] * region[shY][shX + 1];
        dot += coeffs[5] * region[shY + 1][shX - 1];
        dot += coeffs[6] * region[shY + 1][shX];
        dot += coeffs[7] * region[shY + 1][shX + 1];
		x_[(x * height + y)] = dot;
    }
}
)CLC";