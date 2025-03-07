#pragma once
#include <string>
inline const std::string nvf = R"CLC(

__kernel void nvf(__read_only image2d_t image, 
	__global float *nvf,
	__local float region[16 + 2 * (p/2)][16 + 2 * (p/2)])
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	//image2d is transposed, so we read the opposite dimensions
	const int width = get_image_height(image), height = get_image_width(image);
	const int x = get_global_id(1), y = get_global_id(0);
	//p is supplied at compile-time to allow the compiler to optimize more efficiently
	const int pSquared = p * p;
	const int pad = p / 2;
    const int localY = get_local_id(1);
    const int localX = get_local_id(0);
    const int shX = localY + pad;
    const int shY = localX + pad;

    //load current pixel value
    region[shY][shX] = read_imagef(image, sampler, (int2)(y, x)).x;

    // Load the padded regions (only edge threads)
    if (localX == 0)
    {
        for (int i = pad; i > 0; i--)
            region[shY - i][shX] = read_imagef(image, sampler, (int2)(y - i, x)).x;
    }
    if (localX == 15)
    {
        for (int i = pad; i > 0; i--)
            region[shY + i][shX] = read_imagef(image, sampler, (int2)(y + i, x)).x;
    }
    if (localY == 0)
    {
        for (int i = pad; i > 0; i--)
            region[shY][shX - i] = read_imagef(image, sampler, (int2)(y, x - i)).x;
    }
    if (localY == 15)
    {
        for (int i = pad; i > 0; i--)
            region[shY][shX + i] = read_imagef(image, sampler, (int2)(y, x + i)).x;
    }

    // Load the corners of the padded region (only edge threads)
    if (localX == 0 && localY == 0)
    {
        for (int i = -1; i >= -pad; i--)
            for (int j = -1; j >= -pad; j--)
                region[shY + i][shX + j] = read_imagef(image, sampler, (int2)(y + i, x + j)).x;
    }
    if (localX == 15 && localY == 15)
    {
        for (int i = 1; i <= pad; i++)
            for (int j = 1; j <= pad; j++)
                region[shY + i][shX + j] = read_imagef(image, sampler, (int2)(y + i, x + j)).x;
    }
    if (localX == 0 && localY == 15)
    {
        for (int i = -1; i >= -pad; i--)
            for (int j = 1; j <= pad; j++)
                region[shY + i][shX + j] = read_imagef(image, sampler, (int2)(y + i, x + j)).x;
    }
    if (localX == 15 && localY == 0)
    {
        for (int i = 1; i <= pad; i++)
            for (int j = -1; j >= -pad; j--)
                region[shY + i][shX + j] = read_imagef(image, sampler, (int2)(y + i, x + j)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    if (y >= height || x >= width)
        return;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = -pad; i <= pad; i++)
	{
		for (int j = -pad; j <= pad; j++)
		{
			float pixelValue = region[shY + i][shX + j];
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	float mean = sum / pSquared;
	float variance = (sumSq / pSquared) - (mean * mean);
	//calculate mask and write pixel value
	nvf[(x * height) + y] = variance / (1 + variance);
}
)CLC";