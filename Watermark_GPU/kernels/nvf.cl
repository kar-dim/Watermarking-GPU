__kernel void nvf(__read_only image2d_t image, 
				  __global float* nvf)
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	//image2d is transposed, so we read the opposite dimensions
	const int width = get_image_height(image), height = get_image_width(image);
	const int x = get_global_id(1), y = get_global_id(0);
	//p is supplied at compile-time to allow the compiler to optimize more efficiently
	const int pSquared = p * p;
	const int pad = p / 2;

	if (y >= height || x >= width)
		return;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = y - pad; i <= y + pad; i++)
	{
		for (int j = x - pad; j <= x + pad; j++)
		{
			float pixelValue = read_imagef(image, sampler, (int2)(i, j)).x;
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	float mean = sum / pSquared;
	float variance = (sumSq / pSquared) - (mean * mean);
	//calculate mask and write pixel value
	nvf[(x * height) + y] = variance / (1 + variance);
}