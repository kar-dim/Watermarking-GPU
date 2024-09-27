__kernel void nvf(__read_only image2d_t image, 
				  __global float* nvf)
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	const int width = get_image_width(image), height = get_image_height(image);
	const int x = get_global_id(1), y = get_global_id(0);
	const int pSquared = p * p;
	const int pad = p / 2;

	if (y >= height || x >= width)
		return;

	int i, j, k = 0;
	float mean = 0.0f, variance = 0.0f;
	//p_squared is supplied at compile-time to minimize array elements (VLAs not supported in OpenCL)
	float localValues[pSquared];
	for (j = x - pad; j <= x + pad; j++) 
	{
		for (i = y - pad; i <= y + pad; i++) 
		{
			localValues[k] = read_imagef(image, sampler, (int2)(j, i)).x;
			mean += localValues[k];
			k++;
		}
	}
	mean /= pSquared;
	for (i = 0; i < pSquared; i++)
		variance += pown(localValues[i] - mean, 2);

	//calculate mask and write pixel value
	nvf[(x * height) + y] = variance / ((pSquared - 1) + variance);
}