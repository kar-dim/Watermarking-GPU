__kernel void nvf(__read_only image2d_t image, 
				  __global float* m_nvf, 
				    const int p)
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE 
							| CLK_ADDRESS_CLAMP
							| CLK_FILTER_NEAREST;
	
	const uint width = get_image_width(image);
	const uint height = get_image_height(image);
	const int pad = p/2;
	const int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
	const uint p_squared = (uint)pown((float)p,2);
	int i,j, k=0;
	float mean=0.0f;
	float variance=0.0f;
	float values[81]; //9x9 block
	for (i = pixelcoord.y - pad; i <= pixelcoord.y + pad; i++){
		for (j = pixelcoord.x - pad; j<= pixelcoord.x + pad; j++){
			values[k] = read_imagef(image, sampler, (int2)(j, i)).x;
			mean += values[k++];
		}
	}
	mean /= p_squared;
	for (i = 0; i < p_squared; i++){
		variance += pown(values[i] - mean, 2);
	}
	//calculate mask
	const float nvf_mask = 1.0f - (1.0f / (1.0f + (variance/(p_squared - 1)) ) );
	//write pixel value
	m_nvf[(pixelcoord.x * height) + pixelcoord.y] = nvf_mask;
}