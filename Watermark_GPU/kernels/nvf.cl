__kernel void nvf(__read_only image2d_t padded, 
				  __global float* m_nvf, 
				    const int p)
{	
	const sampler_t sampler= CLK_NORMALIZED_COORDS_FALSE 
							| CLK_ADDRESS_NONE 
							| CLK_FILTER_NEAREST;
	
	const uint width = get_image_width(padded);
	const uint height = get_image_height(padded);
	const uint int_pad = p/2;
	
	const int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
	if (pixelcoord.y <= height-int_pad-1 && pixelcoord.y >=int_pad && pixelcoord.x <= width-int_pad-1 && pixelcoord.x >=int_pad){
		float nvf_mask;
		const uint real_height = height - (2*int_pad);
		const uint p_squared = (int)pown((float)p,2);
		int i,j, k=0;
		float mean=0.0f;
		float variance=0.0f;
		float values[81]; //9x9 block
		for (i=pixelcoord.y-int_pad; i<= pixelcoord.y+int_pad; i++){
			for (j= pixelcoord.x-int_pad; j<= pixelcoord.x+int_pad; j++){
				values[k] = read_imagef(padded, sampler, (int2)(j,i) ).x;
				mean = mean + values[k];
				k++;
			}
		}
		mean = mean/p_squared;
		for (i=0; i<p_squared; i++){
			variance = variance + pown(values[i] - mean, 2);
		}
		//calculate mask
		nvf_mask = 1.0f - (1.0f / (1.0f + (variance/(p_squared-1)) ) );
		
		//write pixel value
		m_nvf[(pixelcoord.x - int_pad)*real_height + (pixelcoord.y - int_pad)] = nvf_mask;
	}
}