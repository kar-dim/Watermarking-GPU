#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8
{
	half a, b, c, d, e, f, g, h;
};
__host__ void setCoeffs(const float* c);

__device__ half8 make_half8(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f, const float& g, const float& h);
__device__ half8 make_half8(const half& a, const half& b, const half& c, const half& d, const half& e, const half& f, const half& g, const half& h);

//helper methods of ME kernel, to calculate block-wide Rx/rx values in shared memory
__device__ void me_p3_rxCalculate(half8* RxLocalVec, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_4, const half& x_5, const half& x_6, const half& x_7, const half& x_8);
__device__ void me_p3_RxCalculate(half8* RxLocalVec, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_5, const half& x_6, const half& x_7, const half& x_8);

// NVF kernel, calculates NVF values for each pixel in the image
// works for all p values (3,5,7 and 9)
template<int p, int pSquared = p * p, int pad = p / 2>
__global__ void nvf(cudaTextureObject_t texObj, float* nvf, const unsigned int width, const unsigned int height)
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int shX = threadIdx.y + pad;
    const int shY = threadIdx.x + pad;

    __shared__ float region[16 + (2 * pad)][16 + (2 * pad)]; //hold the 18 x 18 region for this 16 x 16 block

    //load current pixel value
    region[shY][shX] = tex2D<float>(texObj, y, x);

    // Load the padded regions (only edge threads)
    if (threadIdx.x == 0)
    {
		for (int i = pad; i > 0; i--)
			region[shY - i][shX] = tex2D<float>(texObj, y - i, x);
    }
    if (threadIdx.x == 15)
    {
        for (int i = pad; i > 0; i--)
            region[shY + i][shX] = tex2D<float>(texObj, y + i, x);
    }
    if (threadIdx.y == 0)
    {
        for (int i = pad; i > 0; i--)
            region[shY][shX - i] = tex2D<float>(texObj, y, x - i);
    }
    if (threadIdx.y == 15) 
    {
        for (int i = pad; i > 0; i--)
            region[shY][shX + i] = tex2D<float>(texObj, y, x + i);
    }

    // Load the corners of the padded region (only edge threads)
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int i = -1; i >= -pad; i--)
            for (int j = -1; j >= -pad; j--)
                region[shY + i][shX + j] = tex2D<float>(texObj, y + i, x + j);
    }
    if (threadIdx.x == 15 && threadIdx.y == 15)
    {
        for (int i = 1; i <= pad; i++)
            for (int j = 1; j <= pad; j++)
                region[shY + i][shX + j] = tex2D<float>(texObj, y + i, x + j);
    }
    if (threadIdx.x == 0 && threadIdx.y == 15)
    {
        for (int i = -1; i >= -pad; i--)
            for (int j = 1; j <= pad; j++)
                region[shY + i][shX + j] = tex2D<float>(texObj, y + i, x + j);
    }
    if (threadIdx.x == 15 && threadIdx.y == 0)
    {
        for (int i = 1; i <= pad; i++)
            for (int j = -1; j >= -pad; j--)
                region[shY + i][shX + j] = tex2D<float>(texObj, y + i, x + j);
    }
    __syncthreads();

	if (x >= width || y >= height)
		return;

	//load the neighbors to calculate the region's parameters for this pixel
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

//main ME kernel, calculates ME values for each pixel in the image
__global__ void me_p3(cudaTextureObject_t texObj, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height);

//main kernel for scaled neighbors calculation. used in ME kernel
__global__ void calculate_scaled_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height);