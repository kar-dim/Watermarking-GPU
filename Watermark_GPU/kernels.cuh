#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//constant array used for optimizing share memory accesses for Rx
//Helps with reducing the local memory required for each block for Rx arrays from 4096 to 2304
__constant__ int RxMappings[64] =
{
	0,  1,  2,  3,  4,  5,  6,  7,
	1,  8,  9,  10, 11, 12, 13, 14,
	2,  9,  15, 16, 17, 18, 19, 20,
	3,  10, 16, 21, 22, 23, 24, 25,
	4,  11, 17, 22, 26, 27, 28, 29,
	5,  12, 18, 23, 27, 30, 31, 32,
	6,  13, 19, 24, 28, 31, 33, 34,
	7,  14, 20, 25, 29, 32, 34, 35
};

struct alignas(16) half8
{
	half a, b, c, d, e, f, g, h;
};

__device__ half8 make_half8(const float a, const float b, const float c, const float d, const float e, const float f, const float g, const float h);
__device__ half8 make_half8(const half a, const half b, const half c, const half d, const half e, const half f, const half g, const half h);

//helper methods of ME kernel, to calculate block-wide Rx/rx values in shared memory
__device__ void me_p3_rxCalculate(half8* RxLocalVec, const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_4, const half x_5, const half x_6, const half x_7, const half x_8);
__device__ void me_p3_RxCalculate(half8* RxLocalVec, const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_5, const half x_6, const half x_7, const half x_8);

//NVF kernel, calculates NVF values for each pixel in the image
template<int p, int pSquared = p * p, int pad = p / 2>
__global__ void nvf(cudaTextureObject_t texObj, float* nvf, const unsigned int width, const unsigned int height)
{
	const int x = blockIdx.y * blockDim.y + threadIdx.y;
	const int y = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= width || y >= height)
		return;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = y - pad; i <= y + pad; i++)
	{
		for (int j = x - pad; j <= x + pad; j++)
		{
			float pixelValue = tex2D<float>(texObj, i, j);
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
//main kernel for neighbors calculation. used in ME kernel
__global__ void calculate_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height);