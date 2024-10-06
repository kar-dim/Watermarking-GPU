#pragma once
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

template<int p, int pSquared = p * p, int pad = p / 2>
__global__ void nvf(cudaTextureObject_t texObj, float* nvf, const unsigned int width, const unsigned int height)
{
	const int x = blockIdx.y * blockDim.y + threadIdx.y;
	const int y = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= width || y >= height)
		return;

	int i, j, k = 0;
	float mean = 0.0f, variance = 0.0f, localMeanDiff;
	//maximum local values size is 81 for a 9x9 block
	float localValues[pSquared];
	for (i = y - pad; i <= y + pad; i++)
	{
		for (j = x - pad; j <= x + pad; j++) 
		{
			localValues[k] = tex2D<float>(texObj, i, j);
			mean += localValues[k];
			k++;
		}
	}
	mean /= pSquared;
	for (i = 0; i < pSquared; i++) 
	{
		localMeanDiff = localValues[i] - mean;
		variance += localMeanDiff * localMeanDiff;
	}
	//calculate mask and write pixel value
	nvf[(x * height) + y] = variance / ((pSquared - 1) + variance);
}

__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height);
__global__ void calculate_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height);