#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void me_p3_rxCalculate(float RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float currentPixel, const float x_5, const float x_6, const float x_7, const float x_8) 
{
    RxLocal[localId][0] = x_0 * currentPixel;
    RxLocal[localId][1] = x_1 * currentPixel;
    RxLocal[localId][2] = x_2 * currentPixel;
    RxLocal[localId][3] = x_3 * currentPixel;
    RxLocal[localId][4] = x_5 * currentPixel;
    RxLocal[localId][5] = x_6 * currentPixel;
    RxLocal[localId][6] = x_7 * currentPixel;
    RxLocal[localId][7] = x_8 * currentPixel;
}

__device__ void me_p3_RxCalculate(float RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float x_5, const float x_6, const float x_7, const float x_8)
{
    RxLocal[localId][0] = x_0 * x_0;
    RxLocal[localId][1] = x_0 * x_1;
    RxLocal[localId][2] = x_0 * x_2;
    RxLocal[localId][3] = x_0 * x_3;
    RxLocal[localId][4] = x_0 * x_5;
    RxLocal[localId][5] = x_0 * x_6;
    RxLocal[localId][6] = x_0 * x_7;
    RxLocal[localId][7] = x_0 * x_8;
    RxLocal[localId][8] = x_1 * x_1;
    RxLocal[localId][9] = x_1 * x_2;
    RxLocal[localId][10] = x_1 * x_3;
    RxLocal[localId][11] = x_1 * x_5;
    RxLocal[localId][12] = x_1 * x_6;
    RxLocal[localId][13] = x_1 * x_7;
    RxLocal[localId][14] = x_1 * x_8;
    RxLocal[localId][15] = x_2 * x_2;
    RxLocal[localId][16] = x_2 * x_3;
    RxLocal[localId][17] = x_2 * x_5;
    RxLocal[localId][18] = x_2 * x_6;
    RxLocal[localId][19] = x_2 * x_7;
    RxLocal[localId][20] = x_2 * x_8;
    RxLocal[localId][21] = x_3 * x_3;
    RxLocal[localId][22] = x_3 * x_5;
    RxLocal[localId][23] = x_3 * x_6;
    RxLocal[localId][24] = x_3 * x_7;
    RxLocal[localId][25] = x_3 * x_8;
    RxLocal[localId][26] = x_5 * x_5;
    RxLocal[localId][27] = x_5 * x_6;
    RxLocal[localId][28] = x_5 * x_7;
    RxLocal[localId][29] = x_5 * x_8;
    RxLocal[localId][30] = x_6 * x_6;
    RxLocal[localId][31] = x_6 * x_7;
    RxLocal[localId][32] = x_6 * x_8;
    RxLocal[localId][33] = x_7 * x_7;
    RxLocal[localId][34] = x_7 * x_8;
    RxLocal[localId][35] = x_8 * x_8;
}

__global__ void me_p3(cudaTextureObject_t texObj, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int localId = threadIdx.x;
	const int outputIndex = (y * paddedWidth) + x;

    //re-use shared memory for Rx and rx calculation, helps with occupancy
    __shared__ float RxLocal[64][36]; 

    //initialize shared memory, assign a portion for all threads for parallelism
    #pragma unroll
    for (int i = 0; i < 36; i++)
        RxLocal[localId][i] = 0.0f;

    if (y >= height)
        return;

    float x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8;

    //calculate this thread's 8 rx values
    if (x < width)
    {
        x_0 = tex2D<float>(texObj, y - 1, x - 1);
        x_1 = tex2D<float>(texObj, y - 1, x);
        x_2 = tex2D<float>(texObj, y - 1, x + 1);
        x_3 = tex2D<float>(texObj, y, x - 1);
        currentPixel = tex2D<float>(texObj, y, x);
        x_5 = tex2D<float>(texObj, y, x + 1);
        x_6 = tex2D<float>(texObj, y + 1, x - 1);
        x_7 = tex2D<float>(texObj, y + 1, x);
        x_8 = tex2D<float>(texObj, y + 1, x + 1);
        me_p3_rxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8);
    }
    __syncthreads();

    //optimized summation for rx with warp shuffling
    float rxSum = 0;
    const int row = localId / 8;
    #pragma unroll
    for (int i = 0; i < 64; i += 8)
        rxSum += RxLocal[(localId + i) % 64][row];
    // reduce 32 results to 4 per warp
    for (int i = 4; i > 0; i = i / 2)
        rxSum += __shfl_down_sync(0xFFFFFFFF, rxSum, i);
    if (localId % 8 == 0)
        rx[(outputIndex + row) / 8] = rxSum;

    //calculate 36 Rx values
    if (x < width)
        me_p3_RxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    __syncthreads();

    //simplified summation for Rx
    //we cannot use warp shuffling because it introduces too much stalling for Rx
    float reduction_sum_Rx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; j++)
        reduction_sum_Rx += RxLocal[j][RxMappings[localId]];
    Rx[outputIndex] = reduction_sum_Rx;
}

__global__ void calculate_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height)
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputIndex = (x * height + y);

    if (x < width && y < height) 
    {
        //store 8 neighboring pixels into global memory (coalesced writes)
        x_[0 * width * height + outputIndex] = tex2D<float>(texObj, y - 1, x - 1);
        x_[1 * width * height + outputIndex] = tex2D<float>(texObj, y - 1, x);
        x_[2 * width * height + outputIndex] = tex2D<float>(texObj, y - 1, x + 1);
        x_[3 * width * height + outputIndex] = tex2D<float>(texObj, y, x - 1);
        x_[4 * width * height + outputIndex] = tex2D<float>(texObj, y, x + 1);
        x_[5 * width * height + outputIndex] = tex2D<float>(texObj, y + 1, x - 1);
        x_[6 * width * height + outputIndex] = tex2D<float>(texObj, y + 1, x);
        x_[7 * width * height + outputIndex] = tex2D<float>(texObj, y + 1, x + 1);
    }
}