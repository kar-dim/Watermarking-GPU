#include "kernels.cuh"

__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height)
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

    int counter = 0;
    float x_[9];
    if (x < width)
    {
        for (int i = y - 1; i <= y + 1; i++)
            for (int j = x - 1; j <= x + 1; j++)
                x_[counter++] = tex2D<float>(texObj, i, j);
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        #pragma unroll
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 8 rx values
        #pragma unroll
        for (int i = 0; i < 8; i++)
            RxLocal[localId][i] = x_[i] * current_pixel;
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
    {
        counter = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            for (int j = i; j < 8; j++)
                RxLocal[localId][counter++] = x_[i] * x_[j];
    }
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