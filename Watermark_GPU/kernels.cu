#include "kernels.cuh"

__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const int width, const int paddedWidth, const int height) 
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
	const int localId = threadIdx.y;
	const int outputIndex = (y * paddedWidth) + x;

    __shared__ float RxLocal[64][36];
    __shared__ float rxLocal[8][64];
    __shared__ float rxPartial[8][8];

    //initialize shared memory with coalesced access
    for (int i = 0; i < 8; i++)
        rxLocal[i][localId] = 0.0f;
    if (x >= width) 
    {
        #pragma unroll
        for (int i = 0; i < 36; i++)
            RxLocal[localId][i] = 0.0f;
    }
    rxPartial[localId % 8][localId / 8] = 0.0f;

    if (y >= height)
        return;

    if (x < width) 
    {
        int counter = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[counter++] = tex2D<float>(texObj, j, i);
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        #pragma unroll
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 36 local Rx and 8 local rx values
        counter = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) 
        {
            rxLocal[i][localId] = x_[i] * current_pixel;
            #pragma unroll
            for (int j = i; j < 8; j++)
                RxLocal[localId][counter++] = x_[i] * x_[j];
        }
    }

    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    __syncthreads();
    float reduction_sum_Rx = 0.0f, reduction_sum_rx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; j++)
        reduction_sum_Rx += RxLocal[j][RxMappings[localId]];

    //optimized summation for rx: normally we would sum 64 values per line/thread for a total of 8 sums
    //but this introduces heavy uncoalesced shared loads and bank conflicts, so we assign each of the 64 threads
    //to partially sum 8 horizontal values, and then the first 8 threads will fully sum the partial sums
    #pragma unroll
    for (int i = 0; i < 8; i++)
        reduction_sum_rx += rxLocal[localId / 8][((localId % 8) * 8) + i];
    rxPartial[localId % 8][localId / 8] = reduction_sum_rx;
    __syncthreads();

    float row_sum = 0.0f;
    if (localId < 8)
    {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            row_sum += rxPartial[i][localId];
        rx[(outputIndex / 8) + localId] = row_sum;
    }
    Rx[outputIndex] = reduction_sum_Rx;
}

__global__ void calculate_neighbors_p3(cudaTextureObject_t texObj, float* x_, const int width, const int height)
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputIndex = (x * height + y);

    if (x < width && y < height) 
    {
        //store 8 neighboring pixels into global memory (coalesced writes)
        x_[0 * width * height + outputIndex] = tex2D<float>(texObj, x - 1, y - 1);
        x_[1 * width * height + outputIndex] = tex2D<float>(texObj, x - 1, y);
        x_[2 * width * height + outputIndex] = tex2D<float>(texObj, x - 1, y + 1);
        x_[3 * width * height + outputIndex] = tex2D<float>(texObj, x, y - 1);
        x_[4 * width * height + outputIndex] = tex2D<float>(texObj, x, y + 1);
        x_[5 * width * height + outputIndex] = tex2D<float>(texObj, x + 1, y - 1);
        x_[6 * width * height + outputIndex] = tex2D<float>(texObj, x + 1, y);
        x_[7 * width * height + outputIndex] = tex2D<float>(texObj, x + 1, y + 1);
    }
}