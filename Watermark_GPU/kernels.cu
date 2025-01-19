#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

__device__ half8 make_half8(const float a, const float b, const float c, const float d, const float e, const float f, const float g, const float h)
{
    return half8 { __float2half(a), __float2half(b), __float2half(c), __float2half(d), __float2half(e), __float2half(f), __float2half(g), __float2half(h) };
}

__device__ half8 make_half8(const half a, const half b, const half c, const half d, const half e, const half f, const half g, const half h)
{
    return half8 { a, b, c, d, e, f, g, h };
}

__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half x_0, const half x_1, const half x_2, const half x_3, const half x_4, const half x_5, const half x_6, const half x_7, const half x_8)
{
    *RxLocalVec8 = make_half8(__hmul(x_0, x_4), __hmul(x_1, x_4), __hmul(x_2, x_4), __hmul(x_3, x_4),__hmul(x_5, x_4), __hmul(x_6, x_4), __hmul(x_7, x_4), __hmul(x_8, x_4));
}

__device__ void me_p3_RxCalculate(half8* RxLocalVec8, const half x_0, const half x_1, const half x_2, const half x_3, const half x_5, const half x_6, const half x_7, const half x_8)
{
    const half zero = __float2half(0.0f);
    RxLocalVec8[0] = make_half8(__hmul(x_0, x_0), __hmul(x_0, x_1), __hmul(x_0, x_2), __hmul(x_0, x_3), __hmul(x_0, x_5), __hmul(x_0, x_6), __hmul(x_0, x_7), __hmul(x_0, x_8));
    RxLocalVec8[1] = make_half8(__hmul(x_1, x_1), __hmul(x_1, x_2), __hmul(x_1, x_3), __hmul(x_1, x_5), __hmul(x_1, x_6), __hmul(x_1, x_7), __hmul(x_1, x_8), __hmul(x_2, x_2));
    RxLocalVec8[2] = make_half8(__hmul(x_2, x_3), __hmul(x_2, x_5), __hmul(x_2, x_6), __hmul(x_2, x_7), __hmul(x_2, x_8), __hmul(x_3, x_3), __hmul(x_3, x_5), __hmul(x_3, x_6));
    RxLocalVec8[3] = make_half8(__hmul(x_3, x_7), __hmul(x_3, x_8), __hmul(x_5, x_5), __hmul(x_5, x_6), __hmul(x_5, x_7), __hmul(x_5, x_8), __hmul(x_6, x_6), __hmul(x_6, x_7));
    RxLocalVec8[4] = make_half8(__hmul(x_6, x_8), __hmul(x_7, x_7), __hmul(x_7, x_8), __hmul(x_8, x_8), zero, zero, zero, zero);
}

__global__ void me_p3(cudaTextureObject_t texObj, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int outputIndex = (y * paddedWidth) + x;

    //re-use shared memory for Rx and rx calculation, helps with occupancy
    __shared__ half RxLocal[64][40]; //36 + 4 for 16-byte alignment (in order to use vectorized 128-bit load/store)
    half8* RxLocalVec8 = reinterpret_cast<half8*>(RxLocal[threadIdx.x]);

    //initialize shared memory, assign a portion for all threads for parallelism
    #pragma unroll
    for (int i = 0; i < 5; i++)
        RxLocalVec8[i] = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    if (y >= height)
        return;

    //load 3x3 window
    half x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8;
    if (x < width)
    {
        x_0 = __float2half(tex2D<float>(texObj, y - 1, x - 1));
        x_1 = __float2half(tex2D<float>(texObj, y - 1, x));
        x_2 = __float2half(tex2D<float>(texObj, y - 1, x + 1));
        x_3 = __float2half(tex2D<float>(texObj, y, x - 1));
        x_4 = __float2half(tex2D<float>(texObj, y, x)); //x_4 is central pixel
        x_5 = __float2half(tex2D<float>(texObj, y, x + 1));
        x_6 = __float2half(tex2D<float>(texObj, y + 1, x - 1));
        x_7 = __float2half(tex2D<float>(texObj, y + 1, x));
        x_8 = __float2half(tex2D<float>(texObj, y + 1, x + 1));
        //calculate this thread's 8 rx values
        me_p3_rxCalculate(RxLocalVec8, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
    }
    __syncthreads();

    //optimized summation for rx with warp shuffling
    float sum = 0;
    const int row = threadIdx.x / 8;
    #pragma unroll
    for (int i = 0; i < 64; i += 8)
        sum += __half2float(RxLocal[(threadIdx.x + i) % 64][row]);
    // reduce 32 results to 4 per warp
    for (int i = 4; i > 0; i = i / 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    if (threadIdx.x % 8 == 0)
        rx[(outputIndex + row) / 8] = sum;
    __syncthreads();

    //calculate 36 Rx values
    if (x < width)
        me_p3_RxCalculate(RxLocalVec8, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    __syncthreads();

    //simplified summation for Rx
    //we cannot use warp shuffling because it introduces too much stalling for Rx
    sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++)
        sum += __half2float(RxLocal[i][RxMappings[threadIdx.x]]);
    Rx[outputIndex] = sum;
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