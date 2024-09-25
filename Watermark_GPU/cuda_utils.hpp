#pragma once
#include <cuda_runtime.h>
#include <utility>

//Helper functions related to cuda
namespace cuda_utils 
{
    dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols);
    cudaArray* cudaMallocArray(const std::size_t cols, const std::size_t rows);
    cudaResourceDesc createResourceDescriptor(cudaArray* cuArray);
    cudaTextureDesc createTextureDescriptor();
    cudaTextureObject_t createTextureObject(const cudaResourceDesc& pResDesc, const cudaTextureDesc& pTexDesc);
    cudaDeviceProp getDeviceProperties();
    std::pair<cudaTextureObject_t, cudaArray*> createTextureData(const unsigned int rows, const unsigned int cols);
    void copyDataToCudaArray(const float* data, const unsigned int rows, const unsigned int cols, cudaArray* cuArray);
    void copyDataToCudaArrayAsync(const float* data, const unsigned int rows, const unsigned int cols, cudaArray* cuArray, cudaStream_t stream);
}