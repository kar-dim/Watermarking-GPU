#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace cuda_utils {

    float* cudaMallocPtr(const std::size_t count)
    {
        float* ptr = nullptr;
        cudaMalloc(&ptr, count * sizeof(float));
        return ptr;
    }

    dim3 grid_size_calculate(const dim3 blockSize, const int rows, const int cols) {
        return dim3((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    }

    cudaArray* cudaMallocArray(const std::size_t cols, const std::size_t rows)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, cols, rows);
        return cuArray;
    }

    cudaResourceDesc createResourceDescriptor(cudaArray* cuArray)
    {
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        return resDesc;
    }

    cudaTextureDesc createTextureDescriptor()
    {
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        return texDesc;
    }
    cudaTextureObject_t createTextureObject(const cudaResourceDesc& pResDesc, const cudaTextureDesc& pTexDesc)
    {
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &pResDesc, &pTexDesc, NULL);
        return texObj;
    }
}