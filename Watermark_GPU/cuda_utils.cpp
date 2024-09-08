#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <utility>

namespace cuda_utils {

    //Simple wrapper of cudaMalloc for float data to reduce boilerplate
    float* cudaMallocPtr(const std::size_t count)
    {
        float* ptr = nullptr;
        cudaMalloc(&ptr, count * sizeof(float));
        return ptr;
    }

    //Helper method to calculate kernel grid and block size from given 2D dimensions
    dim3 grid_size_calculate(const dim3 blockSize, const int rows, const int cols) 
    {
        return dim3((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    }

    //Simple wrapper of cudaMallocArray to reduce boilerplate
    cudaArray* cudaMallocArray(const std::size_t cols, const std::size_t rows)
    {
        cudaArray* cuArray;
        auto cudaChannelDescriptor = cudaCreateChannelDesc<float>();
        cudaMallocArray(&cuArray, &cudaChannelDescriptor, cols, rows);
        return cuArray;
    }

    //Creates a cudaResourceDesc from a specified cudaArray
    cudaResourceDesc createResourceDescriptor(cudaArray* cuArray)
    {
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        return resDesc;
    }

    //creates a cudaTextureDesc with these properties:
    //Border addressing mode, point filtering mode, element read mode, non-normalized coords
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

    //creates a texture object from the given cuda resource and cuda texture descriptors
    cudaTextureObject_t createTextureObject(const cudaResourceDesc& pResDesc, const cudaTextureDesc& pTexDesc)
    {
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &pResDesc, &pTexDesc, NULL);
        return texObj;
    }

    //get a cudaDeviceProp handle to query for various device information
    cudaDeviceProp getDeviceProperties() 
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        return properties;
    }

    //copy data from arrayfire to a cudaArray, and then create the texture object
    std::pair<cudaTextureObject_t, cudaArray*> copy_array_to_texture_data(const float* data, const unsigned int rows, const unsigned int cols) 
    {
	    cudaArray* cuArray = cuda_utils::cudaMallocArray(cols, rows);
	    cudaMemcpy2DToArray(cuArray, 0, 0, data, cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
	    cudaResourceDesc resDesc = cuda_utils::createResourceDescriptor(cuArray);
	    cudaTextureDesc texDesc = cuda_utils::createTextureDescriptor();
	    cudaTextureObject_t texObj = cuda_utils::createTextureObject(resDesc, texDesc);
	    return std::make_pair(texObj, cuArray);
    }

    //helper method to cleanup cuda texture data and to synchronize the stream
    void synchronize_and_cleanup_texture_data(cudaStream_t stream, const std::pair<cudaTextureObject_t, cudaArray*>& texture_data)
    {
        cudaDestroyTextureObject(texture_data.first);
        cudaFreeArray(texture_data.second);
        cudaStreamSynchronize(stream);
    }
}