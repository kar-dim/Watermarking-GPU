#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace cuda_utils {

    template <typename T>
    T* cudaMallocPtr(std::size_t count)
    {
        T* ptr = nullptr;
        cudaMalloc(&ptr, count * sizeof(T));
        return ptr;
    }

    dim3 grid_size_calculate(const dim3 blockSize, const int rows, const int cols) {
        return dim3 ((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    }
}
