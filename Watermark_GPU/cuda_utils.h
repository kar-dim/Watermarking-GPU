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
}
