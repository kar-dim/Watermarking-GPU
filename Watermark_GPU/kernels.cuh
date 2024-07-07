#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void nvf(cudaTextureObject_t texObj, float* m_nvf, const int p_squared, const int pad, const int width, const int height);
__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const int width, const int padded_width, const int height);