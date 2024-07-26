#pragma once
#include "INIReader.h"
#include <cuda_runtime.h>
int test_for_image(const INIReader& inir, cudaDeviceProp& properties, const int p, const float psnr);
int test_for_video(const INIReader& inir, cudaDeviceProp& properties, const int p, const float psnr);