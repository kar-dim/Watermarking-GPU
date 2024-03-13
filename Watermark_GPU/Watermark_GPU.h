#pragma once
#include "opencl_init.h"
#include "INIReader.h"

int test_for_image(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr);
int test_for_video(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr);
