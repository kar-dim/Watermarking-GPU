#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#undef max
#undef min
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#ifdef __NVIDIA__
#include <CL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif
#endif