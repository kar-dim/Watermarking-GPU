#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#undef max
#undef min
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif