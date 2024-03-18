#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "UtilityFunctions.h"
#include <fstream>
#include <iostream>
#include <string>
#include <arrayfire.h>
#include <chrono>
#include <vector>
#include "CImg.h"

using namespace cimg_library;
using std::cout;

size_t UtilityFunctions::max_workgroup_size = -1;

//normalize στο [0,1] διάστημα
af::array UtilityFunctions::normalize_to_f32(af::array& a)
{
	float mx = af::max<float>(a);
	float mn = af::min<float>(a);
	float diff = mx - mn;
	return (a - mn) / diff;
}

//εμφάνιση πληροφοριών σχετικά με το OpenCL
void UtilityFunctions::print_opencl_info(cl::Platform &plat, cl::Device &device)
{
	cout << "Platform Name: " << plat.getInfo<CL_PLATFORM_NAME>() << "\n";
	cout << "Platform Vendor: " << plat.getInfo<CL_PLATFORM_VENDOR>() << "\n";
	cout << "Device Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
	cout << "Device Vendor Name: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
	cout << "Device OpenCL C version: " << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << "\n\n";
}

//χρησιμοποιείται για τη φόρτωση kernel
std::string UtilityFunctions::loadProgram(std::string input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open()) {
		cout << "Cannot open file: " << input << "\n";
		exit(EXIT_FAILURE);
	}
	std::string s(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
	return s;
}

//χρονομέτρηση
namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	float secs_passed() {
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000.0f);
	}
}