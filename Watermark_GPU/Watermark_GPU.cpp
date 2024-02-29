#pragma warning(disable:4996)
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "INIReader.h"
#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#define cimg_use_opencv
#define cimg_use_cpp11 1
#define cimg_use_png
#include "CImg.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <iomanip>
#include <arrayfire.h>
#include <af/opencl.h>
#include <af/util.h>
#include <thread>
#include <omp.h>

using std::cout;
using std::string;
using namespace cimg_library;

/*!
 *  \brief  This is a project implementation of my Thesis with title: 
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(void)
{
	//open parameters file
	INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load opencl configuration file\n";
		return -1;
	}

	//setup opencl parameters from arrayfire
	const cl::Context context(afcl::getContext());
	const cl::CommandQueue queue(afcl::getQueue());
	const cl::Device device(afcl::getDeviceId());

	af::info();
	cout << "\n";

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", -1.0f));
	UtilityFunctions::max_workgroup_size = inir.GetInteger("opencl", "max_workgroup_size", -1);
	size_t device_maxWorkGroupSize;
	device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_maxWorkGroupSize);
	if (UtilityFunctions::max_workgroup_size == -1)
		UtilityFunctions::max_workgroup_size = static_cast<int>(device_maxWorkGroupSize);
	else {
		if (UtilityFunctions::max_workgroup_size < 2 || (UtilityFunctions::max_workgroup_size % 2 != 0) || UtilityFunctions::max_workgroup_size > device_maxWorkGroupSize) {
			cout << " ERROR: MAX_WORKGROUP_SIZE parameter must NOT exceed selected device's MAX_WORKGROUP_SIZE limitation and must be a positive number and power of 2\n";
			return -1;
		}
	}

	//TODO for p>3 we have problems with ME masking buffers
	if (p != 3) {
		cout << "For now, only p=3 is allowed\n";
		return -1;
	}
	/*if (p <= 0 || p % 2 != 1 || p > 9) {
		cout << "p parameter must be a positive odd number less than 9\n";
		return -1;
	}*/

	if (psnr <= 0) {
		cout << "PSNR must be a positive number\n";
		return -1;
	}

	//compile opencl kernels
	std::string program_data;
	cl::Program program_nvf, program_me;
	try {
		program_data = UtilityFunctions::loadProgram("kernels/nvf.cl");
		program_nvf = cl::Program(context, program_data);
		program_nvf.build({ device }, "-cl-fast-relaxed-math");
		program_data = UtilityFunctions::loadProgram("kernels/me_p3.cl");
		program_me = cl::Program(context, program_data);
		program_me.build({ device }, "-cl-fast-relaxed-math");
	}
	catch (cl::Error& e) {
		cout << "Could not build a kernel, Reason:\n\n";
		cout << e.what();
		if (program_nvf.get() != NULL)
			cout << program_nvf.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		if (program_me.get() != NULL)
			cout << program_me.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return -1;
	}

	//test algorithms
	int return_value = inir.GetInteger("parameters_video", "test_for_video", -1) == 1 ? 
			UtilityFunctions::test_for_video(device, queue, context, program_nvf, program_me, inir, p, psnr) :
			UtilityFunctions::test_for_image(device, queue, context, program_nvf, program_me, inir, p, psnr);
	system("pause");
	return return_value;
}