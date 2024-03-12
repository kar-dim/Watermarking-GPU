#pragma warning(disable:4996)
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Watermark_GPU.h"
#include "UtilityFunctions.h"
#include "INIReader.h"
#include <iostream>
#include <af/opencl.h>

using std::cout;
using std::string;

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

	af::info();
	cout << "\n";

	const cl::Context context(afcl::getContext(true));
	const cl::Device device({ afcl::getDeviceId() });

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", -1.0f));

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
	cl::Program program_nvf, program_me;
	try {
		std::string program_data = UtilityFunctions::loadProgram("kernels/nvf.cl");
		program_nvf = cl::Program(cl::Context{ afcl::getContext()}, program_data);
		const std::string nvf_buildFlags = "-cl-fast-relaxed-math -cl-mad-enable -Dp_squared=" + std::to_string(p * p);
		program_nvf.build({ device }, nvf_buildFlags.c_str());
		program_data = UtilityFunctions::loadProgram("kernels/me_p3.cl");
		program_me = cl::Program(context, program_data);
		program_me.build({ device }, "-cl-fast-relaxed-math -cl-mad-enable");
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
	try {
		inir.GetInteger("parameters_video", "test_for_video", -1) == 1 ?
			test_for_video(device, program_nvf, program_me, inir, p, psnr) :
			test_for_image(device, program_nvf, program_me, inir, p, psnr);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		system("pause");
		return -1;
	}
	system("pause");
	return 0;
}