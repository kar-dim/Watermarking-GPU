#pragma once
#pragma warning(disable:4996)
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <arrayfire.h>
#include <string>
#include <chrono>
#include "INIReader.h"
#include <CImg.h>
using namespace cimg_library;
/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class UtilityFunctions {
public:
	static int test_for_image(const cl::Device& device, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr);
	static int test_for_video(const cl::Device& device, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr);
	static std::string loadProgram(const std::string input);
	template<typename T>
	static af::array cimg_yuv_to_afarray(const CImg<T> &cimg_image) {
		return af::transpose(af::array(cimg_image.width(), cimg_image.height(), cimg_image.get_channel(0).data()).as(af::dtype::f32));
	}
};

/*!
 *  \brief  simple methods to calculate execution times
 *  \author Dimitris Karatzas
 */
namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	float secs_passed();
}