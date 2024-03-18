#pragma once
#pragma warning(disable:4996)
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <string>
#include <random>
#include <arrayfire.h>
#include <chrono>

class UtilityFunctions {
public:
	static size_t max_workgroup_size;
	// normalize στο[0, 1] διάστημα
	static af::array normalize_to_f32(af::array& a);
	//εμφάνιση πληροφοριών σχετικά με το OpenCL
	static void print_opencl_info(cl::Platform& plat, cl::Device& device);
	//χρησιμοποιείται για τη φόρτωση kernel
	static std::string loadProgram(std::string input);
};

namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	float secs_passed();
}