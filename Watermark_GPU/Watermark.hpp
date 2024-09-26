#pragma once
#include "arrayfire.h"
#include "opencl_init.h"
#include <af/opencl.h>
#include <concepts>
#include <string>
#include <utility>

enum MASK_TYPE 
{
	ME,
	NVF
};

enum IMAGE_TYPE 
{
	RGB,
	GRAYSCALE
};

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class Watermark {

private:
	static constexpr int Rx_mappings[64]
	{
		0,  1,  2,  3,  4,  5,  6,  7,
		1,  8,  9,  10, 11, 12, 13, 14,
		2,  9,  15, 16, 17, 18, 19, 20,
		3,  10, 16, 21, 22, 23, 24, 25,
		4,  11, 17, 22, 26, 27, 28, 29,
		5,  12, 18, 23, 27, 30, 31, 32,
		6,  13, 19, 24, 28, 31, 33, 34,
		7,  14, 20, 25, 29, 32, 34, 35
	};
	const cl::Context context{ afcl::getContext(true) };
	const cl::CommandQueue queue{ afcl::getQueue(true) }; /*custom_queue{context, cl::Device{afcl::getDeviceId()}}; */
	const std::vector<cl::Program> programs;
	const std::string w_file_path;
	const int p;
	const float strength_factor;
	af::array rgb_image, image, w;
	cl::Image2D image2d;
	const cl::Buffer Rx_mappings_buff{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 64, (void*)Rx_mappings, NULL };
	af::array Rx_partial, rx_partial, custom_mask, neighbors;

	std::pair<af::array, af::array> correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int rows, const int padded_cols) const;
	float calculate_correlation(const af::array& e_u, const af::array& e_z) const;
	af::array execute_texture_kernel(const af::array& image, const cl::Program& program, const std::string kernel_name, const af::array& output, const unsigned int local_mem_elements = 0) const;
	af::array compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed) const;
	af::array compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& error_sequence) const;
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients) const;
	template<std::same_as<af::array>... Args>
	static void unlock_arrays(const Args&... arrays) { (arrays.unlock(), ...); }
public:
	Watermark(const af::array& rgb_image, const af::array &image, const std::string &w_file_path, const int p, const float psnr, const std::vector<cl::Program> &programs);
	Watermark(const std::string &w_file_path, const int p, const float psnr, const std::vector<cl::Program>&programs);
	void load_W(const dim_t rows, const dim_t cols);
	void load_image(const af::array& image);
	af::array make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type, IMAGE_TYPE image_type) const;
	float mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type) const;
	float mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients) const;
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
	
};