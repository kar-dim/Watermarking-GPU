#pragma once
#include "opencl_init.h"
#include <af/opencl.h>
#include <string>

using std::string;

enum MASK_TYPE {
	ME,
	NVF
};

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class WatermarkFunctions {

private:
	static constexpr int Rx_mappings[64]{
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
	const cl::CommandQueue queue{ afcl::getQueue(true) };
	const cl::Program program_me, program_custom;
	const string w_file_path, custom_kernel_name;
	const int p, p_squared, p_squared_minus_one, p_squared_minus_one_squared, pad;
	const float psnr;
	af::array image, w;
	dim_t rows, cols;

	af::array calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad);
	std::pair<af::array, af::array> correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols);
	float calculate_correlation(const af::array& e_u, const af::array& e_z);
	void compute_custom_mask(const af::array &image, af::array& m);
	void compute_prediction_error_mask(const af::array& image, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed);
	void compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& m_e, af::array& error_sequence);
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients);
	cl::Image2D copyBufferToImage(const cl_mem* image_buff, const dim_t rows, const dim_t cols);
public:
	WatermarkFunctions(const af::array &image, const string &w_file_path, const int p, const float psnr, const cl::Program &program_me, const cl::Program &program_custom, const string &custom_kernel_name);
	WatermarkFunctions(const string &w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const string custom_kernel_name);
	void load_W(const dim_t rows, const dim_t cols);
	void load_image(const af::array& image);
	af::array make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type);
	float mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type);
	float mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients);
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
};