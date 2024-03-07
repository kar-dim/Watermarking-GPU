#pragma once
#include <af/opencl.h>
#pragma warning(disable:4996)

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class WatermarkFunctions {
private:
	const cl::Context context{ afcl::getContext(true) };
	const cl::Device device{ afcl::getDeviceId() };
	const cl::CommandQueue queue{ afcl::getQueue(true) };
	const cl::Program program_me, program_custom;
	af::array image, w;
	std::string w_file_path, custom_kernel_name;
	int p, p_squared, p_squared_minus_one, p_squared_minus_one_squared, pad;
	float psnr;
	bool is_old_opencl;
	dim_t rows, cols;
	size_t max_workgroup_size;

	float calculate_correlation(const af::array& e_u, const af::array& e_z);
	float mask_detector(const af::array& watermarked_image, const std::function<void(const af::array&, af::array&)> &compute_custom_mask);
	float mask_detector(const af::array& watermarked_image, const af::array& coefficients);
	void compute_custom_mask(const af::array &image, af::array& m);
	void compute_prediction_error_mask(const af::array& image, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed);
	void compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& m_e, af::array& error_sequence);
	af::array make_and_add_watermark(float* a, const std::function<void(const af::array&, af::array&, af::array&)>& compute_mask);
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients);
	inline af::array compute_error_sequence(const af::array& u, const af::array& coefficients);
public:
	WatermarkFunctions(const af::array &image, std::string w_file_path, const int p, const float psnr, const cl::Program &program_me, const cl::Program &program_custom, const std::string custom_kernel_name);
	WatermarkFunctions(std::string w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const std::string custom_kernel_name);
	void load_W(const dim_t rows, const dim_t cols);
	void load_image(const af::array& image);
	af::array make_and_add_watermark_custom(float* a);
	af::array make_and_add_watermark_prediction_error(af::array& coefficients, float* a);
	float mask_detector_custom(const af::array& watermarked_image);
	float mask_detector_prediction_error(const af::array& watermarked_image);
	static af::array normalize_to_f32(af::array& a);
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
};