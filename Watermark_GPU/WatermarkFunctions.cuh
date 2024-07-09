#pragma once
#include <functional>
#include <cuda_runtime.h>
#include <arrayfire.h>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class WatermarkFunctions {
private:
	const std::string w_file_path;
	const int p, p_squared, p_squared_minus_one, p_squared_minus_one_squared, pad;
	const float psnr;
	af::array image, w;
	dim_t rows, cols;
	cudaStream_t af_cuda_stream, custom_kernels_stream;

	float calculate_correlation(const af::array& e_u, const af::array& e_z);
	float mask_detector(const af::array& watermarked_image, const std::function<void(const af::array&, af::array&)>& compute_custom_mask);
	void compute_custom_mask(const af::array& image, af::array& m);
	void compute_prediction_error_mask(const af::array& image, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed);
	void compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& m_e, af::array& error_sequence);
	af::array make_and_add_watermark(float& a, const std::function<void(const af::array&, af::array&, af::array&)>& compute_mask);
	af::array calculate_error_sequence(const af::array& u, const af::array& coefficients);
	inline af::array compute_error_sequence(const af::array& u, const af::array& coefficients);
	void synchronize_and_cleanup_texture_data(const std::pair<cudaTextureObject_t, cudaArray*>& texture_data, const af::array& array_to_unlock);
	std::pair<cudaTextureObject_t, cudaArray*> copy_array_to_texture_data(const af::array &image, const unsigned int rows, const unsigned int cols);
public:
	WatermarkFunctions(const af::array& image, const std::string w_file_path, const int p, const float psnr);
	WatermarkFunctions(const std::string w_file_path, const int p, const float psnr);
	void load_W(const dim_t rows, const dim_t cols);
	void load_image(const af::array& image);
	af::array make_and_add_watermark_custom(float& a);
	af::array make_and_add_watermark_prediction_error(af::array& coefficients, float& a);
	float mask_detector_custom(const af::array& watermarked_image);
	float mask_detector_prediction_error(const af::array& watermarked_image);
	float mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients);
	static void display_array(const af::array& array, const int width = 1600, const int height = 900);
};