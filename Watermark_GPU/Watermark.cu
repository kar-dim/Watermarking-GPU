#include "Watermark.cuh"
#include "cuda_utils.hpp"
#include "kernels.cuh"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;

//constructor without specifying input image yet, it must be supplied later by calling the appropriate public method
Watermark::Watermark(const string &w_file_path, const int p, const float psnr)
	:w_file_path(w_file_path), p(p), strength_factor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	rows = -1;
	cols = -1;
	af_cuda_stream = afcu::getStream(afcu::getNativeId(af::getDevice()));
	cudaStreamCreate(&custom_kernels_stream);

}

//full constructor
Watermark::Watermark(const af::array &rgb_image, const af::array& image, const string &w_file_path, const int p, const float psnr)
	:Watermark::Watermark(w_file_path, p, psnr) 
{
	this->rgb_image = rgb_image;
	load_image(image);
	load_W(rows, cols);
}

//destructor, only custom kernels cuda stream must be destroyed
Watermark::~Watermark()
{
	cudaStreamDestroy(custom_kernels_stream);
}

//supply the input image to apply watermarking and detection
void Watermark::load_image(const af::array& image) 
{
	this->image = image;
	rows = image.dims(0);
	cols = image.dims(1);
}

//helper method to load the random noise matrix W from the file specified.
void Watermark::load_W(const dim_t rows, const dim_t cols) 
{
	std::ifstream w_stream(w_file_path.c_str(), std::ios::binary);
	if (!w_stream.is_open())
		throw std::runtime_error(string("Error opening '" + w_file_path + "' file for Random noise W array\n"));
	w_stream.seekg(0, std::ios::end);
	const auto total_bytes = w_stream.tellg();
	w_stream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + ", Image width: " + std::to_string(cols) + ", Image height: " + std::to_string(rows) + "\n"));
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[0]), total_bytes);
	this->w = af::transpose(af::array(cols, rows, w_ptr.get()));
}

//compute custom mask. supports simple kernels that just apply a mask per-pixel without needing any other configuration
af::array Watermark::compute_custom_mask(const af::array& image) const
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	const auto texture_data = cuda_utils::copyArrayToTexture(image_transpose.device<float>(), rows, cols);
	float* mask_output = cuda_utils::cudaMallocPtr(rows * cols);
	const auto dimensions = std::make_pair(cuda_utils::gridSizeCalculate(dim3(16, 16), rows, cols), dim3(16, 16));
	switch (p) {
		case 3: nvf<3> <<<dimensions.first, dimensions.second, 0, custom_kernels_stream >>> (texture_data.first, mask_output, cols, rows); break;
		case 5: nvf<5> <<<dimensions.first, dimensions.second, 0, custom_kernels_stream >>> (texture_data.first, mask_output, cols, rows); break;
		case 7: nvf<7> <<<dimensions.first, dimensions.second, 0, custom_kernels_stream >>> (texture_data.first, mask_output, cols, rows); break;
		case 9: nvf<9> <<<dimensions.first, dimensions.second, 0, custom_kernels_stream >>> (texture_data.first, mask_output, cols, rows); break;
	}
	cuda_utils::synchronizeAndCleanupTexture(custom_kernels_stream, texture_data);
	image_transpose.unlock();
	return af::array(rows, cols, mask_output, afDevice);
}

//helper method to calculate the neighbors ("x_" array)
af::array Watermark::calculate_neighbors_array(const af::array& array) const 
{
	const int center = (p * p) / 2;
	af::array unwrapped = af::unwrap(array, p, p, 1, 1, p / 2, p / 2, false);
	return af::join(1, unwrapped(af::span, af::seq(0, center - 1)), unwrapped(af::span, af::seq(center + 1, af::end)));
}

//helper method to sum the incomplete Rx_partial and rx_partial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int rows, const int padded_cols) const
{
	const int p_sq_minus_one = (p * p) - 1;
	const int p_sq_minus_one_sq = p_sq_minus_one * p_sq_minus_one;
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	const af::array Rx = af::moddims(af::sum(af::moddims(Rx_partial, p_sq_minus_one_sq, (padded_cols * rows) / p_sq_minus_one_sq), 1), p_sq_minus_one, p_sq_minus_one);
	const af::array rx = af::sum(af::moddims(rx_partial, p_sq_minus_one, (padded_cols * rows) / (8 * p_sq_minus_one)), 1);
	return std::make_pair(Rx, rx);
}

//Main watermark embedding method
af::array Watermark::make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type, IMAGE_TYPE type) const
{
	af::array error_sequence;
	const af::array mask = mask_type == MASK_TYPE::ME ?
		compute_prediction_error_mask(image, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES) :
		compute_custom_mask(image);
	const af::array u = mask * w;
	a = strength_factor / sqrt(af::sum<float>(af::pow(u, 2)) / image.elements());
	return af::clamp((type == IMAGE_TYPE::RGB ? rgb_image : image) + (u * a), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed) const
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	const auto padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	//enqueue "x_" kernel (which is heavy)
	const af::array x_ = calculate_neighbors_array(image);
	//initialize custom kernel memory
	float* Rx_buff = cuda_utils::cudaMallocPtr(rows * padded_cols);
	float* rx_buff = cuda_utils::cudaMallocPtr(rows * padded_cols / 8);
	//do a texture copy (for custom kernel)
	const auto texture_data = cuda_utils::copyArrayToTexture(image_transpose.device<float>(), rows, cols);
	const auto dimensions = std::make_pair(cuda_utils::gridSizeCalculate(dim3(1, 64), rows, padded_cols), dim3(1, 64));
	me_p3 <<<dimensions.first, dimensions.second, 0, custom_kernels_stream>>> (texture_data.first, Rx_buff, rx_buff, cols, padded_cols, rows);
	//cleanup and calculation of coefficients, error sequence and mask
	cuda_utils::synchronizeAndCleanupTexture(custom_kernels_stream, texture_data);
	image_transpose.unlock();
	const auto correlation_arrays = correlation_arrays_transformation(af::array(padded_cols, rows, Rx_buff, afDevice), af::array(padded_cols / 8, rows, rx_buff, afDevice), rows, padded_cols);
	coefficients = af::solve(correlation_arrays.first, correlation_arrays.second);
	error_sequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
	if (mask_needed) {
		const af::array error_sequence_abs = af::abs(error_sequence);
		return error_sequence_abs / af::max<float>(error_sequence_abs);
	}
	return af::array();
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::calculate_error_sequence(const af::array& u, const af::array& coefficients) const 
{
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, calculate_neighbors_array(u)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, const af::array& coefficients, af::array& error_sequence) const
{
	error_sequence = calculate_error_sequence(image, coefficients);
	const af::array error_sequence_abs = af::abs(error_sequence);
	return error_sequence_abs / af::max<float>(error_sequence_abs);
}

//helper method used in detectors
float Watermark::calculate_correlation(const af::array& e_u, const af::array& e_z) const 
{
	return af::dot<float>(af::flat(e_u), af::flat(e_z)) / static_cast<float>(af::norm(e_z) * af::norm(e_u));
}

//the main mask detector function
float Watermark::mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type) const
{
	af::array mask, e_z, a_z;
	if (mask_type == MASK_TYPE::NVF) {
		compute_prediction_error_mask(watermarked_image, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = compute_custom_mask(watermarked_image);
	}
	else
		mask = compute_prediction_error_mask(watermarked_image, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	const af::array u = mask * w;
	const af::array e_u = calculate_error_sequence(u, a_z);
	return calculate_correlation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float Watermark::mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients) const
{
	af::array e_z, e_u, a_u;
	const af::array m_e = compute_prediction_error_mask(watermarked_image, coefficients, e_z);
	const af::array u = m_e * w;
	compute_prediction_error_mask(u, e_u, a_u, ME_MASK_CALCULATION_REQUIRED_NO);
	return calculate_correlation(e_u, e_z);
}

//helper method to display an af::array in a window
void Watermark::display_array(const af::array& array, const int width, const int height) 
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}