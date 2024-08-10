#include "WatermarkFunctions.cuh"
#include "kernels.cuh"
#include <fstream>
#include <arrayfire.h>
#include <iostream>
#include <string>
#include <af/cuda.h>
#include <cmath>
#include <memory>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <stdexcept>

using std::string;
using std::cout;

//constructor without specifying input image yet, it must be supplied later by calling the appropriate public method
WatermarkFunctions::WatermarkFunctions(const string &w_file_path, const int p, const float psnr)
	:p(p), p_squared(p* p), p_squared_minus_one(p_squared - 1), p_squared_minus_one_squared(p_squared_minus_one* p_squared_minus_one), pad(p / 2), psnr(psnr), w_file_path(w_file_path) {
	af_cuda_stream = afcu::getStream(afcu::getNativeId(af::getDevice()));
	cudaStreamCreate(&custom_kernels_stream);
	rows = -1;
	cols = -1;
}

WatermarkFunctions::~WatermarkFunctions()
{
	cudaStreamDestroy(custom_kernels_stream);
}

//full constructor
WatermarkFunctions::WatermarkFunctions(const af::array& image, const string &w_file_path, const int p, const float psnr)
	:WatermarkFunctions::WatermarkFunctions(w_file_path, p, psnr) {
	load_image(image);
	load_W(rows, cols);
}

//supply the input image to apply watermarking and detection
void WatermarkFunctions::load_image(const af::array& image) {
	this->image = image;
	this->rows = image.dims(0);
	this->cols = image.dims(1);
}

//helper method to load the random noise matrix W from the file specified.
void WatermarkFunctions::load_W(const dim_t rows, const dim_t cols) {
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

//helper method to copy an arrayfire cuda buffer into a cuda Texture Object Image (fast copy that happens in the device)
std::pair<cudaTextureObject_t, cudaArray*> WatermarkFunctions::copy_array_to_texture_data(const af::array & array, const unsigned int rows, const unsigned int cols) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, cols, rows);
	cudaMemcpy2DToArray(cuArray, 0, 0, array.device<float>(), cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	return std::make_pair(texObj, cuArray);
}

//helper method for cleanup and to execute common tasks after the masking kernels are executed
void WatermarkFunctions::synchronize_and_cleanup_texture_data(const std::pair<cudaTextureObject_t, cudaArray*> &texture_data, const af::array &array_to_unlock) {
	cudaDeviceSynchronize();
	cudaDestroyTextureObject(texture_data.first);
	cudaFreeArray(texture_data.second);
	array_to_unlock.unlock();
}

//compute custom mask. supports simple kernels that just apply a mask per-pixel without needing any other configuration
void WatermarkFunctions::compute_custom_mask(const af::array& image, af::array& m)
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	auto texture_data = copy_array_to_texture_data(image_transpose, rows, cols);
	float* mask_output = cuda_utils::cudaMallocPtr<float>(rows * cols);
	auto dimensions = std::make_pair(cuda_utils::grid_size_calculate(dim3(16, 16), rows, cols), dim3(16, 16));
	nvf <<<dimensions.first, dimensions.second, 0, af_cuda_stream >>> (texture_data.first, mask_output, p*p, pad, cols, rows);
	synchronize_and_cleanup_texture_data(texture_data, image_transpose);
	m = af::array(rows, cols, mask_output, afDevice);
}

//helper method to calculate the neighbors ("x_" array)
af::array WatermarkFunctions::calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad) {
	af::array array_unwrapped = af::unwrap(array, p, p, 1, 1, pad, pad, false);
	return af::join(1, array_unwrapped(af::span, af::seq(0, (p_squared / 2) - 1)), array_unwrapped(af::span, af::seq((p_squared / 2) + 1, af::end)));
}

//helper method to sum the incomplete Rx_partial and rx_partial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> WatermarkFunctions::correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols) {
	af::array Rx_partial_sums = af::moddims(Rx_partial, p_squared_minus_one_squared, (padded_cols * rows) / p_squared_minus_one_squared);
	af::array rx_partial_sums = af::moddims(rx_partial, p_squared_minus_one, (padded_cols * rows) / p_squared_minus_one);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	af::array Rx = af::moddims(af::sum(Rx_partial_sums, 1), p_squared_minus_one, p_squared_minus_one);
	af::array rx = af::sum(rx_partial_sums, 1);
	return std::make_pair(Rx, rx);
}

af::array WatermarkFunctions::make_and_add_watermark(af::array& coefficients, float& a, MASK_TYPE mask_type)
{
	af::array m, error_sequence;
	if (mask_type == MASK_TYPE::ME) {
		compute_prediction_error_mask(image, m, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	}
	else {
		compute_custom_mask(image, m);
	}
	const af::array u = m * w;
	const float divisor = std::sqrt(af::sum<float>(af::pow(u, 2)) / (image.elements()));
	a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	return image + (a * u);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed)
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const af::array image_transpose = image.T();
	const auto padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	//copy arrayfire array from device to device's texture cache and allocate Rx,rx buffers
	auto texture_data = copy_array_to_texture_data(image_transpose, rows, cols);
	float* Rx_buff = cuda_utils::cudaMallocPtr<float>(rows * padded_cols);
	float* rx_buff = cuda_utils::cudaMallocPtr<float>(rows * padded_cols);
	//call custom kernel to fill Rx and rx partial sums (in different stream than arrayfire, may help)
	auto dimensions = std::make_pair(cuda_utils::grid_size_calculate(dim3(1, 64), rows, padded_cols), dim3(1, 64));
	me_p3 <<<dimensions.first, dimensions.second, 0, custom_kernels_stream>>> (texture_data.first, Rx_buff, rx_buff, cols, padded_cols, rows);
	//calculate the neighbors "x_" array
	af::array x_ = calculate_neighbors_array(image, p, p_squared, pad);
	//wait for custom kernel to finish and release texture memory
	synchronize_and_cleanup_texture_data(texture_data, image_transpose);
	//transform the partial Rx,rx arrays by summing and changing their dimensions
	const auto correlation_arrays = correlation_arrays_transformation(af::array(padded_cols, rows, Rx_buff, afDevice), af::array(padded_cols, rows, rx_buff, afDevice), padded_cols);
	//solve the system to get coefficients and error sequence, and optionally the mask if needed
	coefficients = af::solve(correlation_arrays.first, correlation_arrays.second);
	error_sequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
	if (mask_needed) {
		const af::array error_sequence_abs = af::abs(error_sequence);
		m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
	}
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array WatermarkFunctions::calculate_error_sequence(const af::array& u, const af::array& coefficients) {
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, calculate_neighbors_array(u, p, p_squared, pad)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, const af::array& coeficcients, af::array& m_e, af::array& error_sequence)
{
	error_sequence = calculate_error_sequence(image, coeficcients);
	const af::array error_sequence_abs = af::abs(error_sequence);
	m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
}

//helper method used in detectors
float WatermarkFunctions::calculate_correlation(const af::array& e_u, const af::array& e_z) {
	float dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	float d_ez = static_cast<float>(af::norm(e_z));
	float d_eu = static_cast<float>(af::norm(e_u));
	return dot_ez_eu / (d_ez * d_eu);
}

//the main mask detector function
float WatermarkFunctions::mask_detector(const af::array& watermarked_image, MASK_TYPE mask_type)
{
	af::array m, e_z, a_z;
	if (mask_type == MASK_TYPE::NVF) {
		compute_prediction_error_mask(watermarked_image, m, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		compute_custom_mask(watermarked_image, m);
	}
	else {
		compute_prediction_error_mask(watermarked_image, m, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	}
	const af::array u = m * w;
	const af::array e_u = calculate_error_sequence(u, a_z);
	return calculate_correlation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float WatermarkFunctions::mask_detector_prediction_error_fast(const af::array& watermarked_image, const af::array& coefficients)
{
	af::array m_e, e_z, m_eu, e_u, a_u;
	compute_prediction_error_mask(watermarked_image, coefficients, m_e, e_z);
	const af::array u = m_e * w;
	compute_prediction_error_mask(u, m_eu, e_u, a_u, ME_MASK_CALCULATION_REQUIRED_NO);
	return calculate_correlation(e_u, e_z);
}

//helper method to display an af::array in a window
void WatermarkFunctions::display_array(const af::array& array, const int width, const int height) {
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}