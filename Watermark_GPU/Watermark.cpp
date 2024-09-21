#include "Watermark.hpp"
#include "opencl_utils.hpp"
#include <CL/opencl.hpp>
#include <arrayfire.h>
#include <af/opencl.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;

//constructor without specifying input image yet, it must be supplied later by calling the appropriate public method
Watermark::Watermark(const string &w_file_path, const int p, const float psnr, const cl::Program& prog_me, const cl::Program& prog_custom, const string custom_kernel_name)
		:program_me(prog_me), program_custom(prog_custom), w_file_path(w_file_path), custom_kernel_name(custom_kernel_name), p(p), strength_factor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	rows = -1;
	cols = -1;
}

//full constructor
Watermark::Watermark(const af::array &rgb_image, const af::array& image, const string &w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const string &custom_kernel_name)
		:Watermark::Watermark(w_file_path, p, psnr, program_me, program_custom, custom_kernel_name) 
{
	this->rgb_image = rgb_image;
	load_image(image);
	load_W(rows, cols);
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
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	//fix for OpenCL 1.2 limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (cols and rows)
	const auto pad_rows = (rows % 16 == 0) ? rows : rows + 16 - (rows % 16);
	const auto pad_cols = (cols % 16 == 0) ? cols : cols + 16 - (cols % 16);
	const af::array image_transpose = image.T();
	try {
		const cl_mem *image_buff = image_transpose.device<cl_mem>();
		const cl::Image2D image2d = cl_utils::copyBufferToImage(context, queue, image_buff, rows, cols);
		cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * rows * cols, NULL);
		cl_utils::KernelBuilder kernel_builder(program_custom, custom_kernel_name.c_str());
		queue.enqueueNDRangeKernel(kernel_builder.args(image2d, buff).build(), 
			cl::NDRange(), cl::NDRange(pad_rows, pad_cols), cl::NDRange(16, 16));
		queue.finish();
		image_transpose.unlock();
		return afcl::array(rows, cols, buff(), af::dtype::f32, true);
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error(string("ERROR in compute_nvf_mask(): " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"));
	}
}

//helper method to calculate the neighbors ("x_" array)
af::array Watermark::calculate_neighbors_array(const af::array& array, const int p, const int p_squared, const int pad) const 
{
	const af::array array_unwrapped = af::unwrap(array, p, p, 1, 1, pad, pad, false);
	return af::join(1, array_unwrapped(af::span, af::seq(0, (p_squared / 2) - 1)), array_unwrapped(af::span, af::seq((p_squared / 2) + 1, af::end)));
}

//helper method to sum the incomplete Rx_partial and rx_partial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::correlation_arrays_transformation(const af::array& Rx_partial, const af::array& rx_partial, const int padded_cols) const 
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
	const af::array mask = mask_type == MASK_TYPE:: ME ? 
		compute_prediction_error_mask(image, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES) :
		compute_custom_mask(image);
	const af::array u = mask * w;
	a = strength_factor / sqrt(af::sum<float>(af::pow(u, 2)) / (image.elements()));
	return af::clamp((type == IMAGE_TYPE::RGB ? rgb_image : image) + (u * a), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, af::array& error_sequence, af::array& coefficients, const bool mask_needed) const
{
	const unsigned int rows = static_cast<unsigned int>(image.dims(0));
	const unsigned int cols = static_cast<unsigned int>(image.dims(1));
	//fix for OpenCL 1.2 limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (cols)
	const auto padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	const af::array image_transpose = image.T();
	cl_int err;
	try {
		const cl_mem *buffer = image_transpose.device<cl_mem>();
		const cl::Image2D image2d = cl_utils::copyBufferToImage(context, queue, buffer, rows, cols);
		cl::Buffer Rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * padded_cols * rows, NULL, &err);
		cl::Buffer rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * padded_cols * rows / 8, NULL, &err);
		const cl::Buffer Rx_mappings_buff(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 64, (void *)Rx_mappings, &err);
		cl_utils::KernelBuilder kernel_builder(program_me, "me");
		queue.enqueueNDRangeKernel(
				kernel_builder.args(image2d, Rx_buff, rx_buff, Rx_mappings_buff, 
				cl::Local(sizeof(float) * 2304), cl::Local(sizeof(float) * 512), cl::Local(sizeof(float) * 64)).build(),
				cl::NDRange(), cl::NDRange(rows, padded_cols), cl::NDRange(1, 64));
		//enqueue the calculation of neighbors (x_) array before waiting "me" kernel to finish, may help a bit
		const af::array x_ = calculate_neighbors_array(image, p, p * p, p / 2);
		queue.finish(); 
		image_transpose.unlock();
		const auto correlation_arrays = correlation_arrays_transformation(afcl::array(padded_cols, rows, Rx_buff(), af::dtype::f32, true), afcl::array(padded_cols / 8, rows, rx_buff(), af::dtype::f32, true), padded_cols);
		coefficients = af::solve(correlation_arrays.first, correlation_arrays.second);
		error_sequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
		if (mask_needed) {
			const af::array error_sequence_abs = af::abs(error_sequence);
			return error_sequence_abs / af::max<float>(error_sequence_abs);
		}
		return af::array();
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error(string("ERROR in compute_me_mask(): " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"));
	}
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::calculate_error_sequence(const af::array& u, const af::array& coefficients) const 
{
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, calculate_neighbors_array(u, p, p * p, p / 2)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
af::array Watermark::compute_prediction_error_mask(const af::array& image, const af::array& coeficcients, af::array& error_sequence) const
{
	error_sequence = calculate_error_sequence(image, coeficcients);
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