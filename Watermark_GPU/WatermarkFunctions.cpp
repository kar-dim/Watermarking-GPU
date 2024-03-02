#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "WatermarkFunctions.h"
#include <fstream>
#include <arrayfire.h>
#include <iosfwd>
#include <iostream>
#include <string>
#include <af/opencl.h>
#include <cmath>
#include <memory>
#include <functional>

using std::cout;

WatermarkFunctions::WatermarkFunctions(std::string w_file_path, const int p, const float psnr, const cl::Program& prog_me, const cl::Program& prog_custom, const std::string custom_kernel_name)
		:program_me(prog_me), program_custom(prog_custom) {
	this->p = p;
	this->p_squared = static_cast<int>(std::pow(p, 2));
	this->p_squared_minus_one = p_squared - 1;
	this->pad = p / 2;
	this->psnr = psnr;
	this->w_file_path = w_file_path;
	this->custom_kernel_name = custom_kernel_name;
	device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_workgroup_size);
	if (max_workgroup_size > 256)
		max_workgroup_size = 256;
	this->rows = -1;
	this->cols = -1;
}

WatermarkFunctions::WatermarkFunctions(const af::array& image, std::string w_file_path, const int p, const float psnr, const cl::Program& program_me, const cl::Program& program_custom, const std::string custom_kernel_name)
		:WatermarkFunctions::WatermarkFunctions(w_file_path, p, psnr, program_me, program_custom, custom_kernel_name) {
	load_image(image);
	load_W(this->rows, this->cols);
}

void WatermarkFunctions::load_image(const af::array& image) {
	this->image = image;
	this->rows = image.dims(0);
	this->cols = image.dims(1);
}

void WatermarkFunctions::load_W(const dim_t rows, const dim_t cols) {
	std::ifstream w_stream(this->w_file_path.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		std::string error_str("Error opening '" + this->w_file_path + "' file for Random noise W array");
		cout << error_str;
		throw std::exception(error_str.c_str());
	}
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	int value_read_count = 0;
	while (!w_stream.eof())
		w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[value_read_count++]), sizeof(float));
	this->w = af::transpose(af::array(cols, rows, w_ptr.get()));
}

void WatermarkFunctions::compute_custom_mask(const af::array& image, const af::array& padded, af::array& m)
{
	size_t padded_rows = image.dims(0) + 2 * pad;
	const size_t padded_cols = image.dims(1) + 2 * pad;
	try {
		cl_int err = 0;
		cl_mem *padded_buff = padded.device<cl_mem>();
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { padded_cols, padded_rows, 1 };
		err = clEnqueueCopyBufferToImage(queue(), *padded_buff, padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * image.dims(0) * image.dims(1), NULL, &err);
		cl::Kernel kernel = cl::Kernel(program_custom, custom_kernel_name.c_str(), &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, buff);
		err = kernel.setArg(2, p);
		//fix for NVIDIA (OpenCL 1.2) limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (rows, selected arbitarily)
		//there is bound check in kernel, so it is OK.
		if (padded_cols * padded_rows % max_workgroup_size != 0)
			padded_rows += max_workgroup_size - (padded_rows % max_workgroup_size);
		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, max_workgroup_size));
		queue.finish();
		m = afcl::array(image.dims(0), image.dims(1), buff(), af::dtype::f32, true);
		padded.unlock();
	}
	catch (const std::exception & ex) {
		std::string error_str("ERROR in compute_nvf_mask(): " + std::string(ex.what()) + "\n");
		cout << error_str;
		throw std::exception(error_str.c_str());
	}
}

af::array WatermarkFunctions::make_and_add_watermark(float* a, const std::function<void(const af::array&, const af::array&, af::array&, af::array&, float*)>& compute_mask)
{
	af::array padded = af::constant(0, image.dims(1) + 2 * pad, image.dims(0) + 2 * pad);
	padded(af::seq(pad, static_cast<double>(image.dims(1) + pad - 1)), af::seq(pad, static_cast<double>(image.dims(0) + pad - 1))) = image.T();
	af::array m, error_sequence;
	compute_mask(image, padded, m, error_sequence, a);
	af::array u = m * w;
	float divisor = std::sqrt(af::sum<float>(af::pow(u, 2)) / (image.dims(0) * image.dims(1)));
	*a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	return image + (*a * u);
}

af::array WatermarkFunctions::make_and_add_watermark_custom(float* a)
{
	return make_and_add_watermark(a, [&](const af::array& image, const af::array& padded, af::array& m, af::array& error_sequence, float* a) {
		compute_custom_mask(image, padded, m);
	});
}

af::array WatermarkFunctions::make_and_add_watermark_prediction_error(af::array& coefficients, float* a)
{
	return make_and_add_watermark(a, [&](const af::array& image, const af::array& padded, af::array& m, af::array& error_sequence, float* a) {
		compute_prediction_error_mask(image, padded, m, error_sequence, coefficients, true);
	});
}

//Compute ME mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, const af::array& padded, af::array& m_e, af::array& error_sequence, af::array& coefficients, const bool mask_needed)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;
	//padded is Transposed
	auto padded_rows = padded.dims(1);
	auto padded_cols = padded.dims(0);

	cl_int err = 0;
	try {
		cl_mem *padded_buffer = padded.device<cl_mem>();
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { static_cast<size_t>(padded_cols), static_cast<size_t>(padded_rows), 1 };
		err = clEnqueueCopyBufferToImage(queue(), *padded_buffer, padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		cl::Buffer neighb_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_minus_one), NULL, &err);
		cl::Buffer Rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_minus_one) * (p_squared_minus_one), NULL, &err);
		cl::Buffer rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_minus_one), NULL, &err);
		cl::Kernel kernel = cl::Kernel(program_me, "me", &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, Rx_buff);
		err = kernel.setArg(2, rx_buff);
		err = kernel.setArg(3, neighb_buff);
		//err = kernel.setArg(4, cl::Local(sizeof(float) * 64));
		//err = kernel.setArg(4, p);
		//fix for NVIDIA (OpenCL 1.2) limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (rows, selected arbitarily)
		//there is bound check in kernel, so it is OK.
		if (padded_cols * padded_rows % max_workgroup_size != 0)
			padded_rows += max_workgroup_size - (padded_rows % max_workgroup_size);
		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, max_workgroup_size));
		queue.finish();
		af::array Rx_all = afcl::array(rows * p_squared_minus_one * p_squared_minus_one, cols, Rx_buff(), af::dtype::f32, true);
		af::array rx_all = afcl::array(rows * p_squared_minus_one, cols, rx_buff(), af::dtype::f32, true);
		af::array x_ = af::moddims(afcl::array(rows * p_squared_minus_one, cols, neighb_buff(), af::dtype::f32, true), p_squared_minus_one, elems);
		//reduction sum of blocks
		//all [p^2-1,1] blocks will be summed in rx
		//all [p^2-1, p^2-1] blocks will be summed in Rx
		af::array rx = af::sum(af::moddims(rx_all, p_squared_minus_one, elems), 1);
		af::array Rx = af::moddims(af::sum(af::moddims(Rx_all, p_squared_minus_one * p_squared_minus_one, elems), 1), p_squared_minus_one, p_squared_minus_one);
		coefficients = af::solve(Rx, rx);
		error_sequence = af::moddims(af::flat(image).T() - af::matmul(coefficients, x_, AF_MAT_TRANS), rows, cols);
		if (mask_needed) {
			af::array error_sequence_abs = af::abs(error_sequence);
			m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
		}
		padded.unlock();
	}
	catch (const std::exception &ex) {
		std::string error_str("ERROR in compute_me_mask(): " + std::string(ex.what()) + "\n");
		cout << error_str;
		throw std::exception(error_str.c_str());
	}
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array WatermarkFunctions::calculate_error_sequence(const af::array& u, const af::array& coefficients) {
	af::array padded_image_all = af::moddims(af::unwrap(u, p, p, 1, 1, pad, pad, true), p_squared, u.dims(0) * u.dims(1));
	af::array x_ = af::join(0, padded_image_all.rows(0, (p_squared / 2) - 1), padded_image_all.rows((p_squared / 2) + 1, af::end));
	return af::moddims(af::flat(u).T() - af::matmul(coefficients, x_, AF_MAT_TRANS), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
void WatermarkFunctions::compute_prediction_error_mask(const af::array& image, const af::array& coeficcients, af::array& m_e, af::array& error_sequence)
{
	error_sequence = calculate_error_sequence(image, coeficcients);
	af::array error_sequence_abs = af::abs(error_sequence);
	m_e = error_sequence_abs / af::max<float>(error_sequence_abs);
}

//fast prediction error sequence calculation by using a supplied prediction filter (calls helper method)
af::array WatermarkFunctions::compute_error_sequence(const af::array& u, const af::array& coefficients)
{
	return calculate_error_sequence(u, coefficients);
}

//helper method used in detectors
float WatermarkFunctions::calculate_correlation(const af::array& e_u, const af::array& e_z) {
	float dot_ez_eu, d_ez, d_eu;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));
	return dot_ez_eu / (d_ez * d_eu);
}

//the main mask detector function
float WatermarkFunctions::mask_detector(const af::array& image, const std::function<void(const af::array&, const af::array&, af::array&)>& compute_custom_mask)
{
	//padding
	const auto nopadded_region_cols = af::seq(pad, static_cast<double>(image.dims(1) + pad - 1));
	const auto nopadded_region_rows = af::seq(pad, static_cast<double>(image.dims(0) + pad - 1));
	af::array m, e_z, a_z;
	af::array padded = af::constant(0.0f, image.dims(1) + 2 * pad, image.dims(0) + 2 * pad);
	padded(nopadded_region_cols, nopadded_region_rows) = image.T();
	if (compute_custom_mask != nullptr) {
		compute_prediction_error_mask(image, padded, m, e_z, a_z, false);
		compute_custom_mask(image, padded, m);
	}
	else {
		compute_prediction_error_mask(image, padded, m, e_z, a_z, true);
	}
	af::array u = m * w;
	padded(nopadded_region_cols, nopadded_region_rows) = u.T();
	af::array e_u = compute_error_sequence(u, a_z);
	return calculate_correlation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float WatermarkFunctions::mask_detector(const af::array& watermarked_image, const af::array& coefficients)
{
	//padding
	const auto nopadded_region_cols = af::seq(pad, static_cast<double>(watermarked_image.dims(1) + pad - 1));
	const auto nopadded_region_rows = af::seq(pad, static_cast<double>(watermarked_image.dims(0) + pad - 1));
	af::array padded = af::constant(0.0f, watermarked_image.dims(1) + 2 * pad, watermarked_image.dims(0) + 2 * pad);
	padded(nopadded_region_cols, nopadded_region_rows) = watermarked_image.T();
	af::array m_e, e_z, m_eu, e_u, a_u;
	compute_prediction_error_mask(watermarked_image, coefficients, m_e, e_z);
	af::array u = m_e * w;
	padded(nopadded_region_cols, nopadded_region_rows) = u.T();
	compute_prediction_error_mask(u, padded, m_eu, e_u, a_u, false);
	return calculate_correlation(e_u, e_z);
}

float WatermarkFunctions::mask_detector_custom(const af::array& watermarked_image) {
	return mask_detector(watermarked_image, [&](const af::array& watermarked_image, const af::array& padded, af::array& m){
		compute_custom_mask(watermarked_image, padded, m);
	});
}

float WatermarkFunctions::mask_detector_prediction_error(const af::array& watermarked_image) {
	return mask_detector(watermarked_image, nullptr);
}