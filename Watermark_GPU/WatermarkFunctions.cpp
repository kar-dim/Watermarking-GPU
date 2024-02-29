#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <arrayfire.h>
#include <af/opencl.h>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>

#undef min
#undef max
using std::cout;

af::array load_W(std::string w_file, const int rows, const int cols) {
	std::ifstream w_stream(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		std::string error_str("Error opening '" + w_file + "' file for Random noise W array");
		throw std::exception(error_str.c_str());
	}
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	int value_read_count = 0;
	while (!w_stream.eof())
		w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[value_read_count++]), sizeof(float));
	return af::transpose(af::array(cols, rows, w_ptr.get()));
}

void compute_NVF_mask(const af::array& image, const af::array& padded, af::array& m, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf)
{
	size_t padded_rows = rows + 2 * pad;
	const size_t padded_cols = cols + 2 * pad;
	const size_t elems = rows * cols;
	try {
		cl_int err = 0;
		cl_mem *padded_buff = padded.device<cl_mem>();
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { padded_cols, padded_rows, 1 };
		err = clEnqueueCopyBufferToImage(queue(), *padded_buff, padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		cl::Buffer nvf_buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * elems, NULL, &err);
		cl::Kernel kernel = cl::Kernel(program_nvf, "nvf", &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, nvf_buff);
		err = kernel.setArg(2, p);
		//fix for NVIDIA (OpenCL 1.2) limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (rows, selected arbitarily)
		//there is bound check in kernel, so it is OK.
		if (padded_cols * padded_rows % UtilityFunctions::max_workgroup_size != 0)
			padded_rows += UtilityFunctions::max_workgroup_size - (padded_rows % UtilityFunctions::max_workgroup_size);
		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, UtilityFunctions::max_workgroup_size));
		queue.finish();

		m = afcl::array(rows, cols, nvf_buff(), af::dtype::f32, true);
		padded.unlock();
	}
	catch (const std::exception & ex) {
		std::string error_str("ERROR in compute_nvf_mask(): " + std::string(ex.what()) + "\n");
		throw std::exception(error_str.c_str());
	}
}

static af::array make_and_add_watermark(const af::array& image, const af::array& w, const int p, const float psnr, float* a, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program, const std::function<void(const af::array&, const af::array&, af::array&, af::array&, float*, const int, const int, const dim_t, const dim_t, const cl::CommandQueue&, const cl::Context&, const cl::Program&)>& compute_mask)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;
	//padding
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0, padded_cols, padded_rows);
	padded(af::seq(static_cast<double>(pad), static_cast<double>(padded_cols - pad - 1)), af::seq(static_cast<double>(pad), static_cast<double>(padded_rows - pad - 1))) = image.T();

	af::array m, e_x;
	compute_mask(image, padded, m, e_x, a, p, pad, rows, cols, queue, context, program);

	af::array u = m * w;
	float divv = std::sqrt(af::sum<float>(af::pow(u, 2)) / elems);
	*a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;
	return image + (*a * u);
}

af::array make_and_add_watermark_NVF(const af::array& image, const af::array& w, const int p, const float psnr, float* a, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf)
{
	return make_and_add_watermark(image, w, p, psnr, a, queue, context, program_nvf, program_nvf, [&](const af::array& image, const af::array& padded, af::array& m, af::array& e_x, float* a, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf) {
		compute_NVF_mask(image, padded, m, p, pad, rows, cols, queue, context, program_nvf);
	});
}

af::array make_and_add_watermark_ME(const af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, float* a, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_me)
{
	return make_and_add_watermark(image, w, p, psnr, a, queue, context, program_me, program_me, [&](const af::array& image, const af::array& padded, af::array& m, af::array& e_x, float* a, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_me) {
		compute_ME_mask(image, padded, m, e_x, a_x, p, pad, rows, cols, queue, context, program_me, true);
	});
}


//Compute ME mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
void compute_ME_mask(const af::array& image, const af::array& padded, af::array& m_e, af::array& e_x, af::array& a_x, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program, const bool mask_needed)
{
	const auto elems = rows * cols;
	//padded is Transposed
	auto padded_rows = padded.dims(1);
	auto padded_cols = padded.dims(0);
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const int p_squared_1 = p_squared - 1;

	cl_int err = 0;
	try {
		cl_mem *padded_buffer = padded.device<cl_mem>();
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { static_cast<size_t>(padded_cols), static_cast<size_t>(padded_rows), 1 };

		err = clEnqueueCopyBufferToImage(queue(), *padded_buffer, padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		cl::Buffer neighb_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_1), NULL, &err);
		cl::Buffer Rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_1) * (p_squared_1), NULL, &err);
		cl::Buffer rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems * (p_squared_1), NULL, &err);

		cl::Kernel kernel = cl::Kernel(program, "me", &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, Rx_buff);
		err = kernel.setArg(2, rx_buff);
		err = kernel.setArg(3, neighb_buff);
		//err = kernel.setArg(4, p);

		//fix for NVIDIA (OpenCL 1.2) limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (rows, selected arbitarily)
		//there is bound check in kernel, so it is OK.
		if (padded_cols * padded_rows % UtilityFunctions::max_workgroup_size != 0)
			padded_rows += UtilityFunctions::max_workgroup_size - (padded_rows % UtilityFunctions::max_workgroup_size);
		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, UtilityFunctions::max_workgroup_size));
		queue.finish();

		af::array Rx_all = afcl::array(rows * p_squared_1 * p_squared_1, cols, Rx_buff(), af::dtype::f32, true);
		af::array rx_all = afcl::array(rows * p_squared_1, cols, rx_buff(), af::dtype::f32, true);
		af::array x_ = af::moddims(afcl::array(rows * p_squared_1, cols, neighb_buff(), af::dtype::f32, true), p_squared_1, elems);

		//reduction sum of blocks
		//all [p^2-1,1] blocks will be summed in rx
		//all [p^2-1, p^2-1] blocks will be summed in Rx
		af::array rx = af::sum(af::moddims(rx_all, p_squared_1, elems), 1);
		af::array Rx = af::moddims(af::sum(af::moddims(Rx_all, p_squared_1 * p_squared_1, elems), 1), p_squared_1, p_squared_1);

		a_x = af::solve(Rx, rx);
		e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);

		if (mask_needed) {
			af::array e_x_abs = af::abs(e_x);
			m_e = e_x_abs / af::max<float>(e_x_abs);
		}
		padded.unlock();
	}
	catch (const std::exception &ex) {
		std::string error_str("ERROR in compute_me_mask(): " + std::string(ex.what()) + "\n");
		throw std::exception(error_str.c_str());
	}
}

//overloaded, fast mask calculation by using a supplied prediction filter
void compute_ME_mask(const af::array& image, const af::array& a_x, af::array& m_e, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad)
{
	const int p_squared = static_cast<int>(std::pow(p, 2));
	af::array padded_image_all = af::moddims(af::unwrap(image, p, p, 1, 1, pad, pad, true), p_squared, rows * cols);
	af::array x_ = af::join(0, padded_image_all.rows(0, (p_squared / 2) - 1), padded_image_all.rows((p_squared / 2) + 1, af::end));
	e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);
	af::array e_x_abs = af::abs(e_x);
	m_e = e_x_abs / af::max<float>(e_x_abs);
}
//overloaded, fast prediction error sequence calculation by using a supplied prediction filter
void compute_error_sequence(const af::array& image, const af::array& a_x, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad)
{
	const int p_squared = static_cast<int>(std::pow(p, 2));
	af::array padded_image_all = af::moddims(af::unwrap(image, p, p, 1, 1, pad, pad, true), p_squared, rows * cols);
	af::array x_ = af::join(0, padded_image_all.rows(0, (p_squared / 2) - 1), padded_image_all.rows((p_squared / 2) + 1, af::end));
	e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);
}

//the main mask detector function
float mask_detector(const af::array& image, const af::array& w, const int p, const float psnr, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program* program_custom_mask, const cl::Program& program_me)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	
	af::array m, e_z, a_z;
	af::array padded = af::constant(0.0f, padded_cols, padded_rows);
	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();
	if (program_custom_mask != nullptr) {
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, queue, context, program_me, false);
		compute_NVF_mask(image, padded, m, p, pad, rows, cols, queue, context, *program_custom_mask);
	}
	else {
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, queue, context, program_me, true);
	}
	af::array u = m * w;

	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = u.T();
	af::array e_u;
	compute_error_sequence(u, a_z, e_u, rows, cols, p, pad);

	float dot_ez_eu, d_ez, d_eu;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));

	return dot_ez_eu / (d_ez * d_eu);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (a_x, x_ are supplied)
float mask_detector(const af::array& image, const af::array& w, const af::array& a_x, const int p, const float psnr, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_me)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0.0f, padded_cols, padded_rows);
	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();

	af::array m_e, e_z;
	compute_ME_mask(image, a_x, m_e, e_z, rows, cols, p, pad);
	af::array u = m_e * w;

	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = u.T();

	af::array m_eu, e_u, a_u;
	compute_ME_mask(u, padded, m_eu, e_u, a_u, p, pad, rows, cols, queue, context, program_me, false);

	float dot_ez_eu, d_ez, d_eu;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() needs vectors, so we flatten the arrays
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));

	return dot_ez_eu / (d_ez * d_eu);
}