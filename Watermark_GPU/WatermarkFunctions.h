#pragma once
#pragma warning(disable:4996)

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <string>
#include <arrayfire.h>
#include <functional>

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */

af::array load_W(std::string w_file, const int rows, const int cols);
af::array make_and_add_watermark_NVF(const af::array& img, const af::array& w, const int p, const float psnr, float * a, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program);
af::array make_and_add_watermark_ME(const af::array& ima, const af::array& w, af::array& a_x, const int p, const float psnr, float* a, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program);
void compute_ME_mask(const af::array& image, const af::array& padded, af::array& m_e, af::array& e_x, af::array& a_x, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program, const bool mask_needed);
void compute_ME_mask(const af::array& image, af::array& a_x, af::array& m_e, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad);
void compute_error_sequence(const af::array& image, const af::array& a_x, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad);
void compute_NVF_mask(const af::array& image, const af::array& padded, af::array& m, const int p, const int pad, const dim_t rows, const dim_t cols, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf);
float mask_detector(const af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_me);
float mask_detector(const af::array& image, const af::array& w, const int p, const float psnr, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program* program_custom_mask, const cl::Program& program_me);