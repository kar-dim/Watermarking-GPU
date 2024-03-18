#pragma once
#pragma warning(disable:4996)

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <string>
#include <arrayfire.h>

af::array load_W(std::string w_file, const int rows, const int cols);
af::array make_and_add_watermark_NVF(af::array& img, const af::array& w, const int p, const float psnr, float * a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program);
af::array make_and_add_watermark_ME(af::array& img, const af::array& w, const int p, const float psnr, float *a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program);
af::array make_and_add_watermark_ME(af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, float* a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program);
void compute_ME_mask(af::array& image, af::array& padded, af::array& m_e, af::array& e_x, af::array& a_x, const int p, const int pad, const dim_t rows, const dim_t cols, cl::CommandQueue& queue, cl::Context& context, cl::Program& program, const bool mask_needed);
void compute_ME_mask(af::array& image, af::array& a_x, af::array& m_e, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad);
void compute_error_sequence(af::array& image, af::array& a_x, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad);
void compute_NVF_mask(af::array& image, af::array& padded, af::array& m, const int p, const int pad, const dim_t rows, const dim_t cols, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_nvf);
float mask_detector(af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_me);
float mask_detector( af::array& image, const af::array& w, const int p, const float psnr, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_nvf, cl::Program& program_me, const bool is_nvf);