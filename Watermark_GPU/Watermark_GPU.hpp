#pragma once
#include "opencl_init.h"
#include "Watermark.hpp"
#include <arrayfire.h>
#include <INIReader.h>
#include <string>
#include <vector>

void exit_program(const int exit_code);
void realtime_detection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarked_frames, const int frames, const bool display_frames, const float frame_period, const bool show_fps);
std::string execution_time(const bool show_fps, const double seconds);
int test_for_image(const cl::Device& device, const std::vector<cl::Program>& programs, const INIReader& inir, const int p, const float psnr);
int test_for_video(const cl::Device& device, const std::vector<cl::Program>& programs, const INIReader& inir, const int p, const float psnr);
