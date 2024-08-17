#pragma once
#include "INIReader.h"
#include "Watermark.cuh"
#include <arrayfire.h>
#include <driver_types.h>
#include <string>
#include <vector>

void exit_program(const int exit_code);
void realtime_detection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarked_frames, const int frames, const bool display_frames, const float frame_period, const bool show_fps);
std::string execution_time(bool show_fps, double seconds);
int test_for_image(const INIReader& inir, cudaDeviceProp& properties, const int p, const float psnr);
int test_for_video(const INIReader& inir, cudaDeviceProp& properties, const int p, const float psnr);