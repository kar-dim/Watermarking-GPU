#pragma once
#include "Watermark.cuh"
#include <arrayfire.h>
#include <driver_types.h>
#include <INIReader.h>
#include <string>
#include <vector>

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
void exitProgram(const int exitCode);
void realtimeDetection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarkedFrames, const int frames, const bool displayFrames, const float framePeriod, const bool showFps);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr);
int testForVideo(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr);