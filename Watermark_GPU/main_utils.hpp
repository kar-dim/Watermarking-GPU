#pragma once
#include "opencl_init.h"
#include "Watermark.hpp"
#include <arrayfire.h>
#include <INIReader.h>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

void exitProgram(const int exitCode);
void realtimeDetection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarked_frames, const int frames, const bool displayFrames, const float framePeriod, const bool showFps);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const cl::Device& device, const std::vector<cl::Program>& programs, const INIReader& inir, const int p, const float psnr);
int testForVideo(const std::vector<cl::Program>& programs, const std::string& videoFile, const INIReader& inir, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);
