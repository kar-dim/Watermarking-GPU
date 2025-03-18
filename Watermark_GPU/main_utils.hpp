#pragma once
#include "opencl_init.h"
#include "videoprocessingcontext.hpp"
#include <functional>
#include <INIReader.h>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavcodec/codec_par.h>

}

void exitProgram(const int exitCode);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const cl::Device& device, const std::vector<cl::Program>& programs, const INIReader& inir, const int p, const float psnr);
int testForVideo(const std::vector<cl::Program>& programs, const std::string& videoFile, const INIReader& inir, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);
void embedWatermarkFrame(const VideoProcessingContext& data, int& framesCount, AVFrame* frame, FILE* ffmpegPipe);
void detectFrameWatermark(const VideoProcessingContext& data, int& framesCount, AVFrame* frame);
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame);