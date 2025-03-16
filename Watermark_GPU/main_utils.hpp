#pragma once
#include "Watermark.cuh"
#include <arrayfire.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <INIReader.h>
#include <stdio.h>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavcodec/codec_par.h>
}

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
void exitProgram(const int exitCode);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr);
int testForVideo(const INIReader& inir, const std::string& videoFile, const cudaDeviceProp& properties, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);
void embedWatermarkFrame(af::array& inputFrame, af::array& watermarkedFrame, const int height, const int width, const int watermarkInterval, int& framesCount, AVFrame* frame, uint8_t* frameFlatPinned, FILE* ffmpegPipe, const Watermark& watermarkObj);
void detectFrameWatermark(af::array& inputFrame, const int height, const int width, const int watermarkInterval, int& framesCount, AVFrame* frame, uint8_t* frameFlatPinned, const Watermark& watermarkObj);
void processFrames(AVFormatContext* formatCtx, AVCodecContext* decoderCtx, const int videoStreamIndex, int& framesCount, std::function<void(AVFrame*, int&)> processFrame);