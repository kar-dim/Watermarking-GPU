#include "videoprocessingcontext.hpp"
#include "Watermark.cuh"
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

VideoProcessingContext::VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
    const Watermark* watermark, const int h, const int w, const int interval, uint8_t* pinnedMem)
    : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), watermarkObj(watermark),
    height(h), width(w), watermarkInterval(interval), frameFlatPinned(pinnedMem)
{ }