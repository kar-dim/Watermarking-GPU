#include "videoprocessingcontext.hpp"
#include "opencl_init.h"
#include "Watermark.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

VideoProcessingContext::VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
    const Watermark* watermark, const int h, const int w, const int interval, cl_uchar* pinnedMem)
    : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), watermarkObj(watermark),
    height(h), width(w), watermarkInterval(interval), frameFlatPinned(pinnedMem)
{
}