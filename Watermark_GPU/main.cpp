﻿#include "kernels/me_p3.hpp"
#include "kernels/nvf.hpp"
#include "kernels/scaled_neighbors_p3.hpp"
#include "main_utils.hpp"
#include "opencl_init.h"
#include "Utilities.hpp"
#include "videoprocessingcontext.hpp"
#include "Watermark.hpp"
#include <af/opencl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <INIReader.h>
#include <iostream>
#include <memory>
//#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
#include <libavutil/avutil.h>
#include <libavcodec/codec.h>
#include <libavutil/pixfmt.h>
#include "libavcodec/codec_par.h"
#include "libavutil/rational.h"
}

using std::cout;
using std::string;
using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, std::function<void(AVFormatContext*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;
using PinnedMemoryPtr = std::unique_ptr<cl_uchar, std::function<void(cl_uchar*)>>;

//helper lambda function that displays an error message and exits the program if an error condition is true
auto checkError = [](auto criticalErrorCondition, const string& errorMessage) 
{
	if (criticalErrorCondition) 
	{
		cout << errorMessage << "\n";
		exitProgram(EXIT_FAILURE);
	}
};

/*!
 *  \brief  This is a project implementation of my Thesis with title: 
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(void)
{
	//open parameters file
	const INIReader inir("settings.ini");
	checkError(inir.ParseError() < 0, "Could not load settings.ini file");

	//omp_set_num_threads(omp_get_max_threads());
//#pragma omp parallel for
	//for (int i = 0; i < 24; i++) { }

	try {
		af::setDevice(inir.GetInteger("options", "opencl_device", 0));
	}
	catch (const std::exception&) {
		cout << "NOTE: Invalid OpenCL device specified, using default 0" << "\n";
		af::setDevice(0);
	}
	af::info();
	cout << "\n";

	cl::Context context(afcl::getContext(false));
	cl::Device device(afcl::getDeviceId(), false);

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = inir.GetFloat("parameters", "psnr", -1.0f);

	//TODO for p>3 we have problems with ME masking buffers
	checkError(p != 3, "For now, only p=3 is allowed");

	/*if (p != 3 && p != 5 && p != 7 && p != 9) {
		cout << "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9\n";
		exitProgram(EXIT_FAILURE);
	}*/

	checkError(psnr <= 0, "PSNR must be a positive number");

	//compile opencl kernels
	std::vector<cl::Program> programs(3);
	try {
		auto buildProgram = [&context, &device](auto& program, const string& kernelName, const string& buildOptions) 
		{
			program = cl::Program(context, kernelName);
			program.build(device, buildOptions.c_str());
		};
		buildProgram(programs[0], nvf, std::format("-cl-mad-enable -Dp={}", p));
		buildProgram(programs[1], me_p3, "-cl-mad-enable");
		buildProgram(programs[2], scaled_neighbors_p3, "-cl-mad-enable");
	}
	catch (const cl::Error& e) {
		cout << "Could not build a kernel, Reason: " << e.what() << "\n\n";
		for (const cl::Program& program : programs) 
		{
			if (program.get() != NULL && program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS)
				cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		}
		exitProgram(EXIT_FAILURE);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}

	//test algorithms
	try {
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ?
			testForVideo(programs, videoFile, inir, p, psnr) :
			testForImage(device, programs, inir, p, psnr);
		exitProgram(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

//embed watermark for static images
int testForImage(const cl::Device& device, const std::vector<cl::Program>& programs, const INIReader& inir, const int p, const float psnr)
{
	constexpr float rPercent = 0.299f;
	constexpr float gPercent = 0.587f;
	constexpr float bPercent = 0.114f;
	const string imageFile = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";

	//load image from disk into an arrayfire array
	timer::start();
	const af::array rgbImage = af::loadImage(imageFile.c_str(), true);
	const af::array image = af::rgb2gray(rgbImage, rPercent, gPercent, bPercent);
	af::sync();
	timer::end();
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	cout << "Time to load and transfer RGB image from disk to VRAM: " << timer::elapsedSeconds() << "\n\n";

	checkError(cols < 64 || rows < 64, "Image dimensions too low");
	checkError(cols > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) || rows > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()), "Image dimensions too high for this GPU");

	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	Watermark watermarkObj(rows, cols, inir.Get("paths", "watermark", ""), p, psnr, programs);

	float watermarkStrength;
	//warmup for arrayfire
	watermarkObj.makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::NVF);
	watermarkObj.makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::ME);

	double secs = 0;
	//make NVF watermark
	af::array watermarkNVF, watermarkME;
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		watermarkNVF = watermarkObj.makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::NVF);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));

	//make ME watermark
	secs = 0;
	//Prediction error mask calculation
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		watermarkME = watermarkObj.makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));

	const af::array watermarkedNVFgray = af::rgb2gray(watermarkNVF, rPercent, gPercent, bPercent);
	const af::array watermarkedMEgray = af::rgb2gray(watermarkME, rPercent, gPercent, bPercent);
	//warmup for arrayfire
	watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
	watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);

	float correlationNvf, correlationMe;
	secs = 0;
	//NVF mask detection
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		correlationNvf = watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));

	secs = 0;
	//Prediction error mask detection
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		correlationMe = watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
		timer::end();
		secs += timer::elapsedSeconds();
	}
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));
	
	cout << std::format("Correlation [NVF]: {:.16f}\n", correlationNvf);
	cout << std::format("Correlation [ME]: {:.16f}\n", correlationMe);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
	{
		cout << "\nSaving watermarked files to disk...\n";
//#pragma omp parallel sections
		//{
//#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_NVF").c_str(), watermarkNVF.as(u8));
//#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_ME").c_str(), watermarkME.as(u8));
		//}
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const std::vector<cl::Program>& programs, const string& videoFile, const INIReader& inir, const int p, const float psnr)
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = inir.GetInteger("parameters_video", "watermark_interval", 30);

	//Set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//Load input video
	AVFormatContext* rawInputCtx = nullptr;
	checkError(avformat_open_input(&rawInputCtx, videoFile.c_str(), nullptr, nullptr) < 0, "ERROR: Failed to open input video file");
	AVFormatContextPtr inputFormatCtx(rawInputCtx, [](AVFormatContext* ctx) { if (ctx) { avformat_close_input(&ctx); } });
	avformat_find_stream_info(inputFormatCtx.get(), nullptr);
	av_dump_format(inputFormatCtx.get(), 0, videoFile.c_str(), 0);

	//Find video stream and open video decoder
	const int videoStreamIndex = findVideoStreamIndex(inputFormatCtx.get());
	checkError(videoStreamIndex == -1, "ERROR: No video stream found");
	const AVCodecContextPtr inputDecoderCtx(openDecoderContext(inputFormatCtx->streams[videoStreamIndex]->codecpar), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });

	//initialize watermark functions class
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	const Watermark watermarkObj(height, width, inir.Get("paths", "watermark", ""), p, psnr, programs);

	//initialize host pinned memory for fast GPU<->CPU transfers
	const cl::Context context(afcl::getContext(false));
	const cl::CommandQueue queue(afcl::getQueue(false));
	cl::Buffer pinnedBuff(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, width * height * sizeof(cl_uchar), nullptr, nullptr);
	cl_uchar* frameFlatPinned = static_cast<cl_uchar*>(queue.enqueueMapBuffer(pinnedBuff, CL_TRUE, CL_MAP_WRITE, 0, width * height * sizeof(cl_uchar), nullptr, nullptr, nullptr));
	PinnedMemoryPtr framePinned(frameFlatPinned, [&](cl_uchar* ptr) { queue.enqueueUnmapMemObject(pinnedBuff, ptr); });
	
	//group common video data for both embedding and detection
	const VideoProcessingContext videoData(inputFormatCtx.get(), inputDecoderCtx.get(), videoStreamIndex, &watermarkObj, height, width, watermarkInterval, framePinned.get());

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");
		// Build the FFmpeg command
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt yuv420p " << "-s " << width << "x" << height
			<< " -r " << getVideoFrameRate(inputFormatCtx.get(), videoStreamIndex) << " -i - -i " << videoFile << " " << ffmpegOptions
			<< " -c:s copy -c:a copy -map 1:s? -map 0:v -map 1:a? -max_interleave_delta 0 " << makeWatermarkVideoPath;
		cout << "\nFFmpeg encode command: " << ffmpegCmd.str() << "\n\n";

		// Open FFmpeg process (with pipe) for writing
		FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
		checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");

		timer::start();
		//embed watermark on the video frames
		processFrames(videoData, [&](AVFrame* frame, int& framesCount) { embedWatermarkFrame(videoData, framesCount, frame, ffmpegPipe.get()); });
		timer::end();

		cout << "\nWatermark embedding total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
		timer::start();
		//detect watermark on the video frames
		const int framesCount = processFrames(videoData, [&](AVFrame* frame, int& framesCount) { detectFrameWatermark(videoData, framesCount, frame); });
		timer::end();

		cout << "\nWatermark detection total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << executionTime(showFps, timer::elapsedSeconds() / framesCount) << "\n";
	}
	return EXIT_SUCCESS;
}

//Main frames loop logic for video watermark embedding and detection
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame)
{
	const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
	const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
	int framesCount = 0;

	// Read video frames loop
	while (av_read_frame(data.inputFormatCtx, packet.get()) >= 0)
	{
		if (!receivedValidVideoFrame(data.inputDecoderCtx, packet.get(), frame.get(), data.videoStreamIndex))
			continue;
		processFrame(frame.get(), framesCount);
	}
	// Ensure all remaining frames are flushed
	avcodec_send_packet(data.inputDecoderCtx, nullptr);
	while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
	{
		if (frame->format == data.inputDecoderCtx->pix_fmt)
			processFrame(frame.get(), framesCount);
	}
	return framesCount;
}

// Embed watermark in a video frame
void embedWatermarkFrame(const VideoProcessingContext& data, int& framesCount, AVFrame* frame, FILE* ffmpegPipe)
{
	float watermarkStrength;
	const bool embedWatermark = framesCount % data.watermarkInterval == 0;
	//if there is row padding (for alignment), we must copy the data to a contiguous block!
	if (frame->linesize[0] != data.width)
	{
		if (embedWatermark)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.frameFlatPinned + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
			//embed the watermark and receive the watermarked data back to host
			const af::array inputFrame = af::array(data.width, data.height, data.frameFlatPinned, afHost).T().as(f32);
			const af::array watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8).T();
			watermarkedFrame.host(data.frameFlatPinned);
			//write the watermarked image data
			fwrite(data.frameFlatPinned, 1, data.width * frame->height, ffmpegPipe);
		}
		else
		{
			//write from frame buffer row-by-row the the valid image data (and not the alignment bytes)
			for (int y = 0; y < data.height; y++)
				fwrite(frame->data[0] + y * frame->linesize[0], 1, data.width, ffmpegPipe);
		}
		//always write UI planes as-is
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);

	}
	//no row padding, read and write data directly
	else
	{
		if (embedWatermark)
		{
			const af::array inputFrame = af::array(data.width, data.height, frame->data[0], afHost).T().as(f32);
			const af::array watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8).T();
			watermarkedFrame.host(data.frameFlatPinned);
		}
		// Write original or modified frame to ffmpeg (pipe)
		fwrite(embedWatermark ? data.frameFlatPinned : frame->data[0], 1, data.width * frame->height, ffmpegPipe);
		fwrite(frame->data[1], 1, data.width * frame->height / 4, ffmpegPipe);
		fwrite(frame->data[2], 1, data.width * frame->height / 4, ffmpegPipe);
	}
	framesCount++;
}

// Detect the watermark for a video frame
void detectFrameWatermark(const VideoProcessingContext& data, int& framesCount, AVFrame* frame)
{
	//detect watermark after X frames
	if (framesCount % data.watermarkInterval == 0)
	{
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		const bool rowPadding = frame->linesize[0] != data.width;
		if (rowPadding)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.frameFlatPinned + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
		}
		//supply the input frame to the GPU and run the detection of the watermark
		const af::array inputFrame = af::array(data.width, data.height, rowPadding ? data.frameFlatPinned : frame->data[0], afHost).T().as(f32);
		float correlation = data.watermarkObj->detectWatermark(inputFrame, MASK_TYPE::ME);
		cout << "Correlation for frame: " << framesCount << ": " << correlation << "\n";
	}
	framesCount++;
}

// find the first video stream index
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx)
{
	for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
		if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
			return i;
	return -1;
}

//open decoder context for video
AVCodecContext* openDecoderContext(const AVCodecParameters* inputCodecParams)
{
	const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
	AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
	avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams);
	//multithreading decode
	inputDecoderCtx->thread_count = 0;
	if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
		inputDecoderCtx->thread_type = FF_THREAD_FRAME;
	else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
		inputDecoderCtx->thread_type = FF_THREAD_SLICE;
	else
		inputDecoderCtx->thread_count = 1; //don't use multithreading
	avcodec_open2(inputDecoderCtx, inputDecoder, nullptr);
	return inputDecoderCtx;
}

// Get the input video FPS (average)
string getVideoFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex)
{
	const AVRational frameRate = inputFormatCtx->streams[videoStreamIndex]->avg_frame_rate;
	return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
}

//supply a packet to the decoder and check if the received frame is valid by checking its format
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex)
{
	if (packet->stream_index != videoStreamIndex)
	{
		av_packet_unref(packet);
		return false;
	}
	int sendPacketResult = avcodec_send_packet(inputDecoderCtx, packet);
	av_packet_unref(packet);
	if (sendPacketResult != 0 || avcodec_receive_frame(inputDecoderCtx, frame) != 0)
		return false;
	const bool validFormat = frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUVJ420P;
	checkError(!validFormat, "Error: Video frame format not supported, aborting");
	return validFormat;
}

//helper method to calculate execution time in FPS or in seconds
string executionTime(const bool showFps, const double seconds) 
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

//terminates the program
void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}
