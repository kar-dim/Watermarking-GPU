#include "cuda_utils.hpp"
#include "main_utils.hpp"
#include "Utilities.hpp"
#include "Watermark.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <INIReader.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <stdio.h>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}
using std::cout;
using std::string;

#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU, CUDA version
 *  \author Dimitris Karatzas
 */
int main(void)
{
	//open parameters file
	const INIReader inir("settings.ini");
	if (inir.ParseError() < 0) 
	{
		cout << "Could not load CUDA configuration file\n";
		exitProgram(EXIT_FAILURE);
	}

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < 24; i++) { }

	af::info();
	cout << "\n";

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = inir.GetFloat("parameters", "psnr", -1.0f);

	//TODO for p>3 we have problems with ME masking buffers
	if (p != 3) 
	{
		cout << "For now, only p=3 is allowed\n";
		exitProgram(EXIT_FAILURE);
	}
	/*if (p != 3 && p != 5 && p != 7 && p != 9) {
		cout << "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9\n";
		exitProgram(EXIT_FAILURE);
	}*/

	if (psnr <= 0) 
	{
		cout << "PSNR must be a positive number\n";
		exitProgram(EXIT_FAILURE);
	}

	cudaDeviceProp properties = cuda_utils::getDeviceProperties();

	//test algorithms
	try {
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ?
			testForVideo(inir, videoFile, properties, p, psnr) :
			testForImage(inir, properties, p, psnr);
		exitProgram(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

//embed watermark for static images
int testForImage(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr) 
{
	const string imageFile = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";

	//load image from disk into an arrayfire array
	timer::start();
	const af::array rgbImage = af::loadImage(imageFile.c_str(), true);
	const af::array image = af::rgb2gray(rgbImage, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	af::sync();
	timer::end();
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	cout << "Time to load and transfer RGB image from disk to VRAM: " << timer::elapsedSeconds() << "\n\n";
	if (cols <= 64 || rows <= 16) 
	{
		cout << "Image dimensions too low\n";
		return EXIT_FAILURE;
	}

	if (cols > static_cast<dim_t>(properties.maxTexture2D[0]) || rows > static_cast<dim_t>(properties.maxTexture2D[1])) 
	{
		cout << "Image dimensions too high for this GPU\n";
		return EXIT_FAILURE;
	}

	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	Watermark watermarkObj(rows, cols, inir.Get("paths", "watermark", "w.txt"), p, psnr);

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

	const af::array watermarkedNVFgray = af::rgb2gray(watermarkNVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	const af::array watermarkedMEgray = af::rgb2gray(watermarkME, R_WEIGHT, G_WEIGHT, B_WEIGHT);
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
#pragma omp parallel sections
		{
#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_NVF").c_str(), watermarkNVF.as(u8));
#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_ME").c_str(), watermarkME.as(u8));
		}
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const INIReader& inir, const string& videoFile, const cudaDeviceProp& properties, const int p, const float psnr) 
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = inir.GetInteger("parameters_video", "watermark_interval", 30);

	//Set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//Load input video
	AVFormatContext* inputFormatCtx = nullptr;
	if (avformat_open_input(&inputFormatCtx, videoFile.c_str(), nullptr, nullptr) < 0) 
	{
		std::cout << "ERROR: Failed to open input video file\n";
		return EXIT_FAILURE;
	}
	avformat_find_stream_info(inputFormatCtx, nullptr);
	av_dump_format(inputFormatCtx, 0, videoFile.c_str(), 0);

	//Find video stream
	int videoStreamIndex = -1;
	for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++) 
	{
		if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoStreamIndex = i;
			break;
		}
	}
	if (videoStreamIndex == -1)
	{
		std::cout << "ERROR: No video stream found\n";
		return EXIT_FAILURE;
	}

	//Open input video decoder
	const AVCodecParameters* inputCodecParams = inputFormatCtx->streams[videoStreamIndex]->codecpar;
	const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
	AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
	avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams);
	avcodec_open2(inputDecoderCtx, inputDecoder, nullptr);

	//initialize watermark functions class
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	Watermark watermarkObj(height, width, inir.Get("paths", "watermark", ""), p, psnr);

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");
		// Build the FFmpeg command using std::ostringstream
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt yuv420p "
			<< "-s " << width << "x" << height
			<< " -r 30 -i - -i " << videoFile << " " << ffmpegOptions
			<< " -map 0:v -map 1:a -shortest " << makeWatermarkVideoPath;

		// Open FFmpeg process
		FILE* ffmpeg = _popen(ffmpegCmd.str().c_str(), "wb");
		if (!ffmpeg) 
		{
			std::cout << "Error: Could not open FFmpeg pipe." << std::endl;
			return EXIT_FAILURE;
		}

		//read frames
		int counter = 0;
		float watermarkStrength;
		uint8_t* frameFlatPinned = nullptr;
		cudaHostAlloc((void**)&frameFlatPinned, width * height * sizeof(uint8_t), cudaHostAllocDefault);
		af::array inputFrame, watermarkedFrame;
		AVPacket* packet = av_packet_alloc();
		AVFrame* frame = av_frame_alloc();
		while (av_read_frame(inputFormatCtx, packet) >= 0) 
		{
			if (packet->stream_index == videoStreamIndex)
			{
				avcodec_send_packet(inputDecoderCtx, packet);
				if (avcodec_receive_frame(inputDecoderCtx, frame) == 0) 
				{
					//don't write after each frame because compression propagates the watermark and affects the video quality
					if (counter % watermarkInterval == 0) 
					{
						//if there is row padding (for alignment), we must copy the data to a contiguous block!
						if (frame->linesize[0] != width)
						{
							//TODO check if it works? + benchmark with openmp
							for (int y = 0; y < height; y++)
							{
								memcpy(frameFlatPinned + y * width, frame->data[0] + y * frame->linesize[0], width);
							}
							inputFrame = af::array(width, height, frameFlatPinned, afHost).T().as(f32);
							watermarkedFrame = watermarkObj.makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8);
							watermarkedFrame.host(frameFlatPinned);
							//TODO check if it works? + benchmark with openmp
							for (int y = 0; y < height; y++)
							{
								memcpy(frame->data[0] + y * frame->linesize[0], frameFlatPinned + y * width, width);
							}
						}
						//else, use original pointer, no need to copy data
						else
						{
							inputFrame = af::array(width, height, frame->data[0], afHost).T().as(f32);
							watermarkedFrame = watermarkObj.makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8).T();
							watermarkedFrame.host(frame->data[0]);
						}
					}

					// Write modified frame to ffmpeg (pipe)
					fwrite(frame->data[0], 1, frame->linesize[0] * frame->height, ffmpeg);
					fwrite(frame->data[1], 1, frame->linesize[1] * frame->height / 2, ffmpeg);
					fwrite(frame->data[2], 1, frame->linesize[2] * frame->height / 2, ffmpeg);
					counter++;
				}
			}

			av_packet_unref(packet);
		}
		av_frame_free(&frame);
		av_packet_free(&packet);
		cudaFreeHost(frameFlatPinned);
		_pclose(ffmpeg);
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false)) 
	{
		timer::start();

		float correlation;
		
		uint8_t* frameFlatPinned = nullptr;
		cudaHostAlloc((void**)&frameFlatPinned, width * height * sizeof(uint8_t), cudaHostAllocDefault);
		AVPacket* packet = av_packet_alloc();
		AVFrame* frame = av_frame_alloc();
		af::array inputFrame;
		int framesCount = 0;

		//read all frames
		while (av_read_frame(inputFormatCtx, packet) >= 0) 
		{
			if (packet->stream_index == videoStreamIndex) 
			{
				avcodec_send_packet(inputDecoderCtx, packet);
				if (avcodec_receive_frame(inputDecoderCtx, frame) == 0) 
				{
					//detect watermark after X frames
					if (framesCount % watermarkInterval == 0) 
					{
						//if there is row padding (for alignment), we must copy the data to a contiguous block!
						if (frame->linesize[0] != width)
						{
							//TODO check if it works? + benchmark with openmp
							for (int y = 0; y < height; y++)
							{
								memcpy(frameFlatPinned + y * width, frame->data[0] + y * frame->linesize[0], width);
							}
							inputFrame = af::array(width, height, frameFlatPinned, afHost).T().as(f32);
						}
						//else, use original pointer, no need to copy data
						else
						{
							inputFrame = af::array(width, height, frame->data[0], afHost).T().as(f32);
						}

						correlation = watermarkObj.detectWatermark(inputFrame, MASK_TYPE::ME);
						cout << "Correlation for frame: " << framesCount << ": " << correlation << "\n";
					}
					framesCount++;
				}
			}
			av_packet_unref(packet);
		}
		av_frame_free(&frame);
		av_packet_free(&packet);
		cudaFreeHost(frameFlatPinned);

		timer::end();
		cout << "\nWatermark detection total execution time: " << executionTime(false, timer::elapsedSeconds()) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << executionTime(showFps, timer::elapsedSeconds() / framesCount) << "\n";
	}

	// Cleanup
	avformat_close_input(&inputFormatCtx);
	avcodec_free_context(&inputDecoderCtx);
	return EXIT_SUCCESS;
}

string executionTime(const bool showFps, const double seconds) 
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}