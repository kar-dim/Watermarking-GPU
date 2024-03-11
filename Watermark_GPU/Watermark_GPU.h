#pragma once
#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#undef max
#undef min
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <arrayfire.h>
#include <iostream>
#include "INIReader.h"
#define cimg_use_opencv
#define cimg_use_png
#include <CImg.h>
#include "WatermarkFunctions.h"
#include "UtilityFunctions.h"
#include <vector>

using namespace cimg_library;
using std::cout;

int test_for_image(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	//load image from disk into an arrayfire array
	timer::start();
	const af::array image = af::rgb2gray(af::loadImage(strdup(inir.Get("paths", "image", "NO_IMAGE").c_str()), true), 0.299f, 0.587f, 0.114f);
	af::sync();
	timer::end();
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	cout << "Time to load and tranfer RGB image from disk to VRAM: " << timer::secs_passed() << "\n";
	if (cols <= 64 || rows <= 16) {
		cout << "Image dimensions too low\n";
		return -1;
	}
	if (cols > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) || cols > 7680 || rows > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) || rows > 4320) {
		cout << "Image dimensions too high for this GPU\n";
		return -1;
	}

	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	WatermarkFunctions watermarkFunctions(image, inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");

	float a;
	af::array a_x;
	//warmup for arrayfire
	watermarkFunctions.make_and_add_watermark_custom(&a);
	watermarkFunctions.make_and_add_watermark_prediction_error(a_x, &a);

	//make NVF watermark
	timer::start();
	af::array watermark_NVF = watermarkFunctions.make_and_add_watermark_custom(&a);
	timer::end();
	cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
	cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//make ME watermark
	timer::start();
	af::array watermark_ΜΕ = watermarkFunctions.make_and_add_watermark_prediction_error(a_x, &a);
	timer::end();
	cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
	cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//warmup for arrayfire
	watermarkFunctions.mask_detector_custom(watermark_NVF);
	watermarkFunctions.mask_detector_prediction_error(watermark_ΜΕ);

	//detection of NVF
	timer::start();
	float correlation_nvf = watermarkFunctions.mask_detector_custom(watermark_NVF);
	timer::end();
	cout << "Time to calculate correlation (NVF) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//detection of ME
	timer::start();
	float correlation_me = watermarkFunctions.mask_detector_prediction_error(watermark_ΜΕ);
	timer::end();
	cout << "Time to calculate correlation (ME) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";
	cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
	cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";
	return 0;
}

int test_for_video(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	const int rows = inir.GetInteger("parameters_video", "rows", -1);
	const int cols = inir.GetInteger("parameters_video", "cols", -1);
	const int frames = inir.GetInteger("parameters_video", "frames", -1);
	const float fps = (float)inir.GetReal("parameters_video", "fps", -1);
	const bool watermark_first_frame_only = inir.GetBoolean("parameters_video", "watermark_first_frame_only", false);
	const bool watermark_by_two_frames = inir.GetBoolean("parameters_video", "watermark_by_two_frames", false);
	const bool display_frames = inir.GetBoolean("parameters_video", "display_frames", false);
	if (rows <= 64 || cols <= 64) {
		cout << "Video dimensions too low\n";
		return -1;
	}
	if (rows > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || cols > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) {
		cout << "Video dimensions too high for this GPU\n";
		return -1;
	}
	if (fps <= 15 || fps > 60) {
		cout << "Video FPS is too low or too high\n";
		return -1;
	}
	if (frames <= 1) {
		cout << "Frame count too low\n";
		return -1;
	}

	CImgList<unsigned char> video_cimg;
	std::vector<af::array> watermarked_frames;
	std::vector<af::array> coefficients;
	watermarked_frames.reserve(frames);
	coefficients.reserve((frames / 2) + 1);
	//preallocate coefficient's vector with empty arrays
	for (int i = 0; i < (frames / 2) + 1; i++)
		coefficients.push_back(af::constant<float>(0.0f, 1, 1));
	const float frame_period = 1.0f / fps;
	float time_diff, a;

	//initialize watermark functions class
	af::array dummy_a_x;
	WatermarkFunctions watermarkFunctions(inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");
	watermarkFunctions.load_W(rows, cols);

	//realtime watermarking of raw video
	const bool make_watermark = inir.GetBoolean("parameters_video", "watermark_make", false);
	if (make_watermark == true)
	{
		//load video from file
		video_cimg = CImgList<unsigned char>::get_load_yuv(strdup(inir.Get("paths", "video", "NO_VIDEO").c_str()), cols, rows, 420, 0, frames - 1, 1, false);
		if (watermark_first_frame_only == false) {
			int counter = 0;
			for (int i = 0; i < frames; i++) {
				//copy from CImg to arrayfire
				watermarkFunctions.load_image(UtilityFunctions::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(i)));
				//calculate watermarked frame, if "by two frames" is on, we keep coefficients per two frames, to be used per 2 detection frames
				if (watermark_by_two_frames == true) {
					if (i % 2 != 0)
						watermarked_frames.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(dummy_a_x, &a));
					else {
						watermarked_frames.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(coefficients[counter], &a));
						counter++;
					}
				}
				else
					watermarked_frames.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(dummy_a_x, &a));
			}
		}
		else {
			//add the watermark only in the first frame
			//copy from CImg to arrayfire
			watermarkFunctions.load_image(UtilityFunctions::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(0)));
			watermarked_frames.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(dummy_a_x, &a));
			//rest of the frames will be as-is, no watermark
			//NOTE this is useless if there is no compression, because the new frames are irrelevant with the first (watermarked), the correlation will be close to 0
			//with compression, the watermark is "kept alive" in (some) subsequent frames, only then it makes sense to use this method!
			for (int i = 1; i < frames; i++)
				watermarked_frames.push_back(UtilityFunctions::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(i)).as(af::dtype::f32));
		}
	}

	//save watermarked video to raw YUV (must be processed with ffmpeg later to add file headers, then it can be compressed etc)
	if (inir.GetBoolean("parameters_video", "watermark_save_to_file", false) == true)
	{
		if (make_watermark == false) {
			cout << "Please set 'watermark_make' to true in settins file, in order to be able to save it.\n";
		}
		else {
			CImgList<unsigned char> video_cimg_watermarked(frames, cols, rows, 1, 3);
#pragma omp parallel for
			for (int i = 0; i < frames; i++) {
				unsigned char* watermarked_frames_ptr = af::clamp(watermarked_frames[i].T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
				CImg<unsigned char> cimg_Y(cols, rows);
				std::memcpy(cimg_Y.data(), watermarked_frames_ptr, sizeof(unsigned char) * rows * cols);
				af::freeHost(watermarked_frames_ptr);
				watermarked_frames_ptr = NULL;
				video_cimg_watermarked.at(i).draw_image(0, 0, 0, 0, cimg_Y);
				video_cimg_watermarked.at(i).draw_image(0, 0, 0, 1, CImg<unsigned char>(video_cimg.at(i).get_channel(1)));
				video_cimg_watermarked.at(i).draw_image(0, 0, 0, 2, CImg<unsigned char>(video_cimg.at(i).get_channel(2)));
			}
			//save watermark frames to file
			video_cimg_watermarked.save_yuv((inir.Get("parameters_video", "watermark_save_to_file_path", "./watermarked.yuv")).c_str(), 420, false);

			if (display_frames == true) {
				CImgDisplay window;
				for (int i = 0; i < frames; i++) {
					timer::start();
					window.display(video_cimg_watermarked.at(i).get_channel(0));
					timer::end();
					if ((time_diff = frame_period - timer::secs_passed()) > 0) {
						UtilityFunctions::accurate_timer_sleep(time_diff);
					}
				}
			}
		}

	}

	//realtime watermarked video detection
	if (inir.GetBoolean("parameters_video", "watermark_detection", false) == true) {
		if (make_watermark == false)
			cout << "Please set 'watermark_make' to true in settins file, in order to be able to detect the watermark.\n";
		else
			UtilityFunctions::realtime_detection(watermarkFunctions, watermarked_frames, frames, display_frames, frame_period);
	}

	//realtime watermarked video detection by two frames
	if (inir.GetBoolean("parameters_video", "watermark_detection_by_two_frames", false) == true) {
		if (make_watermark == false) {
			cout << "Please set 'watermark_make' to true in settins file, in order to be able to detect the watermark.\n";
		}
		else {
			std::vector<float> correlations(frames);
			int counter = 0;
			for (int i = 0; i < frames; i++) {
				timer::start();
				if (i % 2 != 0) {
					correlations[i] = watermarkFunctions.mask_detector_prediction_error_fast(watermarked_frames[i], coefficients[counter]);
					timer::end();
					cout << "Watermark detection (fast) secs passed: " << timer::secs_passed() << "\n";
					counter++;
				}
				else {
					correlations[i] = watermarkFunctions.mask_detector_prediction_error(watermarked_frames[i]);
					timer::end();
					cout << "Watermark detection secs passed: " << timer::secs_passed() << "\n";
				}
				cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
			}
		}
	}

	//realtimne watermark detection of a compressed file
	if (inir.GetBoolean("parameters_video", "watermark_detection_compressed", false) == true) {
		//read compressed file
		CImgList<unsigned char>video_cimg_w = CImgList<unsigned char>::get_load_video(strdup(inir.Get("paths", "video_compressed", "NO_VIDEO").c_str()), 0, frames - 1);
		std::vector<af::array> watermarked_frames(frames);
#pragma omp parallel for
		for (int i = 0; i < frames; i++) {
			watermarked_frames[i] = UtilityFunctions::cimg_yuv_to_afarray<unsigned char>(video_cimg_w.at(i));
		}

		UtilityFunctions::realtime_detection(watermarkFunctions, watermarked_frames, frames, display_frames, frame_period);
	}

	return 0;
}
