﻿#include "cimg_init.h"
#include "opencl_init.h"
#include "Utilities.hpp"
#include "Watermark.hpp"
#include "Watermark_GPU.hpp"
#include <af/opencl.h>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <INIReader.h>
#include <iostream>
//#include <omp.h>
#include <string>
#include <vector>

#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

using std::cout;
using std::string;
using namespace cimg_library;

/*!
 *  \brief  This is a project implementation of my Thesis with title: 
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(void)
{
	//open parameters file
	INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load opencl configuration file\n";
		exit_program(EXIT_FAILURE);
	}

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

	const cl::Context context(afcl::getContext(true));
	const cl::Device device({ afcl::getDeviceId() });

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", -1.0f));

	//TODO for p>3 we have problems with ME masking buffers
	if (p != 3) {
		cout << "For now, only p=3 is allowed\n";
		exit_program(EXIT_FAILURE);
	}
	/*if (p != 3 && p != 5 && p != 7 && p != 9) {
		cout << "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9\n";
		exit_program(EXIT_FAILURE);
	}*/

	if (psnr <= 0) {
		cout << "PSNR must be a positive number\n";
		exit_program(EXIT_FAILURE);
	}

	//compile opencl kernels
	cl::Program program_nvf, program_me;
	try {
		string program_data = Utilities::load_file_as_string("kernels/nvf.cl");
		program_nvf = cl::Program(cl::Context{ afcl::getContext()}, program_data);
		program_nvf.build({ device }, std::format("-cl-fast-relaxed-math -cl-mad-enable -Dp={}", p).c_str());
		program_data = Utilities::load_file_as_string("kernels/me_p3.cl");
		program_me = cl::Program(context, program_data);
		program_me.build({ device }, "-cl-fast-relaxed-math -cl-mad-enable");
	}
	catch (cl::Error& e) {
		cout << "Could not build a kernel, Reason:\n\n";
		cout << e.what();
		if (program_nvf.get() != NULL)
			cout << program_nvf.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		if (program_me.get() != NULL)
			cout << program_me.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		exit_program(EXIT_FAILURE);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exit_program(EXIT_FAILURE);
	}

	//test algorithms
	try {
		const int code = inir.GetBoolean("parameters_video", "test_for_video", false) == true ?
			test_for_video(device, program_nvf, program_me, inir, p, psnr) :
			test_for_image(device, program_nvf, program_me, inir, p, psnr);
		exit_program(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exit_program(EXIT_FAILURE);
	}
	exit_program(EXIT_SUCCESS);
}

int test_for_image(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	const string image_file = inir.Get("paths", "image", "NO_IMAGE");
	const bool show_fps = inir.GetBoolean("options", "execution_time_in_fps", false);
	//load image from disk into an arrayfire array
	timer::start();
	const af::array rgb_image = af::loadImage(image_file.c_str(), true);
	const af::array image = af::rgb2gray(rgb_image, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	af::sync();
	timer::end();
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	cout << "Time to load and transfer RGB image from disk to VRAM: " << timer::secs_passed() << "\n\n";
	if (cols <= 64 || rows <= 16) {
		cout << "Image dimensions too low\n";
		return EXIT_FAILURE;
	}
	if (cols > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) || cols > 7680 || rows > static_cast<dim_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) || rows > 4320) {
		cout << "Image dimensions too high for this GPU\n";
		return EXIT_FAILURE;
	}

	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	Watermark watermark_obj(rgb_image, image, inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");

	float a;
	af::array a_x;
	//warmup for arrayfire
	watermark_obj.make_and_add_watermark(a_x, a, MASK_TYPE::NVF, IMAGE_TYPE::RGB);
	watermark_obj.make_and_add_watermark(a_x, a, MASK_TYPE::ME, IMAGE_TYPE::RGB);

	//make NVF watermark
	timer::start();
	af::array watermark_NVF = watermark_obj.make_and_add_watermark(a_x, a, MASK_TYPE::NVF, IMAGE_TYPE::RGB);
	timer::end();
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", a, rows, cols, p, psnr, execution_time(show_fps, timer::secs_passed()));

	//make ME watermark
	timer::start();
	af::array watermark_ME = watermark_obj.make_and_add_watermark(a_x, a, MASK_TYPE::ME, IMAGE_TYPE::RGB);
	timer::end();
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", a, rows, cols, p, psnr, execution_time(show_fps, timer::secs_passed()));

	const af::array watermarked_NVF_gray = af::rgb2gray(watermark_NVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	const af::array watermarked_ME_gray = af::rgb2gray(watermark_ME, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	//warmup for arrayfire
	watermark_obj.mask_detector(watermarked_NVF_gray, MASK_TYPE::NVF);
	watermark_obj.mask_detector(watermarked_ME_gray, MASK_TYPE::ME);

	//detection of NVF
	timer::start();
	float correlation_nvf = watermark_obj.mask_detector(watermarked_NVF_gray, MASK_TYPE::NVF);
	timer::end();
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, timer::secs_passed()));

	//detection of ME
	timer::start();
	float correlation_me = watermark_obj.mask_detector(watermarked_ME_gray, MASK_TYPE::ME);
	timer::end();
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, timer::secs_passed()));
	
	cout << std::format("Correlation [NVF]: {:.16f}\n", correlation_nvf);
	cout << std::format("Correlation [ME]: {:.16f}\n", correlation_me);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) {
		cout << "\nSaving watermarked files to disk...\n";
//#pragma omp parallel sections
		//{
//#pragma omp section
			af::saveImageNative(Utilities::add_suffix_before_extension(image_file, "_W_NVF").c_str(), watermark_NVF.as(af::dtype::u8));
//#pragma omp section
			af::saveImageNative(Utilities::add_suffix_before_extension(image_file, "_W_ME").c_str(), watermark_ME.as(af::dtype::u8));
		//}
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

int test_for_video(const cl::Device& device, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	const int rows = inir.GetInteger("parameters_video", "rows", -1);
	const int cols = inir.GetInteger("parameters_video", "cols", -1);
	const bool show_fps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int frames = inir.GetInteger("parameters_video", "frames", -1);
	const float fps = (float)inir.GetReal("parameters_video", "fps", -1);
	const bool watermark_first_frame_only = inir.GetBoolean("parameters_video", "watermark_first_frame_only", false);
	const bool watermark_by_two_frames = inir.GetBoolean("parameters_video", "watermark_by_two_frames", false);
	const bool display_frames = inir.GetBoolean("parameters_video", "display_frames", false);
	if (rows <= 64 || cols <= 64) {
		cout << "Video dimensions too low\n";
		return EXIT_FAILURE;
	}
	if (rows > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || cols > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) {
		cout << "Video dimensions too high for this GPU\n";
		return EXIT_FAILURE;
	}
	if (fps <= 15 || fps > 60) {
		cout << "Video FPS is too low or too high\n";
		return EXIT_FAILURE;
	}
	if (frames <= 1) {
		cout << "Frame count too low\n";
		return EXIT_FAILURE;
	}

	CImgList<unsigned char> video_cimg;
	string video_path;
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
	Watermark watermark_obj(inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");
	watermark_obj.load_W(rows, cols);

	//realtime watermarking of raw video
	const bool make_watermark = inir.GetBoolean("parameters_video", "watermark_make", false);
	if (make_watermark == true)
	{
		//load video from file
		video_path = inir.Get("paths", "video", "NO_VIDEO");
		video_cimg = CImgList<unsigned char>::get_load_yuv(video_path.c_str(), cols, rows, 420, 0, frames - 1, 1, false);
		if (watermark_first_frame_only == false) {
			int counter = 0;
			for (int i = 0; i < frames; i++) {
				//copy from CImg to arrayfire
				watermark_obj.load_image(Utilities::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(i)));
				//calculate watermarked frame, if "by two frames" is on, we keep coefficients per two frames, to be used per 2 detection frames
				if (watermark_by_two_frames == true) {
					if (i % 2 != 0)
						watermarked_frames.push_back(watermark_obj.make_and_add_watermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
					else {
						watermarked_frames.push_back(watermark_obj.make_and_add_watermark(coefficients[counter], a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
						counter++;
					}
				}
				else
					watermarked_frames.push_back(watermark_obj.make_and_add_watermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
			}
		}
		else {
			//add the watermark only in the first frame
			//copy from CImg to arrayfire
			watermark_obj.load_image(Utilities::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(0)));
			watermarked_frames.push_back(watermark_obj.make_and_add_watermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
			//rest of the frames will be as-is, no watermark
			//NOTE this is useless if there is no compression, because the new frames are irrelevant with the first (watermarked), the correlation will be close to 0
			for (int i = 1; i < frames; i++)
				watermarked_frames.push_back(Utilities::cimg_yuv_to_afarray<unsigned char>(video_cimg.at(i)).as(af::dtype::f32));
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
//#pragma omp parallel for
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
						Utilities::accurate_timer_sleep(time_diff);
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
			realtime_detection(watermark_obj, watermarked_frames, frames, display_frames, frame_period, show_fps);
	}

	//realtime watermarked video detection by two frames
	if (inir.GetBoolean("parameters_video", "watermark_detection_by_two_frames", false) == true) {
		if (make_watermark == false) {
			cout << "Please set 'watermark_make' to true in settings file, in order to be able to detect the watermark.\n";
		}
		else {
			std::vector<float> correlations(frames);
			int counter = 0;
			for (int i = 0; i < frames; i++) {
				timer::start();
				if (i % 2 != 0) {
					correlations[i] = watermark_obj.mask_detector_prediction_error_fast(watermarked_frames[i], coefficients[counter]);
					timer::end();
					cout << "Watermark detection execution time (fast): " << execution_time(show_fps, timer::secs_passed()) << "\n";
					counter++;
				}
				else {
					correlations[i] = watermark_obj.mask_detector(watermarked_frames[i], MASK_TYPE::ME);
					timer::end();
					cout << "Watermark detection execution time: " << execution_time(show_fps, timer::secs_passed()) << "\n";
				}
				cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
			}
		}
	}

	//realtimne watermark detection of a compressed file
	if (inir.GetBoolean("parameters_video", "watermark_detection_compressed", false) == true) {
		//read compressed file
		string video_compressed_path = inir.Get("paths", "video_compressed", "NO_VIDEO");
		CImgList<unsigned char>video_cimg_w = CImgList<unsigned char>::get_load_video(video_compressed_path.c_str(), 0, frames - 1);
		std::vector<af::array> watermarked_frames(frames);
		for (int i = 0; i < frames; i++)
			watermarked_frames[i] = Utilities::cimg_yuv_to_afarray<unsigned char>(video_cimg_w.at(i));
		realtime_detection(watermark_obj, watermarked_frames, frames, display_frames, frame_period, show_fps);
	}
	return EXIT_SUCCESS;
}

std::string execution_time(const bool show_fps, const double seconds) {
	return show_fps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

//main detection method of a watermarked sequence thats calls the watermark detector and optionally prints correlation and time passed
void realtime_detection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarked_frames, const int frames, const bool display_frames, const float frame_period, const bool show_fps) {
	std::vector<float> correlations(frames);
	CImgDisplay window;
	const auto rows = static_cast<unsigned int>(watermarked_frames[1].dims(0));
	const auto cols = static_cast<unsigned int>(watermarked_frames[0].dims(1));
	float time_diff;
	for (int i = 0; i < frames; i++) {
		timer::start();
		correlations[i] = watermarkFunctions.mask_detector(watermarked_frames[i], MASK_TYPE::ME);
		timer::end();
		const float watermark_time_secs = timer::secs_passed();
		cout << "Watermark detection execution time: " << execution_time(show_fps, watermark_time_secs) << "\n";
		if (display_frames) {
			timer::start();
			af::array clamped = af::clamp(watermarked_frames[i], 0, 255);
			unsigned char* watermarked_frames_ptr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
			CImg<unsigned char> cimg_watermarked(cols, rows);
			std::memcpy(cimg_watermarked.data(), watermarked_frames_ptr, rows * cols * sizeof(unsigned char));
			af::freeHost(watermarked_frames_ptr);
			watermarked_frames_ptr = NULL;
			timer::end();
			if ((time_diff = frame_period - (watermark_time_secs + timer::secs_passed())) > 0)
				Utilities::accurate_timer_sleep(time_diff);
			window.display(cimg_watermarked);
		}
		cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
	}
}

void exit_program(const int exit_code) {
	std::system("pause");
	std::exit(exit_code);
}
