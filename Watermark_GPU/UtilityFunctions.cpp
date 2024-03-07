#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#include "INIReader.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <arrayfire.h>
#define cimg_use_opencv
#define cimg_use_cpp11 1
#define cimg_use_png
#include "CImg.h"

using namespace cimg_library;
using std::cout;

std::string UtilityFunctions::loadProgram(std::string input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open()) {
		std::string error_str("Could not load Program: " + input);
		throw std::exception(error_str.c_str());
	}
	std::string s(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
	return s;
}

namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	float secs_passed() {
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000.0f);
	}
}

int UtilityFunctions::test_for_image(const cl::Device& device, const cl::CommandQueue &queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader &inir, const int p, const float psnr) {
	//load image from disk into an arrayfire array
	timer::start();
	const af::array image = af::rgb2gray(af::loadImage(strdup(inir.Get("paths", "image", "NO_IMAGE").c_str()), true), 0.299f, 0.587f, 0.114f);
	af::sync();
	timer::end();
	const int rows = image.dims(0);
	const int cols = image.dims(1);
	cout << "Time to load and tranfer RGB image from disk to VRAM: " << timer::secs_passed() << "\n";
	if (cols <= 64 || rows <= 16) {
		cout << "Image dimensions too low\n";
		return -1;
	}
	if (cols > device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() || cols > 7680 || rows > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || rows > 4320) {
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

int UtilityFunctions::test_for_video(const cl::Device& device, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	const int rows = inir.GetInteger("parameters_video", "rows", -1);
	const int cols = inir.GetInteger("parameters_video", "cols", -1);
	const unsigned int frames = (unsigned int)inir.GetInteger("parameters_video", "frames", -1);
	const float fps = (float)inir.GetReal("parameters_video", "fps", -1);
	const bool watermark_first_frame_only = inir.GetBoolean("parameters_video", "watermark_first_frame_only", false);
	const bool watermark_by_two_frames = inir.GetBoolean("parameters_video", "watermark_by_two_frames", false);
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

	const float frame_period = 1.0f / fps;
	float time_diff;

	/* REALTIME Watermarking of RAW video */

	float a;
	//load video
	CImgList<unsigned char> video_cimg;
	video_cimg = video_cimg.load_yuv(strdup(inir.Get("paths", "video", "NO_VIDEO").c_str()), cols, rows, 420, 0, frames - 1, 1, false);
	std::vector<af::array> frames_me;
	frames_me.reserve(frames);
	std::vector<af::array> a_x, x_;
	for (unsigned int i = 0; i < (frames / 2) + 1; i++) {
		a_x.push_back(af::constant(0, 1, 1));
	}

	af::array dummy_a_x;
	WatermarkFunctions watermarkFunctions(inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");
	watermarkFunctions.load_W(rows, cols);
	if (!watermark_first_frame_only) {
		int counter = 0;
		//af::Window window1(1280, 720, "Watermarked Video");
		for (unsigned int i = 0; i < frames; i++) {
			//διάβασμα και μεταφορά από τη RAM -> VRAM της Υ συνιστώσας
			//timer::start();
			CImg<unsigned char> temp(video_cimg.at(i).get_channel(0));
			unsigned char* y_vals = temp.data();
			af::array gpu_frame = af::transpose(af::array(cols, rows, y_vals)).as(af::dtype::f32);
			watermarkFunctions.load_image(gpu_frame);
			//υπολογισμός ME watermarked frame.
			if (i % 2 && watermark_by_two_frames == false)
				frames_me.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(dummy_a_x, &a));
			else {
				frames_me.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(a_x[counter], &a));
				counter++;
			}
			//timer::end();
			//cout << timer::secs_passed() << "\n";

			//θα δείξω σε realtime το τελικό watermarked video.
			//για να κάνουμε "simulate" το realtime, θα πρέπει να αφαιρεθεί ο χρόνος που σπαταλήθηκε στον υπολογισμό
			//αν είναι μεγαλύτερος τότε δε πετυχαίνεται realtime απεικόνιση, οπότε δείχνουμε απευθείας χωρίς sleep
			//if ((time_diff = frame_period - timer::secs_passed()) > 0) {
				//std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
			//}
			//window1.image(WatermarkFunctions::normalize_to_f32(frames_me[i]));
		}
	}
	else {
		//υδατογράφηση ΜΟΝΟ του πρώτου frame
		CImg<unsigned char> temp(video_cimg.at(0).get_channel(0));
		unsigned char* y_vals = temp.data();
		af::array gpu_frame(cols, rows, y_vals);
		gpu_frame = af::transpose(gpu_frame).as(af::dtype::f32);
		watermarkFunctions.load_image(gpu_frame);
		//frames_nvf.push_back(make_and_add_watermark_NVF(gpu_frame, w, p, psnr, queue, context, program_nvf, false));
		frames_me.push_back(watermarkFunctions.make_and_add_watermark_prediction_error(dummy_a_x, &a));
		//τα υπόλοιπα frames θα μπουν όπως έχουν
		for (unsigned int i = 1; i < frames; i++) {
			CImg<unsigned char> temp(video_cimg.at(i).get_channel(0));
			unsigned char* y_vals = temp.data();
			af::array gpu_frame(cols, rows, y_vals);
			gpu_frame = af::transpose(gpu_frame);
			frames_me.push_back(gpu_frame.as(af::dtype::f32));
		}
	}

	/*

				ΑΠΟΘΗΚΕΥΣΗ RAW->WATERMARKED ΒΙΝΤΕΟ

	*/

	//			//εδώ αποθηκεύω σε αρχείο το watermark ME video
	//			{
	//				CImgList<unsigned char> video_cimg_to_w(frames, cols, rows, 1, 3);
	//				for (int i = 0; i < frames; i++) {
	//					unsigned char* ptr = af::clamp(frames_me[i].T(),0,255).as(af::dtype::u8).host<unsigned char>();
	//					unsigned char* ptr2 = new unsigned char[rows * cols];
	//#pragma omp parallel for
	//					for (int i = 0; i < rows * cols; i++)
	//						ptr2[i] = ptr[i];
	//
	//					af::freeHost(ptr);
	//					ptr = NULL;
	//					CImg<unsigned char> temp0(cols, rows);
	//					delete[] temp0._data;
	//					//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
	//					temp0._data = new unsigned char[rows * cols];
	//#pragma omp parallel for
	//					for (int i = 0; i < rows * cols; i++)
	//						temp0._data[i] = ptr2[i];
	//					//temp0._data = ptr2;
	//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 0, temp0);
	//					delete[] ptr2;
	//					ptr2 = NULL;
	//
	//					CImg<unsigned char> temp1(video_cimg.at(i).get_channel(1));
	//					CImg<unsigned char> temp2(video_cimg.at(i).get_channel(2));
	//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 1, temp1);
	//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 2, temp2);
	//				}
	//
	//				video_cimg_to_w.save_yuv("videos\\watermarked.yuv", 420, false);
	//				/*CImgDisplay disp;
	//				for (int i = 0; i < frames; i++) {
	//					timer::start();
	//					CImg<unsigned char> temp(video_cimg_to_w.at(i).get_channel(0));
	//					timer::end();
	//					if ((time_diff = frame_period - timer::secs_passed()) > 0) {
	//						std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
	//					}
	//					disp.display(temp);
	//				}*/
	//				
	//			}


	/*

				REALTIME(?) RAW->WATERMARKED VIDEO DETECTION

	*/

	std::vector<float> frames_me_cor(frames);
	//παρακάτω είναι realtime detection (θα δείξω στην οθόνη τη ΜΕ watermark sequence)
	//μπορεί να τροποποιηθεί ώστε να βρίσκει correlation κάθε 2ο frame (τότε δε δείχνω στην οθόνη προφανώς τίποτα)
	CImgDisplay cimgd0;
	for (unsigned int i = 0; i < frames; i++) {
		timer::start();
		watermarkFunctions.load_image(frames_me[i]);
		frames_me_cor[i] = watermarkFunctions.mask_detector_prediction_error(frames_me[i]);
		timer::end();
		//εφαρμόζω clamping στο 0-255 μόνο για εμφάνιση της εικόνας! (αλλιώς αλλάζουν τα raw data που είναι λάθος)
		af::array clamped = af::clamp(frames_me[i], 0, 255);
		unsigned char* ptr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
		unsigned char* ptr2 = new unsigned char[rows * cols];
		std::memcpy(ptr2, ptr, (rows* cols) * sizeof(unsigned char));
		af::freeHost(ptr);
		ptr = NULL;
		CImg<unsigned char> temp0(cols, rows);
		delete[] temp0._data;
		//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
		temp0._data = new unsigned char[rows * cols];
		std::memcpy(temp0._data, ptr2, rows* cols * sizeof(unsigned char));

		if ((time_diff = frame_period - timer::secs_passed()) > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
		}
		cimgd0.display(temp0);

		cout << timer::secs_passed() << "\n";
		//cout << frames_me_cor[i] << "\n";
	}

	/*

				REALTIME (?) DETECTION RAW->WATERMARKED ΑΝΑ ΔΥΟ FRAMES


	*/


	//std::vector<float> frames_me_cor2(frames);
	//counter = 0;
	//for (unsigned int i = 0; i < frames; i++) {
	//	timer::start();
	//	if (i % 2 != 0){
	//		frames_me_cor2[i] = mask_detector(frames_me[i], w, a_x[counter], p, psnr, queue, context, program_me);
	//		counter++;
	//	}
	//	else {
	//		frames_me_cor2[i] = mask_detector(frames_me[i], w, p, psnr, queue, context, program_me);
	//	}
	//	timer::end();
	//	cout << timer::secs_passed() << "\n";
	//}



/*

			REALTIME (?) DETECTION ΣΥΜΠΙΕΣΜΕΝΟΥ VIDEO FILE


*/

	/*
//αρχικά διαβάζουμε το συμπιεσμένο βίντεο.
	CImgList<unsigned char> video_cimg_w;
	video_cimg_w = video_cimg_w.load_video(strdup(inir.Get("paths", "video_compressed", "NO_VIDEO").c_str()), 0, frames - 1);
	//std::vector<float> frames_nvf_cor_w(frames);
	std::vector<float> frames_me_cor_w(frames);
	//std::vector<af::array> frames_nvf_w;
	std::vector<af::array> frames_me_w;
	//frames_nvf_w.reserve(frames);
	frames_me_w.reserve(frames);
	for (unsigned int i = 0; i < frames; i++) {
		CImg<unsigned char> temp(video_cimg_w.at(i).get_channel(0));
		unsigned char* y_vals = temp.data();
		af::array gpu_frame(cols, rows, y_vals);
		//διάβασμα (συμπιεσμένων frames)
		//frames_nvf_w.push_back(gpu_frame.as(af::dtype::f32));
		frames_me_w.push_back(gpu_frame.T().as(af::dtype::f32));
	}

	//detection
	af::array clamped(rows, cols);
	//af::Window window3(cols, rows, "Watermarked Video");
	CImgDisplay cimgd;
	for (unsigned int i = 0; i < frames; i++) {
		timer::start();
		//frames_nvf_cor_w[i] = watermarks.mask_detector(frames_nvf[i], w, p, psnr, program_me);
		frames_me_cor_w[i] = watermarkFunctions.mask_detector_prediction_error(frames_me_w[i]);
		timer::end();
		//εφαρμόζω clamping στο 0-255 μόνο για εμφάνιση της εικόνας! (αλλιώς αλλάζουν τα raw data που είναι λάθος)
		clamped = af::clamp(frames_me_w[i], 0, 255);
		unsigned char* ptr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
		unsigned char* ptr2 = new unsigned char[rows * cols];
#pragma omp parallel for
		for (int i = 0; i < rows * cols; i++)
			ptr2[i] = ptr[i];

		af::freeHost(ptr);
		ptr = NULL;
		CImg<unsigned char> temp0(cols, rows);
		delete[] temp0._data;
		//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
		temp0._data = new unsigned char[rows * cols];
#pragma omp parallel for
		for (int i = 0; i < rows * cols; i++)
			temp0._data[i] = ptr2[i];

		if ((time_diff = frame_period - timer::secs_passed()) > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
		}
		cimgd.display(temp0);
		//window3.image(WatermarkFunctions::normalize_to_f32(clamped));
		//cout << timer::secs_passed() << "\n";
		//cout << frames_me_cor_w[i] << "\n";

	}
	*/
	return 0;
}