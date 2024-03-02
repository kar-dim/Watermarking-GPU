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

af::array UtilityFunctions::normalize_to_f32(af::array& a)
{
	float mx = af::max<float>(a);
	float mn = af::min<float>(a);
	float diff = mx - mn;
	return (a - mn) / diff;
}

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
	timer::start();
	CImg<unsigned char> rgb_image_cimg(strdup(inir.Get("paths", "image", "NO_IMAGE").c_str()));
	unsigned char* rgb_img_vals = rgb_image_cimg.data();
	timer::end();
	cout << "Time to load and tranfer RGB (uint8 *3 channels) image from disk to RAM: " << timer::secs_passed() << "\n";
	if (rgb_image_cimg.width() <= 16 || rgb_image_cimg.height() <= 16) {
		cout << "Image dimensions too low\n";
		return -1;
	}
	if (rgb_image_cimg.width() > device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() || rgb_image_cimg.width() > 7680 || rgb_image_cimg.height() > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || rgb_image_cimg.height() > 4320) {
		cout << "Image dimensions too high for this GPU\n";
		return -1;
	}
	const int rows = rgb_image_cimg.height();
	const int cols = rgb_image_cimg.width();

	timer::start();
	af::array rgb_image(cols, rows, 3, rgb_img_vals);
	af::sync();
	timer::end();
	cout << "Time to transfer RGB (uint8 *3 channels) image from RAM to VRAM: " << timer::secs_passed() << "\n\n";
	rgb_image = af::transpose(rgb_image);

	//grayscale
	const af::array image = af::round(0.299 * rgb_image(af::span, af::span, 0)) + af::round(0.587 * rgb_image(af::span, af::span, 1)) + af::round(0.114 * rgb_image(af::span, af::span, 2));
	
	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	WatermarkFunctions watermarks(image, inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");

	float a;
	af::array a_x;

	//warmup for arrayfire
	watermarks.make_and_add_watermark_custom(&a);
	watermarks.make_and_add_watermark_prediction_error(a_x, &a);

	//make NVF watermark
	timer::start();
	af::array watermark_NVF = watermarks.make_and_add_watermark_custom(&a);
	timer::end();
	cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
	cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//make ME watermark
	timer::start();
	af::array watermark_ΜΕ = watermarks.make_and_add_watermark_prediction_error(a_x, &a);
	timer::end();
	cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
	cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//warmup for arrayfire
	watermarks.mask_detector_custom(watermark_NVF);
	watermarks.mask_detector_prediction_error(watermark_ΜΕ);

	//detection of NVF
	timer::start();
	float correlation_nvf = watermarks.mask_detector_custom(watermark_NVF);
	timer::end();
	cout << "Time to calculate correlation (NVF) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

	//detection of ME
	timer::start();
	float correlation_me = watermarks.mask_detector_prediction_error(watermark_ΜΕ);
	timer::end();
	cout << "Time to calculate correlation (ME) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";
	cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
	cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";

	return 0;
}

//TODO refactor this...
int UtilityFunctions::test_for_video(const cl::Device& device, const cl::CommandQueue& queue, const cl::Context& context, const cl::Program& program_nvf, const cl::Program& program_me, const INIReader& inir, const int p, const float psnr) {
	const int rows = inir.GetInteger("parameters_video", "rows", -1);
	const int cols = inir.GetInteger("parameters_video", "cols", -1);
	const unsigned int frames = (unsigned int)inir.GetInteger("parameters_video", "frames", -1);
	const float fps = (float)inir.GetReal("parameters_video", "fps", -1);
	if (rows <= 16 || cols <= 16) {
		cout << "Video dimensions too low\n";
		return -1;
	}
	if (rows > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || rows > 2160 || cols > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || cols > 3840) {
		cout << "Video dimensions too high for this GPU\n";
		return -1;
	}
	if (fps <= 1 || fps > 60) {
		cout << "Video FPS is too low or too high\n";
		return -1;
	}
	if (frames <= 1) {
		cout << "Frame count too low\n";
		return -1;
	}
	//πόση ώρα ανάμεσα σε κάθε frame
	const float frame_period = 1.0f / fps;
	float time_diff;

	/*

					REALTIME ΥΔΑΤΟΓΡΑΦΗΣΗ (RAW VIDEO)

	*/
	//επιλογή είτε να υδατογραφήσουμε ΟΛΟ ΤΟ ΒΙΝΤΕΟ ειτε να υδατογραφήσουμε μόνο το πρώτο frame
	bool first_frame_w = false, two_frames_watermark = false;
	float a;
	//φόρτωση του video
	//CImgDisplay disp;
	CImgList<unsigned char> video_cimg;
	video_cimg = video_cimg.load_yuv(strdup(inir.Get("paths", "video", "NO_VIDEO").c_str()), cols, rows, 420, 0, frames - 1, 1, false);
	//οι δυο vectors θα έχουν όλα τα watermarked frames.
	//std::vector<af::array> frames_nvf;
	//frames_nvf.reserve(frames);
	std::vector<af::array> frames_me;
	frames_me.reserve(frames);
	std::vector<af::array> a_x, x_;
	for (unsigned int i = 0; i < (frames / 2) + 1; i++) {
		a_x.push_back(af::constant(0, 1, 1));
	}
	int counter = 0;
	af::array dummy_a_x;
	WatermarkFunctions watermarks(inir.Get("paths", "w_path", "w.txt"), p, psnr, program_me, program_nvf, "nvf");
	watermarks.load_W(rows, cols);
	if (!first_frame_w) {
		//af::Window window1(1280, 720, "Watermarked Video");
		for (unsigned int i = 0; i < frames; i++) {
			//διάβασμα και μεταφορά από τη RAM -> VRAM της Υ συνιστώσας
			//timer::start();
			CImg<unsigned char> temp(video_cimg.at(i).get_channel(0));
			unsigned char* y_vals = temp.data();
			af::array gpu_frame(cols, rows, y_vals);
			//af::print("ar", gpu_frame);
			gpu_frame = af::transpose(gpu_frame).as(af::dtype::f32);
			watermarks.load_image(gpu_frame);
			//υπολογισμός ME watermarked frame.
			if (i % 2 && two_frames_watermark == false)
				frames_me.push_back(watermarks.make_and_add_watermark_prediction_error(dummy_a_x, &a));
			else {
				frames_me.push_back(watermarks.make_and_add_watermark_prediction_error(a_x[counter], &a));
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
			//window1.image(normalize_to_f32(frames_me[i]));
		}
	}
	else {
		//υδατογράφηση ΜΟΝΟ του πρώτου frame
		CImg<unsigned char> temp(video_cimg.at(0).get_channel(0));
		unsigned char* y_vals = temp.data();
		af::array gpu_frame(cols, rows, y_vals);
		gpu_frame = af::transpose(gpu_frame).as(af::dtype::f32);
		watermarks.load_image(gpu_frame);
		//frames_nvf.push_back(make_and_add_watermark_NVF(gpu_frame, w, p, psnr, queue, context, program_nvf, false));
		frames_me.push_back(watermarks.make_and_add_watermark_prediction_error(dummy_a_x, &a));
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

	//Φτιάχνουμε άλλους δυο vectors που θα κρατάνε το correlation
	//std::vector<float> frames_nvf_cor(frames);
	std::vector<float> frames_me_cor(frames);
	//παρακάτω είναι realtime detection (θα δείξω στην οθόνη τη ΜΕ watermark sequence)
	//μπορεί να τροποποιηθεί ώστε να βρίσκει correlation κάθε 2ο frame (τότε δε δείχνω στην οθόνη προφανώς τίποτα)
	CImgDisplay cimgd0;
	for (unsigned int i = 0; i < frames; i++) {
		timer::start();
		watermarks.load_image(frames_me[i]);
		//frames_nvf_cor[i] = watermarks.mask_detector(frames_nvf[i], w, p, psnr, program_me);
		frames_me_cor[i] = watermarks.mask_detector_prediction_error(frames_me[i]);
		timer::end();
		//εφαρμόζω clamping στο 0-255 μόνο για εμφάνιση της εικόνας! (αλλιώς αλλάζουν τα raw data που είναι λάθος)
		af::array clamped = af::clamp(frames_me[i], 0, 255);
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
		frames_me_cor_w[i] = watermarks.mask_detector_prediction_error(frames_me_w[i]);
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
		//window3.image(normalize_to_f32(clamped));
		//cout << timer::secs_passed() << "\n";
		//cout << frames_me_cor_w[i] << "\n";

		return 0;
	}
}