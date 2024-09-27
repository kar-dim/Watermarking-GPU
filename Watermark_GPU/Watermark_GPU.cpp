#include "cimg_init.h"
#include "cuda_utils.hpp"
#include "Utilities.hpp"
#include "Watermark.cuh"
#include "Watermark_GPU.hpp"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <INIReader.h>
#include <iostream>
//#include <omp.h>
#include <string>
#include <vector>

using std::cout;
using std::string;
using namespace cimg_library;

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

	//omp_set_num_threads(omp_get_max_threads());
//#pragma omp parallel for
	//for (int i = 0; i < 24; i++) { }

	af::info();
	cout << "\n";

	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", -1.0f));

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
		const int code = inir.GetBoolean("parameters_video", "test_for_video", false) == true ?
			testForVideo(inir, properties, p, psnr) :
			testForImage(inir, properties, p, psnr);
		exitProgram(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

int testForImage(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr) 
{
	const string imageFile = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
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

	if (cols > static_cast<dim_t>(properties.maxTexture2D[0]) || cols > 7680 || rows > static_cast<dim_t>(properties.maxTexture2D[1]) || rows > 4320) 
	{
		cout << "Image dimensions too high for this GPU\n";
		return EXIT_FAILURE;
	}

	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	Watermark watermarkObj(rgbImage, image, inir.Get("paths", "w_path", "w.txt"), p, psnr);

	float a;
	af::array a_x;
	//warmup for arrayfire
	watermarkObj.makeWatermark(a_x, a, MASK_TYPE::NVF, IMAGE_TYPE::RGB);
	watermarkObj.makeWatermark(a_x, a, MASK_TYPE::ME, IMAGE_TYPE::RGB);

	//make NVF watermark
	timer::start();
	const af::array watermarkNVF = watermarkObj.makeWatermark(a_x, a, MASK_TYPE::NVF, IMAGE_TYPE::RGB);
	timer::end();
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", a, rows, cols, p, psnr, executionTime(showFps, timer::elapsedSeconds()));

	//make ME watermark
	timer::start();
	const af::array watermarkME = watermarkObj.makeWatermark(a_x, a, MASK_TYPE::ME, IMAGE_TYPE::RGB);
	timer::end();
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", a, rows, cols, p, psnr, executionTime(showFps, timer::elapsedSeconds()));

	const af::array watermarkedNVFgray = af::rgb2gray(watermarkNVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	const af::array watermarkedMEgray = af::rgb2gray(watermarkME, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	//warmup for arrayfire
	watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
	watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);

	//detection of NVF
	timer::start();
	float correlationNvf = watermarkObj.detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
	timer::end();
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, timer::elapsedSeconds()));

	//detection of ME
	timer::start();
	float correlationMe = watermarkObj.detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
	timer::end();
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, timer::elapsedSeconds()));
	
	cout << std::format("Correlation [NVF]: {:.16f}\n", correlationNvf);
	cout << std::format("Correlation [ME]: {:.16f}\n", correlationMe);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
	{
		cout << "\nSaving watermarked files to disk...\n";
//#pragma omp parallel sections
		//{
//#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_NVF").c_str(), watermarkNVF.as(af::dtype::u8));
//#pragma omp section
			af::saveImageNative(Utilities::addSuffixBeforeExtension(imageFile, "_W_ME").c_str(), watermarkME.as(af::dtype::u8));
		//}
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

int testForVideo(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr) 
{
	const int rows = inir.GetInteger("parameters_video", "rows", -1);
	const int cols = inir.GetInteger("parameters_video", "cols", -1);
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int frames = inir.GetInteger("parameters_video", "frames", -1);
	const float fps = (float)inir.GetReal("parameters_video", "fps", -1);
	const bool watermarkFirstFrameOnly = inir.GetBoolean("parameters_video", "watermark_first_frame_only", false);
	const bool watermarkByTwoFrames = inir.GetBoolean("parameters_video", "watermark_by_two_frames", false);
	const bool displayFrames = inir.GetBoolean("parameters_video", "display_frames", false);
	if (rows <= 64 || cols <= 64) 
	{
		cout << "Video dimensions too low\n";
		return EXIT_FAILURE;
	}
	if (rows > static_cast<dim_t>(properties.maxTexture2D[1]) || cols > static_cast<dim_t>(properties.maxTexture2D[0])) 
	{
		cout << "Video dimensions too high for this GPU\n";
		return EXIT_FAILURE;
	}
	if (fps <= 15 || fps > 60) 
	{
		cout << "Video FPS is too low or too high\n";
		return EXIT_FAILURE;
	}
	if (frames <= 1) 
	{
		cout << "Frame count too low\n";
		return EXIT_FAILURE;
	}

	CImgList<unsigned char> videoCimg;
	string videoPath;
	std::vector<af::array> watermarkedFrames;
	std::vector<af::array> coefficients;
	watermarkedFrames.reserve(frames);
	coefficients.reserve((frames / 2) + 1);
	//preallocate coefficient's vector with empty arrays
	for (int i = 0; i < (frames / 2) + 1; i++)
		coefficients.push_back(af::constant<float>(0.0f, 1, 1));
	const float framePeriod = 1.0f / fps;
	float timeDiff, a;

	//initialize watermark functions class
	af::array dummy_a_x;
	Watermark watermarkFunctions(inir.Get("paths", "w_path", "w.txt"), p, psnr);
	watermarkFunctions.loadRandomMatrix(rows, cols);

	//realtime watermarking of raw video
	const bool makeWatermark = inir.GetBoolean("parameters_video", "watermark_make", false);
	if (makeWatermark == true)
	{
		//load video from file
		videoPath = inir.Get("paths", "video", "NO_VIDEO");
		videoCimg = CImgList<unsigned char>::get_load_yuv(videoPath.c_str(), cols, rows, 420, 0, frames - 1, 1, false);
		if (watermarkFirstFrameOnly == false) 
		{
			int counter = 0;
			for (int i = 0; i < frames; i++) 
			{
				//copy from CImg to arrayfire
				watermarkFunctions.loadImage(Utilities::cimgYuvToAfarray<unsigned char>(videoCimg.at(i)));
				//calculate watermarked frame, if "by two frames" is on, we keep coefficients per two frames, to be used per 2 detection frames
				if (watermarkByTwoFrames == true) 
				{
					if (i % 2 != 0)
						watermarkedFrames.push_back(watermarkFunctions.makeWatermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
					else 
					{
						watermarkedFrames.push_back(watermarkFunctions.makeWatermark(coefficients[counter], a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
						counter++;
					}
				}
				else
					watermarkedFrames.push_back(watermarkFunctions.makeWatermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
			}
		}
		else 
		{
			//add the watermark only in the first frame
			//copy from CImg to arrayfire
			watermarkFunctions.loadImage(Utilities::cimgYuvToAfarray<unsigned char>(videoCimg.at(0)));
			watermarkedFrames.push_back(watermarkFunctions.makeWatermark(dummy_a_x, a, MASK_TYPE::ME, IMAGE_TYPE::GRAYSCALE));
			//rest of the frames will be as-is, no watermark
			//NOTE this is useless if there is no compression, because the new frames are irrelevant with the first (watermarked), the correlation will be close to 0
			for (int i = 1; i < frames; i++)
				watermarkedFrames.push_back(Utilities::cimgYuvToAfarray<unsigned char>(videoCimg.at(i)).as(af::dtype::f32));
		}
	}

	//save watermarked video to raw YUV (must be processed with ffmpeg later to add file headers, then it can be compressed etc)
	if (inir.GetBoolean("parameters_video", "watermark_save_to_file", false) == true)
	{
		if (makeWatermark == false) 
		{
			cout << "Please set 'watermark_make' to true in settins file, in order to be able to save it.\n";
		}
		else 
		{
			CImgList<unsigned char> videoCimgWatermarked(frames, cols, rows, 1, 3);
//#pragma omp parallel for
			for (int i = 0; i < frames; i++) 
			{
				unsigned char* watermarkedFramesPtr = af::clamp(watermarkedFrames[i].T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
				CImg<unsigned char> cimgY(cols, rows);
				std::memcpy(cimgY.data(), watermarkedFramesPtr, sizeof(unsigned char) * rows * cols);
				af::freeHost(watermarkedFramesPtr);
				watermarkedFramesPtr = NULL;
				videoCimgWatermarked.at(i).draw_image(0, 0, 0, 0, cimgY);
				videoCimgWatermarked.at(i).draw_image(0, 0, 0, 1, CImg<unsigned char>(videoCimg.at(i).get_channel(1)));
				videoCimgWatermarked.at(i).draw_image(0, 0, 0, 2, CImg<unsigned char>(videoCimg.at(i).get_channel(2)));
			}
			//save watermark frames to file
			videoCimgWatermarked.save_yuv((inir.Get("parameters_video", "watermark_save_to_file_path", "./watermarked.yuv")).c_str(), 420, false);

			if (displayFrames == true) 
			{
				CImgDisplay window;
				for (int i = 0; i < frames; i++)
				{
					timer::start();
					window.display(videoCimgWatermarked.at(i).get_channel(0));
					timer::end();
					if ((timeDiff = framePeriod - timer::elapsedSeconds()) > 0)
						Utilities::accurateSleep(timeDiff);
				}
			}
		}

	}

	//realtime watermarked video detection
	if (inir.GetBoolean("parameters_video", "watermark_detection", false) == true) 
	{
		if (makeWatermark == false)
			cout << "Please set 'watermark_make' to true in settins file, in order to be able to detect the watermark.\n";
		else
			realtimeDetection(watermarkFunctions, watermarkedFrames, frames, displayFrames, framePeriod, showFps);
	}

	//realtime watermarked video detection by two frames
	if (inir.GetBoolean("parameters_video", "watermark_detection_by_two_frames", false) == true) 
	{
		if (makeWatermark == false) 
		{
			cout << "Please set 'watermark_make' to true in settings file, in order to be able to detect the watermark.\n";
		}
		else 
		{
			std::vector<float> correlations(frames);
			int counter = 0;
			for (int i = 0; i < frames; i++) 
			{
				timer::start();
				if (i % 2 != 0) 
				{
					correlations[i] = watermarkFunctions.detectWatermarkPredictionErrorFast(watermarkedFrames[i], coefficients[counter]);
					timer::end();
					cout << "Watermark detection execution time (fast): " << executionTime(showFps, timer::elapsedSeconds()) << "\n";
					counter++;
				}
				else 
				{
					correlations[i] = watermarkFunctions.detectWatermark(watermarkedFrames[i], MASK_TYPE::ME);
					timer::end();
					cout << "Watermark detection execution time: " << executionTime(showFps, timer::elapsedSeconds()) << "\n";
				}
				cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
			}
		}
	}

	//realtimne watermark detection of a compressed file
	if (inir.GetBoolean("parameters_video", "watermark_detection_compressed", false) == true) 
	{
		//read compressed file
		string videoCompressedPath = inir.Get("paths", "video_compressed", "NO_VIDEO");
		CImgList<unsigned char>videoCimgW = CImgList<unsigned char>::get_load_video(videoCompressedPath.c_str(), 0, frames - 1);
		std::vector<af::array> watermarkedFrames(frames);
		for (int i = 0; i < frames; i++)
			watermarkedFrames[i] = Utilities::cimgYuvToAfarray<unsigned char>(videoCimgW.at(i));
		realtimeDetection(watermarkFunctions, watermarkedFrames, frames, displayFrames, framePeriod, showFps);
	}
	return EXIT_SUCCESS;
}

std::string executionTime(bool showFps, double seconds) 
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

//main detection method of a watermarked sequence thats calls the watermark detector and optionally prints correlation and time passed
void realtimeDetection(Watermark& watermarkFunctions, const std::vector<af::array>& watermarkedFrames, const int frames, const bool displayFrames, const float framePeriod, const bool showFps) 
{
	std::vector<float> correlations(frames);
	CImgDisplay window;
	const auto rows = static_cast<unsigned int>(watermarkedFrames[1].dims(0));
	const auto cols = static_cast<unsigned int>(watermarkedFrames[0].dims(1));
	float timeDiff;
	for (int i = 0; i < frames; i++) 
	{
		timer::start();
		correlations[i] = watermarkFunctions.detectWatermark(watermarkedFrames[i], MASK_TYPE::ME);
		timer::end();
		const float watermarkTimeSecs = timer::elapsedSeconds();
		cout << "Watermark detection execution time: " << executionTime(showFps, watermarkTimeSecs) << "\n";
		if (displayFrames) 
		{
			timer::start();
			af::array clamped = af::clamp(watermarkedFrames[i], 0, 255);
			unsigned char* watermarkedFramesPtr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
			CImg<unsigned char> cimg_watermarked(cols, rows);
			std::memcpy(cimg_watermarked.data(), watermarkedFramesPtr, rows * cols * sizeof(unsigned char));
			af::freeHost(watermarkedFramesPtr);
			watermarkedFramesPtr = NULL;
			timer::end();
			if ((timeDiff = framePeriod - (watermarkTimeSecs + timer::elapsedSeconds())) > 0)
				Utilities::accurateSleep(timeDiff);
			window.display(cimg_watermarked);
		}
		cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
	}
}

void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}