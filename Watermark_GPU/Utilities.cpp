#pragma warning(disable:4996)
#include "Utilities.hpp"
#include "Watermark.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <arrayfire.h>
#include <thread>
#include <cmath>
#include <stdexcept>

using namespace cimg_library;
using std::string;
using std::cout;

string Utilities::loadProgram(const string& input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open())
		throw std::runtime_error("Could not load Program: " + input + "\n");
	return string(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
}

string Utilities::add_suffix_before_extension(const string& file, const string& suffix) {
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
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

//see https://blog.bearcats.nl/accurate-sleep-function/
void Utilities::accurate_timer_sleep(double seconds) {
	double estimate = 5e-3, mean = 5e-3, m2 = 0;
	int64_t count = 1;
	while (seconds > estimate) {
		auto start = std::chrono::high_resolution_clock::now();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		auto end = std::chrono::high_resolution_clock::now();
		double observed = (end - start).count() / 1e9;
		seconds -= observed;
		++count;
		double delta = observed - mean;
		mean += delta / count;
		m2 += delta * (observed - mean);
		double stddev = sqrt(m2 / (count - 1));
		estimate = mean + stddev;
	}
	// spin lock
	auto start = std::chrono::high_resolution_clock::now();
	while ((std::chrono::high_resolution_clock::now() - start).count() / 1e9 < seconds);
}

//main detection method of a watermarked sequence thats calls the watermark detector and optionally prints correlation and time passed
void Utilities::realtime_detection(Watermark& watermarkFunctions, const std::vector<af::array> &watermarked_frames, const int frames, const bool display_frames, const float frame_period) {
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
		cout << "Watermark detection seconds passed: " << watermark_time_secs << "\n";
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
				accurate_timer_sleep(time_diff);
			window.display(cimg_watermarked);
		}
		cout << "Correlation of " << i + 1 << " frame: " << correlations[i] << "\n\n";
	}
}