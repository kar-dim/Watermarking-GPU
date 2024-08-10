#pragma once
#include <arrayfire.h>
#include <string>
#include "cimg_init.h"
#include <vector>
#include "WatermarkFunctions.cuh"

using namespace cimg_library;
using std::string;

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class UtilityFunctions {
public:
	static string add_suffix_before_extension(const string& file, const string& suffix);
	static void accurate_timer_sleep(double seconds);
	template<typename T>
	static af::array cimg_yuv_to_afarray(const CImg<T>& cimg_image) {
		return af::transpose(af::array(cimg_image.width(), cimg_image.height(), cimg_image.get_channel(0).data()).as(af::dtype::f32));
	}
	static void realtime_detection(WatermarkFunctions& watermarkFunctions, const std::vector<af::array>& watermarked_frames, const int frames, const bool display_frames, const float frame_period);
};

/*!
 *  \brief  simple methods to calculate execution times
 *  \author Dimitris Karatzas
 */
namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	float secs_passed();
}