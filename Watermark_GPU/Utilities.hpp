﻿#pragma once
#include <arrayfire.h>
#include <chrono>
#include <CImg.h>
#include <string>

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class Utilities 
{
public:
	static std::string loadFileString(const std::string& input);
	static std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
	static void accurateSleep(double seconds);
	template<typename T>
	static af::array cimgYuvToAfarray(const cimg_library::CImg<T> &cimgImage) 
	{
		return af::transpose(af::array(cimgImage.width(), cimgImage.height(), cimgImage.get_channel(0).data()).as(f32));
	}
};

/*!
 *  \brief  simple methods to calculate execution times
 *  \author Dimitris Karatzas
 */
namespace timer 
{
	static std::chrono::time_point<std::chrono::steady_clock> startTime, currentTime;
	void start();
	void end();
	float elapsedSeconds();
}