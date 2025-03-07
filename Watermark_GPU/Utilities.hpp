#pragma once
#include <arrayfire.h>
#include <chrono>
#include <string>

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class Utilities 
{
public:
	static std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
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