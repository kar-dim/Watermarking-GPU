#include "Utilities.hpp"
#include <chrono>
#include <string>

using std::string;

string Utilities::addSuffixBeforeExtension(const string& file, const string& suffix)
{
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

namespace timer 
{
	void start() 
	{
		startTime = std::chrono::high_resolution_clock::now();
	}
	void end() 
	{
		currentTime = std::chrono::high_resolution_clock::now();
	}
	float elapsedSeconds() 
	{
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000.0f);
	}
}