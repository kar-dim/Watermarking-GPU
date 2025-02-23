#include "Utilities.hpp"
#include <chrono>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

using std::string;

string Utilities::loadFileString(const string& input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open())
		throw std::runtime_error("Could not load Program: " + input + "\n");
	return string(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
}

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