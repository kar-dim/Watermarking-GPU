#pragma once
#include "arrayfire.h"
#include "opencl_init.h"
#include <af/opencl.h>
#include <concepts>
#include <string>
#include <utility>

enum MASK_TYPE 
{
	ME,
	NVF
};

struct dim2
{
	dim_t rows;
	dim_t cols;
};

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class Watermark {

private:
	static constexpr int RxMappings[64]
	{
		0,  1,  2,  3,  4,  5,  6,  7,
		1,  8,  9,  10, 11, 12, 13, 14,
		2,  9,  15, 16, 17, 18, 19, 20,
		3,  10, 16, 21, 22, 23, 24, 25,
		4,  11, 17, 22, 26, 27, 28, 29,
		5,  12, 18, 23, 27, 30, 31, 32,
		6,  13, 19, 24, 28, 31, 33, 34,
		7,  14, 20, 25, 29, 32, 34, 35
	};
	const cl::Context context{ afcl::getContext(true) };
	const cl::CommandQueue queue{ afcl::getQueue(true) };
	const cl::Buffer RxMappingsBuff{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 64, (void*)RxMappings, NULL };
	dim2 dims, meKernelDims, texKernelDims;
	std::vector<cl::Program> programs;
	int p;
	float strengthFactor;
	af::array randomMatrix;
	cl::Image2D image2d;

	void initializeMemory();
	void loadRandomMatrix(const std::string randomMatrixPath);
	std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const;
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const;
	af::array computeCustomMask() const;
	af::array computeScaledNeighbors(const af::array& coefficients) const;
	af::array computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const;
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const;
	void copyDataToTexture(const af::array& image) const;
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }

public:
	Watermark(const dim_t rows, const dim_t cols, const std::string randomMatrixPath, const int p, const float psnr, const std::vector<cl::Program>& programs);
	Watermark(const Watermark& other);
	Watermark(Watermark&& other) noexcept = default; //default move constructor is fine, will call std::move for all fields
	Watermark& operator=(Watermark&& other) noexcept = default; //default move assignmet is fine
	Watermark& operator=(const Watermark& other);
	void reinitialize(const std::string randomMatrixPath, const dim_t rows, const dim_t cols);
	af::array makeWatermark(const af::array& inputImage, const af::array& outputImage, float& watermarkStrength, MASK_TYPE maskType) const;
	float detectWatermark(const af::array& watermarkedImage, MASK_TYPE maskType) const;
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900);
};