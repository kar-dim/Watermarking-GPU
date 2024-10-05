#pragma once
#include <arrayfire.h>
#include <concepts>
#include <cuda_runtime.h>
#include <string>
#include <utility>

enum MASK_TYPE 
{
	ME,
	NVF
};

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class Watermark 
{
private:
	int p;
	float strengthFactor;
	af::array randomMatrix, RxPartial, rxPartial, customMask, neighbors;
	cudaStream_t customStream;
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	static cudaStream_t afStream;

	void initializeMemory(const dim_t rows, const dim_t cols);
	void loadRandomMatrix(const std::string randomMatrixPath, const dim_t rows, const dim_t cols);
	std::pair<af::array, af::array> transformCorrelationArrays() const;
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const;
	af::array executeTextureKernel(void (*kernel)(cudaTextureObject_t, float*, unsigned, unsigned), const af::array& image, const af::array& output) const;
	af::array computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const;
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const;
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }
public:
	Watermark(const dim_t rows, const dim_t cols, const std::string randomMatrixPath, const int p, const float psnr);
	Watermark(const Watermark& other);
	Watermark(Watermark&& other) noexcept;
	Watermark& operator=(Watermark&& other) noexcept;
	Watermark& operator=(const Watermark& other);
	~Watermark();
	void reinitialize(const std::string randomMatrixPath, const dim_t rows, const dim_t cols);
	af::array makeWatermark(const af::array& inputImage, const af::array& output_image, af::array& coefficients, float& a, MASK_TYPE maskType) const;
	float detectWatermark(const af::array& watermarkedImage, MASK_TYPE maskType) const;
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900);
};