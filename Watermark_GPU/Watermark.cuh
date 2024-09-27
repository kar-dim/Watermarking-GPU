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

enum IMAGE_TYPE 
{
	RGB,
	GRAYSCALE
};

/*!
 *  \brief  Functions for watermark computation and detection
 *  \author Dimitris Karatzas
 */
class Watermark 
{
private:
	const std::string randomMatrixPath;
	const int p;
	const float strengthFactor;
	af::array rgbImage, image, randomMatrix;
	cudaStream_t afStream, customStream;
	cudaTextureObject_t texObj = 0;
	cudaArray* texArray = nullptr;
	af::array RxPartial, rxPartial, customMask, neighbors;

	float computeCorrelation(const af::array& e_u, const af::array& e_z) const;
	af::array computeCustomMask(const af::array& image) const;
	af::array computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const;
	af::array computePredictionErrorMask(const af::array& image, const af::array& coefficients, af::array& errorSequence) const;
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const;
	af::array computeNeighborsArray(const af::array image) const;
	std::pair<af::array, af::array> transformCorrelationArrays() const;
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }
public:
	Watermark(const af::array& rgbImage, const af::array& image, const std::string& randomMatrixPath, const int p, const float psnr);
	Watermark(const std::string &randomMatrixPath, const int p, const float psnr);
	~Watermark();
	void loadRandomMatrix(const dim_t rows, const dim_t cols);
	void loadImage(const af::array& image);
	af::array makeWatermark(af::array& coefficients, float& a, MASK_TYPE type, IMAGE_TYPE imageType) const;
	float detectWatermark(const af::array& watermarkedImage, MASK_TYPE maskType) const;
	float detectWatermarkPredictionErrorFast(const af::array& watermarkedImage, const af::array& coefficients) const;
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900);
};