#include "Watermark.cuh"
#include "cuda_utils.hpp"
#include "kernels.cuh"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;


//initialize data and memory
Watermark::Watermark(const dim_t rows, const dim_t cols, const string randomMatrixPath, const int p, const float psnr)
	:p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	initializeMemory(rows, cols);
	loadRandomMatrix(randomMatrixPath, rows, cols);
	afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));
	cudaStreamCreate(&customStream);
}

//destructor, only custom kernels cuda stream must be destroyed
Watermark::~Watermark()
{
	cudaStreamDestroy(customStream);
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
}

//supply the input image to apply watermarking and detection
void Watermark::initializeMemory(const dim_t rows, const dim_t cols)
{
	//initialize texture
	auto textureData = cuda_utils::createTextureData(static_cast<unsigned int>(rows), static_cast<unsigned int>(cols));
	texObj = textureData.first;
	texArray = textureData.second;
	//allocate memory (Rx/rx partial sums and custom maks output) to avoid constant cudaMalloc
	const dim_t padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	RxPartial = af::array(rows, padded_cols);
	rxPartial = af::array(rows, padded_cols / 8);
	customMask = af::array(rows, cols);
	neighbors = af::array(rows * cols, (p * p) - 1);
}

//helper method to load the random noise matrix W from the file specified.
void Watermark::loadRandomMatrix(const string randomMatrixPath, const dim_t rows, const dim_t cols)
{
	std::ifstream randomMatrixStream(randomMatrixPath.c_str(), std::ios::binary);
	if (!randomMatrixStream.is_open())
		throw std::runtime_error(string("Error opening '" + randomMatrixPath + "' file for Random noise W array\n"));
	randomMatrixStream.seekg(0, std::ios::end);
	const auto total_bytes = randomMatrixStream.tellg();
	randomMatrixStream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + ", Image width: " + std::to_string(cols) + ", Image height: " + std::to_string(rows) + "\n"));
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	randomMatrixStream.read(reinterpret_cast<char*>(w_ptr.get()), total_bytes);
	randomMatrix = af::transpose(af::array(cols, rows, w_ptr.get()));
}

//compute custom mask. supports simple kernels that just apply a mask per-pixel without needing any other configuration
af::array Watermark::computeCustomMask(const af::array& image) const
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const dim3 blockSize(16, 16);
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols);
	//transfer ownership from arrayfire and copy data to cuda array
	const af::array image_transpose = image.T();
	cuda_utils::copyDataToCudaArray(image_transpose.device<float>(), rows, cols, texArray);
	float* mask_output = customMask.device<float>();
	switch (p) 
	{
		case 3: nvf<3> <<<gridSize, blockSize, 0, afStream >>> (texObj, mask_output, cols, rows); break;
		case 5: nvf<5> <<<gridSize, blockSize, 0, afStream >>> (texObj, mask_output, cols, rows); break;
		case 7: nvf<7> <<<gridSize, blockSize, 0, afStream >>> (texObj, mask_output, cols, rows); break;
		case 9: nvf<9> <<<gridSize, blockSize, 0, afStream >>> (texObj, mask_output, cols, rows); break;
	}
	//transfer ownership to arrayfire and return mask
	unlockArrays(image_transpose, customMask);
	return customMask;
}

//calls custom kernel to calculate neighbors array ("x_" array)
af::array Watermark::computeNeighborsArray(const af::array image) const 
{
	const dim3 blockSize(16, 16);
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols);
	const af::array image_transpose = image.T();
	//do a texture copy
	cuda_utils::copyDataToCudaArray(image_transpose.device<float>(), rows, cols, texArray);
	//transfer ownership from arrayfire
	float* neighbors_output = neighbors.device<float>();
	calculate_neighbors_p3<<<gridSize, blockSize, 0, afStream >>>(texObj, neighbors_output, cols, rows);
	//transfer ownership to arrayfire and return x_ array
	unlockArrays(neighbors, image_transpose);
	return neighbors;
}

//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::transformCorrelationArrays() const
{
	const int p_sq_minus_one = (p * p) - 1;
	const int p_sq_minus_one_sq = p_sq_minus_one * p_sq_minus_one;
	const auto rows = RxPartial.dims(0);
	const auto paddedCols = RxPartial.dims(1);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, p_sq_minus_one_sq, (paddedCols * rows) / p_sq_minus_one_sq), 1), p_sq_minus_one, p_sq_minus_one);
	const af::array rx = af::sum(af::moddims(rxPartial, p_sq_minus_one, (paddedCols * rows) / (8 * p_sq_minus_one)), 1);
	return std::make_pair(Rx, rx);
}

//Main watermark embedding method
//it embeds the watermark computed fom "inputImage" (always grayscale)
//into a new array based on "outputImage" (can be grayscale or RGB).
af::array Watermark::makeWatermark(const af::array& inputImage, const af::array& outputImage, af::array& coefficients, float& a, MASK_TYPE maskType) const
{
	af::array error_sequence;
	const af::array mask = maskType == MASK_TYPE::ME ?
		computePredictionErrorMask(inputImage, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES) :
		computeCustomMask(inputImage);
	const af::array u = mask * randomMatrix;
	a = strengthFactor / sqrt(af::sum<float>(af::pow(u, 2)) / (inputImage.elements()));
	return af::clamp(outputImage + (u * a), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool mask_needed) const
{
	//constant data
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	float* RxPartialData = RxPartial.device<float>();
	float* rxPartialData = rxPartial.device<float>();
	const auto paddedCols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	const dim3 blockSize(1, 64);
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, paddedCols);
	
	//enqueue "x_" kernel
	const af::array x_ = computeNeighborsArray(image);
	//enqueue prediction error mask kernel
	me_p3 <<<gridSize, blockSize, 0, customStream>>> (texObj, RxPartialData, rxPartialData, cols, paddedCols, rows);
	//wait for both streams to finish
	cudaStreamSynchronize(customStream);
	cudaStreamSynchronize(afStream);

	unlockArrays(RxPartial, rxPartial);
	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays();
	coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	errorSequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
	if (mask_needed) 
	{
		const af::array errorSequenceAbs = af::abs(errorSequence);
		return errorSequenceAbs / af::max<float>(errorSequenceAbs);
	}
	return af::array();
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::computeErrorSequence(const af::array& u, const af::array& coefficients) const 
{
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, computeNeighborsArray(u)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
af::array Watermark::computePredictionErrorMask(const af::array& image, const af::array& coefficients, af::array& errorSequence) const
{
	errorSequence = computeErrorSequence(image, coefficients);
	const af::array errorSequenceAbs = af::abs(errorSequence);
	return errorSequenceAbs / af::max<float>(errorSequenceAbs);
}

//helper method used in detectors
float Watermark::computeCorrelation(const af::array& e_u, const af::array& e_z) const 
{
	return af::dot<float>(af::flat(e_u), af::flat(e_z)) / static_cast<float>(af::norm(e_z) * af::norm(e_u));
}

//the main mask detector function
float Watermark::detectWatermark(const af::array& watermarkedImage, MASK_TYPE maskType) const
{
	af::array mask, e_z, a_z;
	if (maskType == MASK_TYPE::NVF)
	{
		computePredictionErrorMask(watermarkedImage, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = computeCustomMask(watermarkedImage);
	}
	else
		mask = computePredictionErrorMask(watermarkedImage, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	const af::array u = mask * randomMatrix;
	const af::array e_u = computeErrorSequence(u, a_z);
	return computeCorrelation(e_u, e_z);
}

//fast mask detector, used only for a video frame, by detecting the watermark based on previous frame (coefficients, x_ are supplied)
float Watermark::detectWatermarkPredictionErrorFast(const af::array& watermarkedImage, const af::array& coefficients) const
{
	af::array e_z, e_u, a_u;
	const af::array m_e = computePredictionErrorMask(watermarkedImage, coefficients, e_z);
	const af::array u = m_e * randomMatrix;
	computePredictionErrorMask(u, e_u, a_u, ME_MASK_CALCULATION_REQUIRED_NO);
	return computeCorrelation(e_u, e_z);
}

//helper method to display an af::array in a window
void Watermark::displayArray(const af::array& array, const int width, const int height) 
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}