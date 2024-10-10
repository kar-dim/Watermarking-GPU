#include "cuda_utils.hpp"
#include "kernels.cuh"
#include "Watermark.cuh"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;

cudaStream_t Watermark::afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));

//initialize data and memory
Watermark::Watermark(const dim_t rows, const dim_t cols, const string randomMatrixPath, const int p, const float psnr)
	:p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	initializeMemory(rows, cols);
	loadRandomMatrix(randomMatrixPath, rows, cols);
	cudaStreamCreate(&customStream);
}

//copy constructor
Watermark::Watermark(const Watermark& other) : p(other.p), strengthFactor(other.strengthFactor), randomMatrix(other.randomMatrix)
{
	//we don't need to copy the internal buffers data, only to allocate the correct size based on other
	initializeMemory(other.customMask.dims(0), other.customMask.dims(1));
	cudaStreamCreate(&customStream);
}

//move constructor
Watermark::Watermark(Watermark&& other) noexcept : p(other.p), strengthFactor(other.strengthFactor),
randomMatrix(std::move(other.randomMatrix)), RxPartial(std::move(other.RxPartial)), rxPartial(std::move(other.rxPartial)), customMask(std::move(other.customMask)), neighbors(std::move(other.neighbors))
{
	//move texture data and nullify other
	texObj = other.texObj;
	texArray = other.texArray;
	customStream = other.customStream;
	other.texObj = 0;
	other.customStream = nullptr;
	other.texArray = nullptr;
}

//move assignment operator
Watermark& Watermark::operator=(Watermark&& other) noexcept
{
	if (this != &other) 
	{
		p = other.p;
		strengthFactor = other.strengthFactor;
		randomMatrix = std::move(other.randomMatrix);
		//move custom stream
		cudaStreamDestroy(customStream);
		customStream = other.customStream;
		other.customStream = nullptr;
		//move texture object
		cudaDestroyTextureObject(texObj);
		texObj = other.texObj;
		other.texObj = 0;
		//move texture array
		cudaFreeArray(texArray);
		texArray = other.texArray;
		other.texArray = nullptr;
		//move arrayfire arrays
		RxPartial = std::move(other.RxPartial);
		rxPartial = std::move(other.rxPartial);
		customMask = std::move(other.customMask);
		neighbors = std::move(other.neighbors);
	}
	return *this;
}

//copy assignment operator
Watermark& Watermark::operator=(const Watermark& other)
{
	if (this != &other) 
	{
		p = other.p;
		strengthFactor = other.strengthFactor;
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(texArray);
		initializeMemory(other.customMask.dims(0), other.customMask.dims(1));
		randomMatrix = other.randomMatrix;
	}
	return *this;
}

//destroy texture data (texture object, cuda array) and custom cuda stream, only if they have not been moved
Watermark::~Watermark()
{
	if (customStream != nullptr)
	    cudaStreamDestroy(customStream);
	if (texObj != 0)
		cudaDestroyTextureObject(texObj);
	if (texArray != nullptr)
		cudaFreeArray(texArray);
}

//supply the input image to apply watermarking and detection
void Watermark::initializeMemory(const dim_t rows, const dim_t cols)
{
	//initialize texture (transposed dimensions, arrayfire is column wise, we skip an extra transpose)
	auto textureData = cuda_utils::createTextureData(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
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
	const auto totalBytes = randomMatrixStream.tellg();
	randomMatrixStream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != totalBytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(totalBytes / (sizeof(float))) + ", Image width: " + std::to_string(cols) + ", Image height: " + std::to_string(rows) + "\n"));
	std::unique_ptr<float> wPtr(new float[rows * cols]);
	randomMatrixStream.read(reinterpret_cast<char*>(wPtr.get()), totalBytes);
	randomMatrix = af::transpose(af::array(cols, rows, wPtr.get()));
}

//deletes and re-initializes memory (texture, kernel arrays, random matrix array) for new image size
void Watermark::reinitialize(const string randomMatrixPath, const dim_t rows, const dim_t cols)
{
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	initializeMemory(rows, cols);
	loadRandomMatrix(randomMatrixPath, rows, cols);
}

//computes custom Mask (NVF) or neighbors array (x_) used in prediction error mask
af::array Watermark::executeTextureKernel(void (*kernel)(cudaTextureObject_t, float *, unsigned, unsigned), const af::array& image, const af::array& output) const
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const dim3 blockSize(16, 16);
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols, true);
	//transfer ownership from arrayfire and copy data to cuda array
	cuda_utils::copyDataToCudaArray(image.device<float>(), cols, rows, texArray);
	float* outputValues = output.device<float>();
	kernel << <gridSize, blockSize, 0, afStream >> > (texObj, outputValues, cols, rows);
	//transfer ownership to arrayfire and return output array
	unlockArrays(image, output);
	return output;
}

//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::transformCorrelationArrays() const
{
	const int neighborsSize = (p * p) - 1;
	const int neighborsSizeSq = neighborsSize * neighborsSize;
	const auto rows = RxPartial.dims(0);
	const auto paddedElems = RxPartial.dims(0) * RxPartial.dims(1);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, neighborsSizeSq, paddedElems / neighborsSizeSq), 1), neighborsSize, neighborsSize);
	const af::array rx = af::sum(af::moddims(rxPartial, neighborsSize, paddedElems / (8 * neighborsSize)), 1);
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
		executeTextureKernel(nvf<3>, inputImage, customMask);
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
	const dim3 blockSize(64, 1);
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, paddedCols);
	
	//enqueue "x_" kernel
	const af::array x_ = executeTextureKernel(calculate_neighbors_p3, image, neighbors);
	//enqueue prediction error mask kernel
	me_p3 <<<gridSize, blockSize, 0, customStream>>> (texObj, RxPartialData, rxPartialData, cols, paddedCols, rows);
	
	//wait for both streams to finish
	cudaStreamSynchronize(customStream);
	cudaStreamSynchronize(afStream);
	unlockArrays(RxPartial, rxPartial);
	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays();
	coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	errorSequence = af::moddims(af::moddims(image, 1, rows * cols) - af::matmulTT(coefficients, x_), rows, cols);
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
	return af::moddims(af::moddims(u, 1, u.dims(0) * u.dims(1)) - af::matmulTT(coefficients, executeTextureKernel(calculate_neighbors_p3, u, neighbors)), u.dims(0), u.dims(1));
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
		mask = executeTextureKernel(nvf<3>, watermarkedImage, customMask);
	}
	else
		mask = computePredictionErrorMask(watermarkedImage, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	const af::array u = mask * randomMatrix;
	const af::array e_u = computeErrorSequence(u, a_z);
	return computeCorrelation(e_u, e_z);
}

//helper method to display an af::array in a window
void Watermark::displayArray(const af::array& array, const int width, const int height) 
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}