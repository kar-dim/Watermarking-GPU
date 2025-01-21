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
#define UINT(x) static_cast<unsigned int>(x)

using std::string;

cudaStream_t Watermark::afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));

//initialize data and memory
Watermark::Watermark(const dim_t rows, const dim_t cols, const string randomMatrixPath, const int p, const float psnr)
	: dims(UINT(cols), UINT(rows)), meKernelDims(UINT((cols + 63) & ~63), UINT(rows)), p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	initializeMemory();
	loadRandomMatrix(randomMatrixPath);
}

//copy constructor
Watermark::Watermark(const Watermark& other) 
	: dims(other.dims), meKernelDims(other.meKernelDims), p(other.p), strengthFactor(other.strengthFactor), randomMatrix(other.randomMatrix)
{
	//we don't need to copy the internal buffers data, only to allocate the correct size based on other
	initializeMemory();
}

//move constructor
Watermark::Watermark(Watermark&& other) noexcept 
	: dims(other.dims), meKernelDims(other.meKernelDims), p(other.p), strengthFactor(other.strengthFactor), randomMatrix(std::move(other.randomMatrix)),
	  RxPartial(std::move(other.RxPartial)), rxPartial(std::move(other.rxPartial)), customMask(std::move(other.customMask)), neighbors(std::move(other.neighbors))
{
	static constexpr auto moveMember = [](auto& thisData, auto& otherData, auto value) { thisData = otherData; otherData = value; };
	//move texture data and nullify other
	moveMember(texObj, other.texObj, 0);
	moveMember(texArray, other.texArray, nullptr);
}

//helper method to copy the parameters of another watermark object (for move/copy operators)
void Watermark::copyParams(const Watermark& other) noexcept
{
	dims = other.dims;
	meKernelDims = other.meKernelDims;
	p = other.p;
	strengthFactor = other.strengthFactor;
}

//move assignment operator
Watermark& Watermark::operator=(Watermark&& other) noexcept
{
	static constexpr auto moveAndDestroyMember = [](auto& thisData, auto& otherData, auto& deleter, auto value) { deleter(thisData); thisData = otherData; otherData = value; };
	if (this != &other) 
	{
		copyParams(other);
		//move texture object/array and arrayfire arrays
		moveAndDestroyMember(texObj, other.texObj, cudaDestroyTextureObject, 0);
		moveAndDestroyMember(texArray, other.texArray, cudaFreeArray, nullptr);
		//move arrayfire arrays
		randomMatrix = std::move(other.randomMatrix);
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
		copyParams(other);
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(texArray);
		initializeMemory();
		randomMatrix = other.randomMatrix;
	}
	return *this;
}

//destroy texture data (texture object, cuda array) only if they have not been moved
Watermark::~Watermark()
{
	static constexpr auto destroy = [](auto&& resource, auto&& deleter) { if (resource) deleter(resource); };
	destroy(texObj, cudaDestroyTextureObject);
	destroy(texArray, cudaFreeArray);
}

//supply the input image to apply watermarking and detection
void Watermark::initializeMemory()
{
	//initialize texture (transposed dimensions, arrayfire is column wise, we skip an extra transpose)
	auto textureData = cuda_utils::createTextureData(dims.x, dims.y);
	texObj = textureData.first;
	texArray = textureData.second;
	//allocate memory (Rx/rx partial sums and custom maks output) to avoid constant cudaMalloc
	RxPartial = af::array(dims.y, meKernelDims.x);
	rxPartial = af::array(dims.y, meKernelDims.x / 8);
	customMask = af::array(dims.y, dims.x);
	neighbors = af::array(dims.y, dims.x);
}

//helper method to load the random noise matrix W from the file specified.
//This is the random generated watermark generated from a Normal distribution generator with mean 0 and standard deviation 1
void Watermark::loadRandomMatrix(const string randomMatrixPath)
{
	std::ifstream randomMatrixStream(randomMatrixPath.c_str(), std::ios::binary);
	if (!randomMatrixStream.is_open())
		throw std::runtime_error(string("Error opening '" + randomMatrixPath + "' file for Random noise W array\n"));
	randomMatrixStream.seekg(0, std::ios::end);
	const auto totalBytes = randomMatrixStream.tellg();
	randomMatrixStream.seekg(0, std::ios::beg);
	if (dims.y * dims.x * sizeof(float) != totalBytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(totalBytes / (sizeof(float))) + ", Image width: " + std::to_string(dims.x) + ", Image height: " + std::to_string(dims.y) + "\n"));
	std::unique_ptr<float> wPtr(new float[dims.y * dims.x]);
	randomMatrixStream.read(reinterpret_cast<char*>(wPtr.get()), totalBytes);
	randomMatrix = af::transpose(af::array(dims.x, dims.y, wPtr.get()));
}

//deletes and re-initializes memory (texture, kernel arrays, random matrix array) for new image size
void Watermark::reinitialize(const string randomMatrixPath, const dim_t rows, const dim_t cols)
{
	dims = { UINT(cols), UINT(rows) };
	meKernelDims = { UINT((cols + 63) & ~63), UINT(rows) };
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	initializeMemory();
	loadRandomMatrix(randomMatrixPath);
}

//computes custom Mask (NVF) or neighbors array (x_) used in prediction error mask
af::array Watermark::computeCustomMask(const af::array& image) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(texKernelBlockSize, dims.y, dims.x, true);
	//transfer ownership from arrayfire and copy data to cuda array
	cuda_utils::copyDataToCudaArray(image.device<float>(), dims.x, dims.y, texArray);
	nvf<3> << <gridSize, texKernelBlockSize, 0, afStream >> > (texObj, customMask.device<float>(), dims.x, dims.y);
	//transfer ownership to arrayfire and return output array
	unlockArrays(image, customMask);
	return customMask;
}

//computes custom Mask (NVF) or neighbors array (x_) used in prediction error mask
af::array Watermark::computeScaledNeighbors(const af::array& coefficients) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(texKernelBlockSize, dims.y, dims.x, true);
	calculate_neighbors_p3 << <gridSize, texKernelBlockSize, 0, afStream >> > (texObj, neighbors.device<float>(), coefficients.device<float>(), dims.x, dims.y);
	//transfer ownership to arrayfire and return output array
	unlockArrays(neighbors, coefficients);
	return neighbors;
}

//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::transformCorrelationArrays() const
{
	const int neighborsSize = (p * p) - 1;
	const int neighborsSizeSq = neighborsSize * neighborsSize;
	const auto paddedElems = RxPartial.dims(0) * RxPartial.dims(1);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, neighborsSizeSq, paddedElems / neighborsSizeSq), 1), neighborsSize, neighborsSize);
	const af::array rx = af::sum(af::moddims(rxPartial, neighborsSize, paddedElems / (8 * neighborsSize)), 1);
	return std::make_pair(Rx, rx);
}

//Main watermark embedding method
//it embeds the watermark computed from "inputImage" (always grayscale)
//into a new array based on "outputImage" (can be grayscale or RGB).
af::array Watermark::makeWatermark(const af::array& inputImage, const af::array& outputImage, float& watermarkStrength, MASK_TYPE maskType) const
{
	af::array errorSequence, coefficients;
	const af::array mask = maskType == MASK_TYPE::ME ?
		computePredictionErrorMask(inputImage, errorSequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES) :
		computeCustomMask(inputImage);
	const af::array u = mask * randomMatrix;
	watermarkStrength = strengthFactor / sqrt(af::sum<float>(af::pow(u, 2)) / (inputImage.elements()));
	return af::clamp(outputImage + (u * watermarkStrength), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//Can also calculate error sequence and prediction error filter
af::array Watermark::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(meKernelBlockSize, meKernelDims.y, meKernelDims.x);
	cuda_utils::copyDataToCudaArray(image.device<float>(), dims.x, dims.y, texArray);
	//enqueue "x_" and prediction error mask kernel in two streams
	me_p3 <<<gridSize, meKernelBlockSize, 0, afStream >>> (texObj, RxPartial.device<float>(), rxPartial.device<float>(), dims.x, meKernelDims.x, dims.y);
	unlockArrays(RxPartial, rxPartial);

	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays();
	coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	const af::array x_ = computeScaledNeighbors(coefficients);
	errorSequence = image - x_;
	if (maskNeeded)
	{
		const af::array errorSequenceAbs = af::abs(errorSequence);
		return errorSequenceAbs / af::max<float>(errorSequenceAbs);
	}
	return af::array();
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::computeErrorSequence(const af::array& u, const af::array& coefficients) const 
{
	cuda_utils::copyDataToCudaArray(u.device<float>(), dims.x, dims.y, texArray);
	unlockArrays(u);
	return u - computeScaledNeighbors(coefficients);
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

//helper method to display an af::array in a window
void Watermark::displayArray(const af::array& array, const int width, const int height) 
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}