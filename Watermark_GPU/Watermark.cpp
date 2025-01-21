#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "Watermark.hpp"
#include <arrayfire.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using std::string;

//initialize data and memory
Watermark::Watermark(const dim_t rows, const dim_t cols, const string randomMatrixPath, const int p, const float psnr, const std::vector<cl::Program>& programs)
	: dims({ rows, cols }), meKernelDims({ rows, (cols + 63) & ~63 }), programs(programs), p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	reinitialize(randomMatrixPath, rows, cols);
}

//copy constructor
Watermark::Watermark(const Watermark& other) 
	: dims(other.dims), meKernelDims(other.meKernelDims), programs(other.programs), p(other.p), strengthFactor(other.strengthFactor), randomMatrix(other.randomMatrix)
{
	initializeMemory();
}

//copy assignment operator
Watermark& Watermark::operator=(const Watermark& other)
{
	if (this != &other) 
	{
		dims = other.dims;
		meKernelDims = other.meKernelDims;
		programs = other.programs;
		p = other.p;
		strengthFactor = other.strengthFactor;
		randomMatrix = other.randomMatrix;
		initializeMemory();
	}
	return *this;
}

//supply the input image size, and pre-allocate buffers and arrays
void Watermark::initializeMemory() 
{
	//initialize texture (transposed dimensions, arrayfire is column wise, we skip an extra transpose)
	image2d = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), dims.rows, dims.cols, 0, NULL);
	//allocate memory (Rx/rx partial sums and custom maks output) to avoid constant cudaMalloc
	RxPartial = af::array(dims.rows, meKernelDims.cols);
	rxPartial = af::array(dims.rows, meKernelDims.cols / 8);
	customMask = af::array(dims.rows, dims.cols);
	neighbors = af::array(dims.rows, dims.cols);
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
	if (dims.rows * dims.cols * sizeof(float) != totalBytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(totalBytes / (sizeof(float))) + ", Image width: " + std::to_string(dims.cols) + ", Image height: " + std::to_string(dims.rows) + "\n"));
	std::unique_ptr<float> wPtr(new float[dims.rows * dims.cols]);
	randomMatrixStream.read(reinterpret_cast<char*>(wPtr.get()), totalBytes);
	randomMatrix = af::transpose(af::array(dims.cols, dims.rows, wPtr.get()));
}

//re-initializes memory (texture, kernel arrays, random matrix array) for new image size
void Watermark::reinitialize(const string randomMatrixPath, const dim_t rows, const dim_t cols)
{
	dims = { rows, cols };
	meKernelDims = { rows, (cols + 63) & ~63 };
	initializeMemory();
	loadRandomMatrix(randomMatrixPath);
}

//can be called for computing a custom mask, or for a neighbors (x_) array, depending on the cl::Program param and kernel name
af::array Watermark::computeCustomMask(const af::array& image) const
{
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	const std::unique_ptr<cl_mem> outputMem(customMask.device<cl_mem>());
	//copy to texture cache and execute kernel
	try {
		cl_utils::copyBufferToImage(queue, image2d, imageMem.get(), dims.cols, dims.rows);
		cl::Buffer buff(*outputMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[0],"nvf").args(image2d, buff).build(),
			cl::NDRange(), cl::NDRange(dims.rows, dims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image, customMask);
		return customMask;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in nvf: " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

//can be called for computing a custom mask, or for a neighbors (x_) array, depending on the cl::Program param and kernel name
af::array Watermark::computeScaledNeighbors(const af::array& coefficients) const
{
	const std::unique_ptr<cl_mem> coeffsMem(coefficients.device<cl_mem>());
	const std::unique_ptr<cl_mem> neighborsMem(neighbors.device<cl_mem>());
	//execute kernel
	try {
		cl::Buffer neighborsBuff(*neighborsMem.get(), true);
		cl::Buffer coeffsBuff(*coeffsMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[2], "calculate_scaled_neighbors_p3").args(image2d, neighborsBuff, coeffsBuff).build(),
			cl::NDRange(), cl::NDRange(dims.rows, dims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(coefficients, neighbors);
		return neighbors;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in calculate_scaled_neighbors_p3: " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
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
//it embeds the watermark computed fom "inputImage" (always grayscale)
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
//can also calculate error sequence and prediction error filter
af::array Watermark::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const std::unique_ptr<cl_mem> RxPartialMem(RxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> rxPartialMem(rxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	try {
		//copy image to texture cache
		cl_utils::copyBufferToImage(queue, image2d, imageMem.get(), dims.cols, dims.rows);
		//initialize custom kernel memory
		cl::Buffer Rx_buff(*RxPartialMem.get(), true);
		cl::Buffer rx_buff(*rxPartialMem.get(), true);
		//call prediction error mask kernel
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[1], "me").args(image2d, Rx_buff, rx_buff, RxMappingsBuff,
			cl::Local(sizeof(float) * 2304)).build(),
			cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(64, 1));
		//finish and return memory to arrayfire
		queue.finish();
		unlockArrays(RxPartial, rxPartial);
		//calculation of coefficients, error sequence and mask
		const auto correlationArrays = transformCorrelationArrays();
		coefficients = af::solve(correlationArrays.first, correlationArrays.second);
		//enqueue "x_" kernel
		errorSequence = image - computeScaledNeighbors(coefficients);
		if (maskNeeded) 
		{
			const af::array errorSequenceAbs = af::abs(errorSequence);
			return errorSequenceAbs / af::max<float>(errorSequenceAbs);
		}
		return af::array();
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error(string("ERROR in compute_me_mask(): " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"));
	}
}

//helper method that calculates the error sequence by using a supplied prediction filter coefficients
af::array Watermark::computeErrorSequence(const af::array& u, const af::array& coefficients) const 
{
	const std::unique_ptr<cl_mem> uMem(u.device<cl_mem>());
	cl_utils::copyBufferToImage(queue, image2d, uMem.get(), dims.cols, dims.rows);
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
	af::array mask, errorSequenceW, coefficients;
	if (maskType == MASK_TYPE::NVF)
	{
		computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = computeCustomMask(watermarkedImage);
	}
	else
		mask = computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	const af::array u = mask * randomMatrix;
	return computeCorrelation(computeErrorSequence(u, coefficients), errorSequenceW);
}

//helper method to display an af::array in a window
void Watermark::displayArray(const af::array& array, const int width, const int height) 
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}