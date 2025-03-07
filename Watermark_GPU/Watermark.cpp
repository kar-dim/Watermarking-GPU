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
#define ALIGN_UP_16(x) (x + 15) & ~15
#define ALIGN_UP_64(x) (x + 63) & ~63

using std::string;

//initialize data and memory
Watermark::Watermark(const dim_t rows, const dim_t cols, const string randomMatrixPath, const int p, const float psnr, const std::vector<cl::Program>& programs)
	: dims({ rows, cols }), texKernelDims({ ALIGN_UP_16(rows), ALIGN_UP_16(cols) }), meKernelDims({ rows, ALIGN_UP_64(cols) }), programs(programs), p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	reinitialize(randomMatrixPath, rows, cols);
}

//copy constructor
Watermark::Watermark(const Watermark& other) 
	: dims(other.dims), texKernelDims(other.texKernelDims), meKernelDims(other.meKernelDims), programs(other.programs), p(other.p), strengthFactor(other.strengthFactor), randomMatrix(other.randomMatrix)
{
	initializeMemory();
}

//copy assignment operator
Watermark& Watermark::operator=(const Watermark& other)
{
	if (this != &other) 
	{
		dims = other.dims;
		texKernelDims = other.texKernelDims;
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
	texKernelDims = { ALIGN_UP_16(rows), ALIGN_UP_16(cols) };
	meKernelDims = { rows, ALIGN_UP_64(cols) };
	initializeMemory();
	loadRandomMatrix(randomMatrixPath);
}

//copy data to texture and transfer ownership back to arrayfire
void Watermark::copyDataToTexture(const af::array& image) const
{
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	cl_utils::copyBufferToImage(queue, image2d, imageMem.get(), dims.cols, dims.rows);
	unlockArrays(image);
}

//computes the custom mask (NVF).
af::array Watermark::computeCustomMask() const
{
	const af::array customMask(dims.rows, dims.cols);
	const std::unique_ptr<cl_mem> outputMem(customMask.device<cl_mem>());
	const int pad = p / 2;
	//execute kernel
	try {
		cl::Buffer buff(*outputMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[0],"nvf").args(image2d, buff, cl::Local(sizeof(float) * ( (16 + (2 * pad)) * (16 + (2 * pad)) ) )).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(customMask);
		return customMask;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in nvf: " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

//Computes scaled neighbors array, which calculates the dot product of the coefficients with the neighbors of each pixel
af::array Watermark::computeScaledNeighbors(const af::array& coefficients) const
{
	const af::array neighbors(dims.rows, dims.cols);
	const std::unique_ptr<cl_mem> coeffsMem(coefficients.device<cl_mem>());
	const std::unique_ptr<cl_mem> neighborsMem(neighbors.device<cl_mem>());
	//execute kernel
	try {
		cl::Buffer neighborsBuff(*neighborsMem.get(), true);
		cl::Buffer coeffsBuff(*coeffsMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[2], "scaled_neighbors_p3").args(image2d, neighborsBuff, coeffsBuff, cl::Local(sizeof(float) * 324)).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(coefficients, neighbors);
		return neighbors;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in scaled_neighbors_p3: " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
//and to transform them to the correct size, so that they can be used by the system solver
std::pair<af::array, af::array> Watermark::transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const
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
	af::array mask, errorSequence, coefficients;
	copyDataToTexture(inputImage);
	if (maskType == MASK_TYPE::ME)
	{
		mask = computePredictionErrorMask(inputImage, errorSequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
		//if the system is not solvable, don't waste time embeding the watermark, return output image without modification
		if (coefficients.elements() == 0)
			return outputImage;
	}
	else
		mask = computeCustomMask();
	const af::array u = mask * randomMatrix;
	watermarkStrength = strengthFactor / static_cast<float>(af::norm(u) / sqrt(inputImage.elements()));
	return af::clamp(outputImage + (u * watermarkStrength), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const af::array RxPartial(dims.rows, meKernelDims.cols);
	const af::array rxPartial(dims.rows, meKernelDims.cols / 8);
	const std::unique_ptr<cl_mem> RxPartialMem(RxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> rxPartialMem(rxPartial.device<cl_mem>());
	try {
		//initialize custom kernel memory
		cl::Buffer Rx_buff(*RxPartialMem.get(), true);
		cl::Buffer rx_buff(*rxPartialMem.get(), true);
		//call prediction error mask kernel
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[1], "me").args(image2d, Rx_buff, rx_buff, RxMappingsBuff,
			cl::Local(sizeof(cl_half) * 2304)).build(),
			cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(64, 1));
		//finish and return memory to arrayfire
		queue.finish();
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error(string("ERROR in compute_me_mask(): " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"));
	}

	unlockArrays(RxPartial, rxPartial);
	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays(RxPartial, rxPartial);
	//solve() may crash in OpenCL ArrayFire implementation if the system is not solvable.
	try {
		coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	}
	catch (const af::exception&) {
		coefficients = af::array(0, f32);
		return af::array();
	}
	//call scaled neighbors kernel and compute error sequence
	errorSequence = image - computeScaledNeighbors(coefficients);
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
	copyDataToTexture(u);
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
	copyDataToTexture(watermarkedImage);
	if (maskType == MASK_TYPE::NVF)
	{
		computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = computeCustomMask();
	}
	else
		mask = computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	//if the system is not solvable, don't waste time computing the correlation, there is no watermark
	if (coefficients.elements() == 0)
		return 0.0f;
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