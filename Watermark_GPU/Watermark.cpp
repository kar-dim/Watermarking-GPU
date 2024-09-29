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
		:programs(programs), p(p), strengthFactor((255.0f / sqrt(pow(10.0f, psnr / 10.0f))))
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	reinitialize(randomMatrixPath, rows, cols);
}

//copy constructor
Watermark::Watermark(const Watermark& other) : programs(other.programs), p(other.p), strengthFactor(other.strengthFactor), randomMatrix(other.randomMatrix)
{
	initializeMemory(other.customMask.dims(0), other.customMask.dims(1));
}

//copy assignment operator
Watermark& Watermark::operator=(const Watermark& other)
{
	if (this != &other) 
	{
		programs = other.programs;
		p = other.p;
		strengthFactor = other.strengthFactor;
		randomMatrix = other.randomMatrix;
		initializeMemory(other.customMask.dims(0), other.customMask.dims(1));
	}
	return *this;
}

//supply the input image size, and pre-allocate buffers and arrays
void Watermark::initializeMemory(const dim_t rows, const dim_t cols) 
{
	//initialize texture
	image2d = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), cols, rows, 0, NULL);
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

//re-initializes memory (texture, kernel arrays, random matrix array) for new image size
void Watermark::reinitialize(const string randomMatrixPath, const dim_t rows, const dim_t cols)
{
	initializeMemory(rows, cols);
	loadRandomMatrix(randomMatrixPath, rows, cols);
}

//can be called for computing a custom mask, or for a neighbors (x_) array, depending on the cl::Program param and kernel name
af::array Watermark::executeTextureKernel(const af::array& image, const cl::Program& program, const string kernelName, const af::array& output) const
{
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	const auto pad_rows = (rows % 16 == 0) ? rows : rows + 16 - (rows % 16);
	const auto pad_cols = (cols % 16 == 0) ? cols : cols + 16 - (cols % 16);
	const af::array image_transpose = image.T();
	const std::unique_ptr<cl_mem> imageT_ptr(image_transpose.device<cl_mem>());
	const std::unique_ptr<cl_mem> output_ptr(output.device<cl_mem>());

	//copy to texture cache and execute kernel
	try {
		cl_utils::copyBufferToImage(queue, image2d, imageT_ptr.get(), rows, cols);
		cl::Buffer buff(*output_ptr.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(program, kernelName.c_str()).args(image2d, buff).build(),
			cl::NDRange(), cl::NDRange(pad_rows, pad_cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image_transpose, output);
		return output;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in " + kernelName + ": " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

//helper method to sum the incomplete RxPartial and rxPartial arrays which were produced from the custom kernel
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
		executeTextureKernel(inputImage, programs[0], "nvf", customMask);
	const af::array u = mask * randomMatrix;
	a = strengthFactor / sqrt(af::sum<float>(af::pow(u, 2)) / (inputImage.elements()));
	return af::clamp(outputImage + (u * a), 0, 255);
}

//Compute prediction error mask. Used in both creation and detection of the watermark.
//can also calculate error sequence and prediction error filter
af::array Watermark::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const unsigned int rows = static_cast<unsigned int>(image.dims(0));
	const unsigned int cols = static_cast<unsigned int>(image.dims(1));
	//fix for OpenCL 1.2 limitation: GlobalGroupSize % LocalGroupSize should be 0, so we pad GlobalGroupSize (cols)
	const auto padded_cols = (cols % 64 == 0) ? cols : cols + 64 - (cols % 64);
	const std::unique_ptr<cl_mem> Rx_partial_mem(RxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> rx_partial_mem(rxPartial.device<cl_mem>());
	try {
		//enqueue "x_" kernel (which is heavy)
		const af::array x_ = executeTextureKernel(image, programs[2], "calculate_neighbors_p3", neighbors);
		//initialize custom kernel memory
		cl::Buffer Rx_buff(*Rx_partial_mem.get(), true);
		cl::Buffer rx_buff(*rx_partial_mem.get(), true);
		//call prediction error mask kernel
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[1], "me").args(image2d, Rx_buff, rx_buff, RxMappingsBuff,
			cl::Local(sizeof(float) * 2304), cl::Local(sizeof(float) * 512), cl::Local(sizeof(float) * 64)).build(),
			cl::NDRange(), cl::NDRange(rows, padded_cols), cl::NDRange(1, 64));
		//finish and return memory to arrayfire
		queue.finish();
		unlockArrays(RxPartial, rxPartial);
		//calculation of coefficients, error sequence and mask
		const auto correlation_arrays = transformCorrelationArrays();
		coefficients = af::solve(correlation_arrays.first, correlation_arrays.second);
		errorSequence = af::moddims(af::flat(image).T() - af::matmulTT(coefficients, x_), rows, cols);
		if (maskNeeded) 
		{
			const af::array error_sequence_abs = af::abs(errorSequence);
			return error_sequence_abs / af::max<float>(error_sequence_abs);
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
	return af::moddims(af::flat(u).T() - af::matmulTT(coefficients, executeTextureKernel(u, programs[2], "calculate_neighbors_p3", neighbors)), u.dims(0), u.dims(1));
}

//overloaded, fast mask calculation by using a supplied prediction filter
af::array Watermark::computePredictionErrorMask(const af::array& image, const af::array& coefficients, af::array& errorSequence) const
{
	errorSequence = computeErrorSequence(image, coefficients);
	const af::array error_sequence_abs = af::abs(errorSequence);
	return error_sequence_abs / af::max<float>(error_sequence_abs);
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
		mask = executeTextureKernel(watermarkedImage, programs[0], "nvf", customMask);
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