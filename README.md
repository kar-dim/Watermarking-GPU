# ICSD thesis Part 2 / GPU Watermarking

Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit". [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672)

The original Thesis code is in the <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/old">old</a> branch. The code was later refactored and the algorithms improved with much better execution times, now in the default <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/master">master</a> branch. There is a newer implementation with the CUDA framework, which works only on NVIDIA GPUs, and is slightly faster on NVIDIA GPUs: <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/cuda">CUDA branch</a>


This Diploma thesis aims to compare the above algorithms (mainly in execution speed) when they are implemented in GPU and CPU.
Part 2 / GPU usage for calculations. (Part 1 / CPU usage [here](https://github.com/kar-dim/diploma-thesis_CPU) )

Libraries used:
- [ArrayFire](https://arrayfire.org)
- [CImg](https://cimg.eu/)
- [inih](https://github.com/jtilly/inih)
    
For building the project, the below must be installed:
- OpenCL implementation: The [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL C++ Bindings](https://github.com/KhronosGroup/OpenCL-CLHPP) and [OpenCL Library file](https://github.com/KhronosGroup/OpenCL-SDK) are already included and configured for this project.
- CUDA implementation: NVIDIA CUDA Toolkit.
- LibPNG (can be installed from Visual Studio marketplace).
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- OpenCV (for video testing, used internally by CImg), with default installation options. Environment Variable "OPENCV_DIR" will be defined automatically.
