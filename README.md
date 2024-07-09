# ICSD thesis Part 2 / GPU Watermarking

Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit".

The original Thesis code is in the <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/old">old</a> branch. The code was later refactored and the algorithms improved with much better execution times, now in the default <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/master">master</a> branch. There is a newer implementation with the CUDA framework, which works only on NVIDIA GPUs, which is even faster: <a href="https://github.com/kar-dim/diploma-thesis_GPU/tree/cuda">CUDA branch</a>


This Diploma thesis aims to compare the above algorithms (mainly in execution speed) when they are implemented in GPU and CPU.
Part 2 / GPU usage for calculations. (Part 1 / CPU usage [here](https://github.com/kar-dim/diploma-thesis_CPU) )

Libraries used:
- [ArrayFire](https://arrayfire.org)
- [CImg](https://cimg.eu/)
- [inih](https://github.com/jtilly/inih)
    
For building the project, the below must be installed:
- OpenCL implementation: AMD APP SDK 3.0 (or Nvidia CUDA Toolkit which includes OpenCL support for NVIDIA GPUs). Must be included in default include path. If AMD APP SDK 3.0 is used, occurrences of "cl.hpp" in code should be replaced with "cl2.hpp"
- CUDA implementation: NVIDIA CUDA Toolkit.
- LibPNG (can be installed from Visual Studio marketplace).
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- OpenCV (for video testing, used internally by CImg), with default installation options. Environment Variable "OPENCV_DIR" will be defined automatically.
