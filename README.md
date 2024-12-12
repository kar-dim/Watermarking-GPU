# ICSD thesis Part 2 / GPU Watermarking

![512](https://github.com/user-attachments/assets/6544f178-4f99-43ff-850c-9f40db478f35)


Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit". [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672)

The original Thesis code is in the <a href="https://github.com/kar-dim/Watermarking-GPU/tree/old">old</a> branch. The code was later refactored and the algorithms improved with much better execution times, now in the default <a href="https://github.com/kar-dim/Watermarking-GPU/tree/master">master</a> branch. There is a newer implementation with the CUDA framework, which works only on NVIDIA GPUs, and is slightly faster on NVIDIA GPUs: <a href="https://github.com/kar-dim/Watermarking-GPU/tree/cuda">CUDA branch</a>

# Overview

The aim of this project is to compare the performance (primarily execution speed) of watermarking algorithms when implemented on CPU versus GPU. This repository focuses on the GPU implementation (Part 2). The CPU implementation can be found in the corresponding CPU repository (Part 1 / CPU usage [here](https://github.com/kar-dim/Watermarking-CPU) ).

# Key Features

- Implementation of watermark embedding and detection algorithms for images.
- Comparative performance analysis between CPU and GPU implementations.

# Libraries Used

- [ArrayFire](https://arrayfire.org): A C++ library for fast GPU computing.
- [CImg](https://cimg.eu/): A C++ library for image processing.
- [inih](https://github.com/jtilly/inih): A lightweight C++ library for parsing .ini configuration files.

# Additional Dependencies

- OpenCL implementation: The [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL C++ Bindings](https://github.com/KhronosGroup/OpenCL-CLHPP) and [OpenCL Library file](https://github.com/KhronosGroup/OpenCL-SDK) are already included and configured for this project.
- CUDA implementation: NVIDIA CUDA Toolkit.
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- OpenCV (for video testing, used internally by CImg), with default installation options. Environment Variable "OPENCV_DIR" should be defined in the "build" directory (for example: C:\opencv\build\x64\vc16).
