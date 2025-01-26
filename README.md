# ICSD thesis Part 2 / GPU Watermarking

![512](https://github.com/user-attachments/assets/6544f178-4f99-43ff-850c-9f40db478f35)


Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit". [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672)

The original Thesis code is in the <a href="https://github.com/kar-dim/Watermarking-GPU/tree/old">old</a> branch. The code was later refactored and the algorithms improved with much better execution times, now in the default <a href="https://github.com/kar-dim/Watermarking-GPU/tree/master">master</a> branch. There is a newer implementation with the CUDA framework, which works only on NVIDIA GPUs, and is slightly faster on NVIDIA GPUs: <a href="https://github.com/kar-dim/Watermarking-GPU/tree/cuda">CUDA branch</a>

# Overview

The aim of this project is to compare the performance (primarily execution speed) of watermarking algorithms when implemented on CPU versus GPU. This repository focuses on the GPU implementation (Part 2). The CPU implementation can be found in the corresponding CPU repository (Part 1 / CPU usage [here](https://github.com/kar-dim/Watermarking-CPU) ).

# Key Features

- Implementation of watermark embedding and detection algorithms for images.
- Comparative performance analysis between CPU and GPU implementations.

# Run the pre-build binaries
- Get the latest binaries [here](https://github.com/kar-dim/Watermarking-GPU/releases) for OpenCL or CUDA platform. The binary contains the sample application and the CUDA kernels (OpenCL builds the kernels at runtime, so the kernels are provided in the corresponding folder). Before we can emded the watermark, we have to create it first.
- This implementation is based on Normal-distributed random values with zero mean and standard deviation of one. The ```CommonRandomMatrix``` produces pseudo-random values. A bat file is included to generate the watermarks, with sizes exactly the same as the provided sample images. Of course, one can generate a random watermark for any desired image size like this:  
```CommonRandomMatrix.exe [rows] [cols] [seed] [fileName]``` .Then pass the provided watermark file path in the sample project configuration.

The sample application:
   - Embeds the watermark using the NVF and the proposed Prediction-Error mask.
   - Detects the watermark using the proposed Prediction-Error based detector.
   - Prints FPS for both operations, and both masks.
Needs to be parameterized from the corresponding ```settings.ini``` file. Here is a detailed explanation for each parameter:

| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------               |
| image                             | Path to the input image.                                                                                                                   |
| w_path                            | Path to the Random Matrix (watermark). This is produced by the CommonRandomMatrix project. Watermark and Image sizes should match exactly. |
| video                             | Path to the video file, if we want to embed the watermark for a raw YUV video.                                                             |
| save_watermarked_files_to_disk    | \[true/false\]: Set to true to save the watermarked NVF and Prediction-Error files to disk.                                                |
| execution_time_in_fps             | \[true/false\]: Set to true to display execution times in FPS. Else, it will display execution time in seconds.                            |
| p                                 | Window size for masking algorithms. Currently only ```p=3``` is allowed.                                                                         |
| psnr                              | PSNR (Peak Signal-to-Noise Ratio). Higher values correspond to less watermark in the image, reducing noise, but making detection harder.   |
| loops_for_test                    | Loops the algorithms many times, simulating more work. A value of 1000 produces almost identical execution times.                          |
| test_for_video                    | \[true/false\]: If set to true, the sample will work for videos only, else it will work for images.                                        |


**Video-only settings:**


| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------               |
| display_frames                    | \[true/false\]: Set to true to display the video frames as they are processed. This "simulates" video playback to display frames based on the original video's FPS. |
| watermark_make                    | \[true/false\]: Set to true to embed the watermark in the raw YUV video provided by the ```video``` parameter.                              |
| watermark_first_frame_only        | \[true/false\]: Set to true to embed the watermark only in the first frame. Mostly obsolete after the algorithms' execution speed were improved. Useful for inspecting how long the watermark "survives" compression. |
| watermark_detection               | \[true/false\]: Set to true to try to detect the watermark created with the ```watermark_make``` flag (works only if ```watermark_make = true``` .|
| rows                              | Height of the raw YUV video file frames (raw YUV lacks any metadata like rows, columns, fps and total frame count).                         |
| cols                              | Width of the raw YUV video file frames (raw YUV lacks any metadata like rows, columns, fps and total frame count).                          |
| frames                            | Total frame count of the raw YUV video file frames (raw YUV lacks any metadata like rows, columns, fps and total frame count).              |
| fps                               | FPS of the raw YUV video file frames (raw YUV lacks any metadata like rows, columns, fps and total frame count).                            |


# Libraries Used

- [ArrayFire](https://arrayfire.org): A C++ library for fast GPU computing.
- [CImg](https://cimg.eu/): A C++ library for image processing.
- [inih](https://github.com/jtilly/inih): A lightweight C++ library for parsing .ini configuration files.

# Additional Dependencies

- OpenCL implementation: The [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL C++ Bindings](https://github.com/KhronosGroup/OpenCL-CLHPP) and [OpenCL Library file](https://github.com/KhronosGroup/OpenCL-SDK) are already included and configured for this project.
- CUDA implementation: NVIDIA CUDA Toolkit.
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- OpenCV (for video testing, used internally by CImg), with default installation options. Environment Variable "OPENCV_DIR" should be defined in the "build" directory (for example: C:\opencv\build\x64\vc16).
