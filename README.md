# Modified-NVIDIA-Linux-x86_64-460.67

This is the repository for RTSS22 submission "Making Powerful Enemies on NVIDIA GPUs"

Severl data files (fc6.txt & fc7.txt) for AlexNet benchmarks were removed due to file size limit imposed by Git.

naive-stressors/: baseline stressors

eviction-victims/: victim kernels we used

stress.cu, stress_kernel.cuh: codes for configurable enemy kernels

NVIDIA-Linux-x86_64-460.67/: modified driver codes to support contiguous memory allocation

allocator.cpp, gpu.cu, hash_function.cpp, hash_function.hpp, memory.cu, 
reverse_engineering.cpp, reverse_engineering.hpp, userlib.hpp, test.hpp : codes for user space support to ensure contig memory (adapted from FGPU).

*.sh: scripts for launching experiments & collecting results.
