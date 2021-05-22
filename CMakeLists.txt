set(TARGET sm_61)

# TODO: Which version do we want?
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
enable_language(CUDA)
add_library(myptx OBJECT stress.cu stress_kernel.cuh)
set_property(TARGET myptx PROPERTY CUDA_PTX_COMPILATION ON)

SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
set( CMAKE_VERBOSE_MAKEFILE on )
project(fractional_gpu VERSION 1.0.1 LANGUAGES CXX CUDA DESCRIPTION "Split GPU into Fractional GPUs")


# Configure the build type
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(./)
include_directories(NVIDIA-Linux-x86_64-460.67/kernel/nvidia-uvm/)
include_directories(NVIDIA-Linux-x86_64-460.67/kernel/common/inc/)
include_directories(/usr/local/cuda-11.1/samples/common/inc)
# string(APPEND CUDA_NVCC_FLAGS "-Xptxas -dlcm=cg")
string(APPEND CMAKE_CUDA_FLAGS "-m 64 -arch=${TARGET} -Xptxas -dlcm=cg")
# string(REPLACE "O3" "O0" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
#string(APPEND CUDA_NVCC_FLAGS "-O0")
# Add programs to be built using persistent threads model
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_persistent_target target include_dir)

    add_executable(${target} ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} fractional_gpu cuda cudart nvidia-ml)
    target_include_directories(${target}  PUBLIC ${include_dir})

endfunction(add_persistent_target)

# Add programs to be built _without_ using persistent threads model
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_native_target target include_dir)

    add_executable(${target} ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_include_directories(${target}  PUBLIC ${include_dir})

endfunction(add_native_target)

# Library
add_library(fractional_gpu SHARED
    memory.cu
    allocator.cpp
)
set_property(TARGET fractional_gpu PROPERTY VERSION ${PROJECT_VERSION})
target_compile_features(fractional_gpu PUBLIC cxx_std_11)
target_link_libraries(fractional_gpu cuda cudart nvidia-ml)

#add_persistent_target(gpu_reverse_engineering .
#       gpu.cu
#       reverse_engineering.cpp
#       hash_function.cpp
#       )

add_persistent_target(stress .
        gpu.cu
        stress.cu
        )

#message("-----------------------------------------------\n
#        ${CMAKE_CUDA_FLAGS_RELEASE}\n-----------------------------------------------")
