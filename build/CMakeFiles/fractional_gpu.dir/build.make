# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build

# Include any dependencies generated for this target.
include CMakeFiles/fractional_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/fractional_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/fractional_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fractional_gpu.dir/flags.make

CMakeFiles/fractional_gpu.dir/memory.cu.o: CMakeFiles/fractional_gpu.dir/flags.make
CMakeFiles/fractional_gpu.dir/memory.cu.o: ../memory.cu
CMakeFiles/fractional_gpu.dir/memory.cu.o: CMakeFiles/fractional_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/fractional_gpu.dir/memory.cu.o"
	/usr/local/cuda-11.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/fractional_gpu.dir/memory.cu.o -MF CMakeFiles/fractional_gpu.dir/memory.cu.o.d -x cu -c /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/memory.cu -o CMakeFiles/fractional_gpu.dir/memory.cu.o

CMakeFiles/fractional_gpu.dir/memory.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/fractional_gpu.dir/memory.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/fractional_gpu.dir/memory.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/fractional_gpu.dir/memory.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/fractional_gpu.dir/allocator.cpp.o: CMakeFiles/fractional_gpu.dir/flags.make
CMakeFiles/fractional_gpu.dir/allocator.cpp.o: ../allocator.cpp
CMakeFiles/fractional_gpu.dir/allocator.cpp.o: CMakeFiles/fractional_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fractional_gpu.dir/allocator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fractional_gpu.dir/allocator.cpp.o -MF CMakeFiles/fractional_gpu.dir/allocator.cpp.o.d -o CMakeFiles/fractional_gpu.dir/allocator.cpp.o -c /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/allocator.cpp

CMakeFiles/fractional_gpu.dir/allocator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fractional_gpu.dir/allocator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/allocator.cpp > CMakeFiles/fractional_gpu.dir/allocator.cpp.i

CMakeFiles/fractional_gpu.dir/allocator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fractional_gpu.dir/allocator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/allocator.cpp -o CMakeFiles/fractional_gpu.dir/allocator.cpp.s

# Object files for target fractional_gpu
fractional_gpu_OBJECTS = \
"CMakeFiles/fractional_gpu.dir/memory.cu.o" \
"CMakeFiles/fractional_gpu.dir/allocator.cpp.o"

# External object files for target fractional_gpu
fractional_gpu_EXTERNAL_OBJECTS =

libfractional_gpu.so.1.0.1: CMakeFiles/fractional_gpu.dir/memory.cu.o
libfractional_gpu.so.1.0.1: CMakeFiles/fractional_gpu.dir/allocator.cpp.o
libfractional_gpu.so.1.0.1: CMakeFiles/fractional_gpu.dir/build.make
libfractional_gpu.so.1.0.1: CMakeFiles/fractional_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libfractional_gpu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fractional_gpu.dir/link.txt --verbose=$(VERBOSE)
	$(CMAKE_COMMAND) -E cmake_symlink_library libfractional_gpu.so.1.0.1 libfractional_gpu.so.1.0.1 libfractional_gpu.so

libfractional_gpu.so: libfractional_gpu.so.1.0.1
	@$(CMAKE_COMMAND) -E touch_nocreate libfractional_gpu.so

# Rule to build all files generated by this target.
CMakeFiles/fractional_gpu.dir/build: libfractional_gpu.so
.PHONY : CMakeFiles/fractional_gpu.dir/build

CMakeFiles/fractional_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fractional_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fractional_gpu.dir/clean

CMakeFiles/fractional_gpu.dir/depend:
	cd /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/leochanj/fgpure/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles/fractional_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fractional_gpu.dir/depend

