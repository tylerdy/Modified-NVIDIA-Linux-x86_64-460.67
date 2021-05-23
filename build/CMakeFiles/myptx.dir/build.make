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
CMAKE_SOURCE_DIR = /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build

# Include any dependencies generated for this target.
include CMakeFiles/myptx.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/myptx.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/myptx.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/myptx.dir/flags.make

CMakeFiles/myptx.dir/stress.ptx: CMakeFiles/myptx.dir/flags.make
CMakeFiles/myptx.dir/stress.ptx: ../stress.cu
CMakeFiles/myptx.dir/stress.ptx: CMakeFiles/myptx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/myptx.dir/stress.ptx"
	/usr/local/cuda-11.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/myptx.dir/stress.ptx -MF CMakeFiles/myptx.dir/stress.ptx.d -x cu -ptx /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/stress.cu -o CMakeFiles/myptx.dir/stress.ptx

CMakeFiles/myptx.dir/stress.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/myptx.dir/stress.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/myptx.dir/stress.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/myptx.dir/stress.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

myptx: CMakeFiles/myptx.dir/stress.ptx
myptx: CMakeFiles/myptx.dir/build.make
.PHONY : myptx

# Rule to build all files generated by this target.
CMakeFiles/myptx.dir/build: myptx
.PHONY : CMakeFiles/myptx.dir/build

CMakeFiles/myptx.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/myptx.dir/cmake_clean.cmake
.PHONY : CMakeFiles/myptx.dir/clean

CMakeFiles/myptx.dir/depend:
	cd /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles/myptx.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/myptx.dir/depend

