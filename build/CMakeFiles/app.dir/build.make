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
CMAKE_SOURCE_DIR = /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build

# Include any dependencies generated for this target.
include CMakeFiles/app.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/app.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/app.dir/flags.make

CMakeFiles/app.dir/gpu.cu.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/gpu.cu.o: ../gpu.cu
CMakeFiles/app.dir/gpu.cu.o: CMakeFiles/app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/app.dir/gpu.cu.o"
	/usr/local/cuda-11.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/app.dir/gpu.cu.o -MF CMakeFiles/app.dir/gpu.cu.o.d -x cu -c /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/gpu.cu -o CMakeFiles/app.dir/gpu.cu.o

CMakeFiles/app.dir/gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/app.dir/gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/app.dir/gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/app.dir/gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/app.dir/app.cu.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/app.cu.o: ../app.cu
CMakeFiles/app.dir/app.cu.o: CMakeFiles/app.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/app.dir/app.cu.o"
	/usr/local/cuda-11.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/app.dir/app.cu.o -MF CMakeFiles/app.dir/app.cu.o.d -x cu -c /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/app.cu -o CMakeFiles/app.dir/app.cu.o

CMakeFiles/app.dir/app.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/app.dir/app.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/app.dir/app.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/app.dir/app.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target app
app_OBJECTS = \
"CMakeFiles/app.dir/gpu.cu.o" \
"CMakeFiles/app.dir/app.cu.o"

# External object files for target app
app_EXTERNAL_OBJECTS =

app: CMakeFiles/app.dir/gpu.cu.o
app: CMakeFiles/app.dir/app.cu.o
app: CMakeFiles/app.dir/build.make
app: libfractional_gpu.so.1.0.1
app: CMakeFiles/app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/app.dir/build: app
.PHONY : CMakeFiles/app.dir/build

CMakeFiles/app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/app.dir/clean

CMakeFiles/app.dir/depend:
	cd /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67 /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build /playpen/tylerdy/Modified-NVIDIA-Linux-x86_64-460.67/build/CMakeFiles/app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/app.dir/depend

