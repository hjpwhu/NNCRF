# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hjp/Downloads/NNCRF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hjp/Downloads/NNCRF

# Include any dependencies generated for this target.
include CMakeFiles/SparseRNNLabeler.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SparseRNNLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SparseRNNLabeler.dir/flags.make

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o: CMakeFiles/SparseRNNLabeler.dir/flags.make
CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o: SparseRNNLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o -c /home/hjp/Downloads/NNCRF/SparseRNNLabeler.cpp

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hjp/Downloads/NNCRF/SparseRNNLabeler.cpp > CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.i

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hjp/Downloads/NNCRF/SparseRNNLabeler.cpp -o CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.s

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.requires:

.PHONY : CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.requires

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.provides: CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.requires
	$(MAKE) -f CMakeFiles/SparseRNNLabeler.dir/build.make CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.provides.build
.PHONY : CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.provides

CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.provides.build: CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o


# Object files for target SparseRNNLabeler
SparseRNNLabeler_OBJECTS = \
"CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o"

# External object files for target SparseRNNLabeler
SparseRNNLabeler_EXTERNAL_OBJECTS =

SparseRNNLabeler: CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o
SparseRNNLabeler: CMakeFiles/SparseRNNLabeler.dir/build.make
SparseRNNLabeler: CMakeFiles/SparseRNNLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SparseRNNLabeler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SparseRNNLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SparseRNNLabeler.dir/build: SparseRNNLabeler

.PHONY : CMakeFiles/SparseRNNLabeler.dir/build

CMakeFiles/SparseRNNLabeler.dir/requires: CMakeFiles/SparseRNNLabeler.dir/SparseRNNLabeler.o.requires

.PHONY : CMakeFiles/SparseRNNLabeler.dir/requires

CMakeFiles/SparseRNNLabeler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SparseRNNLabeler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SparseRNNLabeler.dir/clean

CMakeFiles/SparseRNNLabeler.dir/depend:
	cd /home/hjp/Downloads/NNCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF/CMakeFiles/SparseRNNLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SparseRNNLabeler.dir/depend

