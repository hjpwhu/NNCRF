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
include CMakeFiles/RNNLabeler.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RNNLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RNNLabeler.dir/flags.make

CMakeFiles/RNNLabeler.dir/RNNLabeler.o: CMakeFiles/RNNLabeler.dir/flags.make
CMakeFiles/RNNLabeler.dir/RNNLabeler.o: RNNLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RNNLabeler.dir/RNNLabeler.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RNNLabeler.dir/RNNLabeler.o -c /home/hjp/Downloads/NNCRF/RNNLabeler.cpp

CMakeFiles/RNNLabeler.dir/RNNLabeler.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RNNLabeler.dir/RNNLabeler.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hjp/Downloads/NNCRF/RNNLabeler.cpp > CMakeFiles/RNNLabeler.dir/RNNLabeler.i

CMakeFiles/RNNLabeler.dir/RNNLabeler.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RNNLabeler.dir/RNNLabeler.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hjp/Downloads/NNCRF/RNNLabeler.cpp -o CMakeFiles/RNNLabeler.dir/RNNLabeler.s

CMakeFiles/RNNLabeler.dir/RNNLabeler.o.requires:

.PHONY : CMakeFiles/RNNLabeler.dir/RNNLabeler.o.requires

CMakeFiles/RNNLabeler.dir/RNNLabeler.o.provides: CMakeFiles/RNNLabeler.dir/RNNLabeler.o.requires
	$(MAKE) -f CMakeFiles/RNNLabeler.dir/build.make CMakeFiles/RNNLabeler.dir/RNNLabeler.o.provides.build
.PHONY : CMakeFiles/RNNLabeler.dir/RNNLabeler.o.provides

CMakeFiles/RNNLabeler.dir/RNNLabeler.o.provides.build: CMakeFiles/RNNLabeler.dir/RNNLabeler.o


# Object files for target RNNLabeler
RNNLabeler_OBJECTS = \
"CMakeFiles/RNNLabeler.dir/RNNLabeler.o"

# External object files for target RNNLabeler
RNNLabeler_EXTERNAL_OBJECTS =

RNNLabeler: CMakeFiles/RNNLabeler.dir/RNNLabeler.o
RNNLabeler: CMakeFiles/RNNLabeler.dir/build.make
RNNLabeler: CMakeFiles/RNNLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RNNLabeler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RNNLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RNNLabeler.dir/build: RNNLabeler

.PHONY : CMakeFiles/RNNLabeler.dir/build

CMakeFiles/RNNLabeler.dir/requires: CMakeFiles/RNNLabeler.dir/RNNLabeler.o.requires

.PHONY : CMakeFiles/RNNLabeler.dir/requires

CMakeFiles/RNNLabeler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RNNLabeler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RNNLabeler.dir/clean

CMakeFiles/RNNLabeler.dir/depend:
	cd /home/hjp/Downloads/NNCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF/CMakeFiles/RNNLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RNNLabeler.dir/depend

