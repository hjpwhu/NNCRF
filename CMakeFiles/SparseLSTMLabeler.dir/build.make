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
include CMakeFiles/SparseLSTMLabeler.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SparseLSTMLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SparseLSTMLabeler.dir/flags.make

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o: CMakeFiles/SparseLSTMLabeler.dir/flags.make
CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o: SparseLSTMLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o -c /home/hjp/Downloads/NNCRF/SparseLSTMLabeler.cpp

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hjp/Downloads/NNCRF/SparseLSTMLabeler.cpp > CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.i

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hjp/Downloads/NNCRF/SparseLSTMLabeler.cpp -o CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.s

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.requires:

.PHONY : CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.requires

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.provides: CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.requires
	$(MAKE) -f CMakeFiles/SparseLSTMLabeler.dir/build.make CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.provides.build
.PHONY : CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.provides

CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.provides.build: CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o


# Object files for target SparseLSTMLabeler
SparseLSTMLabeler_OBJECTS = \
"CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o"

# External object files for target SparseLSTMLabeler
SparseLSTMLabeler_EXTERNAL_OBJECTS =

SparseLSTMLabeler: CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o
SparseLSTMLabeler: CMakeFiles/SparseLSTMLabeler.dir/build.make
SparseLSTMLabeler: CMakeFiles/SparseLSTMLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SparseLSTMLabeler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SparseLSTMLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SparseLSTMLabeler.dir/build: SparseLSTMLabeler

.PHONY : CMakeFiles/SparseLSTMLabeler.dir/build

CMakeFiles/SparseLSTMLabeler.dir/requires: CMakeFiles/SparseLSTMLabeler.dir/SparseLSTMLabeler.o.requires

.PHONY : CMakeFiles/SparseLSTMLabeler.dir/requires

CMakeFiles/SparseLSTMLabeler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SparseLSTMLabeler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SparseLSTMLabeler.dir/clean

CMakeFiles/SparseLSTMLabeler.dir/depend:
	cd /home/hjp/Downloads/NNCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF/CMakeFiles/SparseLSTMLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SparseLSTMLabeler.dir/depend

