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
include CMakeFiles/GatedLabeler.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GatedLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GatedLabeler.dir/flags.make

CMakeFiles/GatedLabeler.dir/GatedLabeler.o: CMakeFiles/GatedLabeler.dir/flags.make
CMakeFiles/GatedLabeler.dir/GatedLabeler.o: GatedLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GatedLabeler.dir/GatedLabeler.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GatedLabeler.dir/GatedLabeler.o -c /home/hjp/Downloads/NNCRF/GatedLabeler.cpp

CMakeFiles/GatedLabeler.dir/GatedLabeler.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GatedLabeler.dir/GatedLabeler.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hjp/Downloads/NNCRF/GatedLabeler.cpp > CMakeFiles/GatedLabeler.dir/GatedLabeler.i

CMakeFiles/GatedLabeler.dir/GatedLabeler.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GatedLabeler.dir/GatedLabeler.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hjp/Downloads/NNCRF/GatedLabeler.cpp -o CMakeFiles/GatedLabeler.dir/GatedLabeler.s

CMakeFiles/GatedLabeler.dir/GatedLabeler.o.requires:

.PHONY : CMakeFiles/GatedLabeler.dir/GatedLabeler.o.requires

CMakeFiles/GatedLabeler.dir/GatedLabeler.o.provides: CMakeFiles/GatedLabeler.dir/GatedLabeler.o.requires
	$(MAKE) -f CMakeFiles/GatedLabeler.dir/build.make CMakeFiles/GatedLabeler.dir/GatedLabeler.o.provides.build
.PHONY : CMakeFiles/GatedLabeler.dir/GatedLabeler.o.provides

CMakeFiles/GatedLabeler.dir/GatedLabeler.o.provides.build: CMakeFiles/GatedLabeler.dir/GatedLabeler.o


# Object files for target GatedLabeler
GatedLabeler_OBJECTS = \
"CMakeFiles/GatedLabeler.dir/GatedLabeler.o"

# External object files for target GatedLabeler
GatedLabeler_EXTERNAL_OBJECTS =

GatedLabeler: CMakeFiles/GatedLabeler.dir/GatedLabeler.o
GatedLabeler: CMakeFiles/GatedLabeler.dir/build.make
GatedLabeler: CMakeFiles/GatedLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hjp/Downloads/NNCRF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GatedLabeler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GatedLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GatedLabeler.dir/build: GatedLabeler

.PHONY : CMakeFiles/GatedLabeler.dir/build

CMakeFiles/GatedLabeler.dir/requires: CMakeFiles/GatedLabeler.dir/GatedLabeler.o.requires

.PHONY : CMakeFiles/GatedLabeler.dir/requires

CMakeFiles/GatedLabeler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GatedLabeler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GatedLabeler.dir/clean

CMakeFiles/GatedLabeler.dir/depend:
	cd /home/hjp/Downloads/NNCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF /home/hjp/Downloads/NNCRF/CMakeFiles/GatedLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GatedLabeler.dir/depend

