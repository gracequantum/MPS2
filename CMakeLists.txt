# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 13:23
#  
#  Description: GraceQ/mps2 project. Main CMake file.
# 
cmake_minimum_required(VERSION 3.12)


project("GraceQ/mps2"
  DESCRIPTION "MPS-MPO algorithm library based on GraceQ/tensor."
  LANGUAGES CXX)


# Build type control.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()


# Build output control.
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


# C++ options.
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g)


# Build dependencies.
add_subdirectory(external)


# Build library.
add_subdirectory(src lib)


# Build unittests.
enable_testing()
add_subdirectory(tests tests)
