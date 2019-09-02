#  SPDX-License-Identifier: LGPL-3.0-only
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
  set(CMAKE_BUILD_TYPE Release)
endif()


# Build output control.
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


# Advanced options.
option(GQMPS2_TIMING_MODE "Timing mode." OFF)
if(GQMPS2_TIMING_MODE)
  add_definitions(-DGQMPS2_TIMING_MODE)
endif()

option(GQMPS2_BUILD_UNITTEST "Build unittests for GraceQ/mps2." OFF)

option(GQMPS2_MPI_PARALLEL "Build MPI support." OFF)
if(GQMPS2_MPI_PARALLEL)
  find_package(MPI REQUIRED)
  add_definitions(-DGQMPS2_MPI_PARALLEL)
  # For dependence setting.
  set(GQTEN_MPI_PARALLEL ON)
  add_definitions(-DGQTEN_MPI_PARALLEL)
endif()


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g)


# Build dependencies.
add_subdirectory(external)


# Build library.
add_subdirectory(src lib)


# Build unittests.
if(GQMPS2_BUILD_UNITTEST)
  enable_testing()
  add_subdirectory(tests tests)
endif()
