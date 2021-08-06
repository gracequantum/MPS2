#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 13:23
#  
#  Description: GraceQ/mps2 project. Main CMake file.
# 
cmake_minimum_required(VERSION 3.12)


project("GraceQ/mps2"
  DESCRIPTION "A high-performance matrix product state(MPS) algorithms library based on GraceQ/tensor"
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

option(TEST_BUILD_USE_EXTERNAL_GQTEN "Use external gqten library when building unittests for GraceQ/mps2." OFF)

option(TEST_BUILD_USE_EXTERNAL_HPTT_LIB "Use external hptt library when building unittests for GraceQ/mps2." OFF)

# Compilation and linking control.
set(CMAKE_CXX_STANDARD 14)
add_definitions(-g)


# Installation path
set(GQMPS2_TOOL_SCRIPT_DEST "${CMAKE_INSTALL_PREFIX}/bin")
set(GQMPS2_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")


# Build(Install) library.
set(GQMPS2_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(GQMPS2_HEADER_DIR ${GQMPS2_HEADER_PATH}/gqmps2)
install(
  DIRECTORY ${GQMPS2_HEADER_DIR}
  DESTINATION ${GQMPS2_INCLUDE_DEST})


# Build tool scripts
add_subdirectory(script)


# Build unittests.
if(GQMPS2_BUILD_UNITTEST)
  if(NOT TEST_BUILD_USE_EXTERNAL_GQTEN)
    add_subdirectory(external)
    set(GQMPS2_TENSOR_LIB_HEADER_PATH "${PROJECT_SOURCE_DIR}/external/gqten/include")
  else()
    find_path(GQMPS2_TENSOR_LIB_HEADER_PATH "gqten")
  endif()

  if(NOT TEST_BUILD_USE_EXTERNAL_HPTT_LIB)
    set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/gqten/external/hptt/include")
    set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/gqten/external/hptt/libhptt.a")
  else()
    find_path(hptt_INCLUDE_DIR "hptt.h")
    find_library(hptt_LIBRARY "libhptt.a")
  endif()

  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests tests)
endif()
