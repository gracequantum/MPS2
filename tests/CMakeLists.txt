#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-12 10:15
#  
#  Description: GraceQ/MPS2 project. CMake file to control unittest.
# 
if(NOT GQMPS2_BUILD_GQTEN_USE_EXTERNAL_HPTT_LIB)
  set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/gqten/external/hptt/libhptt.a")
endif()
# Set MKL compile flags and link flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
  set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
  # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
  set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
  set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
endif()


macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
  add_executable(${TEST_NAME}
      ${TEST_SRC})

    target_include_directories(${TEST_NAME}
      PRIVATE ${GQMPS2_HEADER_PATH}
      PRIVATE ${GQMPS2_TENSOR_LIB_HEADER_PATH}
      PRIVATE ${CFLAGS})
    target_link_libraries(${TEST_NAME}
      gqten
      GTest::GTest GTest::Main
      ${hptt_LIBRARY}
      ${LINK_LIBS} "${LINK_LIB_FLAGS}")

    add_test(
        NAME ${TEST_NAME}
        COMMAND "${TEST_NAME}" "${INPUT_ARGS}")

  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

# Test Lanczos algorithm.
add_unittest(test_lanczos test_lanczos.cc "" "" "${MATH_LIB_LINK_FLAGS}" "")

## Test MPO generation.
add_unittest(test_mpogen_coef_op_alg test_mpogen_coef_op_alg.cc "" "" "" "")
add_unittest(test_mpogen test_mpogen.cc "" "" "${MATH_LIB_LINK_FLAGS}" "")

# Test two site algorithm.
add_unittest(test_two_site_algo
  test_two_site_algo.cc "" "" "${MATH_LIB_LINK_FLAGS}" "")

# Test MPS measurement.
add_unittest(test_mps_measu
  test_mps_measu.cc "" "" "${MATH_LIB_LINK_FLAGS}" "")

# Test simulation case parameters parser.
add_unittest(test_case_params_parser
  test_case_params_parser.cc
  "" "" "${MATH_LIB_LINK_FLAGS}"
  "${CMAKE_CURRENT_LIST_DIR}/test_data/test-params.json" "")
