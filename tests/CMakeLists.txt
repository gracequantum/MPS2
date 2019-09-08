#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-12 10:15
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
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
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS MPI_TEST)
  add_executable(${TEST_NAME}
      ${TEST_SRC})

  if(GQMPS2_MPI_PARALLEL)
    target_include_directories(${TEST_NAME}
      PUBLIC "${PROJECT_SOURCE_DIR}/include"
      PUBLIC "${PROJECT_SOURCE_DIR}/external/gqten/include"
      PUBLIC "/usr/local/include"
      PRIVATE ${MPI_CXX_INCLUDE_PATH}
      PRIVATE ${CFLAGS})
    target_compile_options(${TEST_NAME}
      PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(${TEST_NAME}
        "-L/usr/local/lib"
        gtest gtest_main
        ${hptt_LIBRARY}
        ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
        ${LINK_LIBS} "${LINK_LIB_FLAGS}")
  else()
    target_include_directories(${TEST_NAME}
      PUBLIC "${PROJECT_SOURCE_DIR}/include"
      PUBLIC "${PROJECT_SOURCE_DIR}/external/gqten/include"
      PUBLIC "/usr/local/include"
      PRIVATE ${CFLAGS})
    target_link_libraries(${TEST_NAME}
        "-L/usr/local/lib"
        gtest gtest_main
        ${hptt_LIBRARY}
        ${LINK_LIBS} "${LINK_LIB_FLAGS}")
  endif()

  if("${MPI_TEST}" STREQUAL "")
    add_test(
        NAME ${TEST_NAME}
        COMMAND "${TEST_NAME}" "${INPUT_ARGS}")
  else()
    set(GEMM_BATCH_WORKER ../external/gqten/bin/gqten_mpi_gemm_batch_worker)
    if(APPLE)
      add_test(${TEST_NAME}
        mpirun --host localhost:4 -np 1 ${TEST_NAME} : --host localhost:4 -np 3 ${GEMM_BATCH_WORKER})
    else()
      add_test(${TEST_NAME}
          mpirun -np 1 ./${TEST_NAME} : -np 3 ${GEMM_BATCH_WORKER})
    endif()
  endif()

  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

# Test Lanczos algorithm.
add_unittest(test_lanczos test_lanczos.cc "" "gqten;gqmps2" "${MATH_LIB_LINK_FLAGS}" "" "")

# Test distributed Lanczos algorithm.
if(GQMPS2_MPI_PARALLEL)
  add_unittest(test_mpi_lanczos
    test_mpi_lanczos.cc "" "gqten;gqmps2" "${MATH_LIB_LINK_FLAGS}" "" "MPI_TEST")
endif()

# Test MPO generation.
add_unittest(test_mpogen test_mpogen.cc "" "gqten;gqmps2" "${MATH_LIB_LINK_FLAGS}" "" "")

# Test two site algorithm.
add_unittest(test_two_site_algo
  test_two_site_algo.cc "" "gqten;gqmps2" "${MATH_LIB_LINK_FLAGS}" "" "")

# Test distributed two site algorithm.
if(GQMPS2_MPI_PARALLEL)
  add_unittest(test_mpi_two_site_algo
    test_mpi_two_site_algo.cc "" "gqten;gqmps2" "${MATH_LIB_LINK_FLAGS}" "" "MPI_TEST")
endif()

# Test simulation case parameters parser.
add_unittest(test_case_params_parser
  test_case_params_parser.cc
  "" "gqmps2" "${MATH_LIB_LINK_FLAGS}"
  "${CMAKE_CURRENT_LIST_DIR}/test_data/test-params.json" "")
