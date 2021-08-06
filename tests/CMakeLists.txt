#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-12 10:15
#  
#  Description: GraceQ/MPS2 project. CMake file to control unittest.
# 
# Set MKL compile flags and link flags.

if(APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
    #Need test
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
  endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # May not work
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # Note as of Intel oneAPI 2021.2, "source /opt/intel/oneapi/compiler/latest/env/vars.sh"
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -Wl, -rpath $ENV{CMPLR_ROOT}/mac/compiler/lib/libiomp5.dylib -liomp5 -lpthread -lm -ldl)
  endif()
elseif(UNIX)
  #UNIX include APPLE, but we except it here
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
  endif()
endif()


macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
  add_executable(${TEST_NAME}
      ${TEST_SRC})

    target_compile_options(${TEST_NAME}
      PRIVATE ${CFLAGS}
    )

    target_include_directories(${TEST_NAME}
      PRIVATE ${GQMPS2_HEADER_PATH}
      PRIVATE ${GQMPS2_TENSOR_LIB_HEADER_PATH}
      PRIVATE ${hptt_INCLUDE_DIR})
    target_link_libraries(${TEST_NAME}
      GTest::GTest GTest::Main
      ${hptt_LIBRARY}
      ${LINK_LIBS} "${LINK_LIB_FLAGS}")

    add_test(
        NAME ${TEST_NAME}
        COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


## Test sites vector.
add_unittest(test_site_vec test_site_vec.cc "${MATH_LIB_COMPILE_FLAGS}" "" "" "")

## Test one-dimensional tensor networks.
# Test DuoVector class
add_unittest(test_duovector test_one_dim_tn/test_duovector.cc "" "" "" "")
# Test TenVec class
add_unittest(test_ten_vec
  "test_one_dim_tn/test_ten_vec.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPS class
add_unittest(test_finite_mps
  "test_one_dim_tn/test_finite_mps/test_finite_mps.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test MPS measurement.
add_unittest(test_finite_mps_measu
  "test_one_dim_tn/test_finite_mps/test_finite_mps_measu.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPO class
add_unittest(test_mpo
    "test_one_dim_tn/test_mpo.cc"
    "${MATH_LIB_COMPILE_FLAGS}" "" "" ""
)
# Test MPO generation.
add_unittest(test_mpogen_coef_op_alg
    "test_one_dim_tn/test_mpogen/test_mpogen_coef_op_alg.cc"
    "" "" "" ""
)
add_unittest(test_mpogen_fsm
    "test_one_dim_tn/test_mpogen/test_mpogen_fsm.cc"
    "" "" "" ""
)
add_unittest(test_mpogen
  "test_one_dim_tn/test_mpogen/test_mpogen.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test algorithms
# Lanczos solver
add_unittest(test_lanczos_solver
  "test_algorithm/test_lanczos_solver.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Two-site update finite vMPS
add_unittest(test_two_site_algo
  "test_algorithm/test_two_site_update_finite_vmps.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Single-site update finite vMPS
add_unittest(test_single_site_algo
  "test_algorithm/test_single_site_update_finite_vmps.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

## Test simulation case parameters parser.
add_unittest(test_case_params_parser
  test_case_params_parser.cc
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
  "${CMAKE_CURRENT_LIST_DIR}/testdata/test-params.json" "")
