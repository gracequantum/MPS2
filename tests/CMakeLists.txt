# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-12 10:15
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
# 
macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
  add_executable(${TEST_NAME}
      ${TEST_SRC})
  target_include_directories(${TEST_NAME} PUBLIC
      "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/external/gqten/include"
      "/usr/local/include"
      ${CFLAGS})
  target_link_libraries(${TEST_NAME}
      "-L/usr/local/lib"
      gtest gtest_main
      ${LINK_LIBS}
      "${LINK_LIB_FLAGS}")
  add_test(NAME ${TEST_NAME} COMMAND "${TEST_NAME}" "${INPUT_ARGS}")
  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

# Test Lanczos algorithm.
add_unittest(test_lanczos test_lanczos.cc "" "gqten;gqmps2" "" "")

# Test MPO generation.
add_unittest(test_mpogen test_mpogen.cc "" "gqten;gqmps2" "" "")

# Test two site algorithm.
add_unittest(test_two_site_algo test_two_site_algo.cc "" "gqten;gqmps2" "" "")

# Test simulation case parameters parser.
add_unittest(test_case_params_parser
  test_case_params_parser.cc
  "" "gqmps2" ""
  "${CMAKE_CURRENT_LIST_DIR}/test_data/test-params.json")
