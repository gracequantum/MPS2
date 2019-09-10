#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 14:20
#  
#  Description: GraceQ/mps2 project. Cmake file to control the library source code.
# 
set(GQMPS2_HEASERS ${GQMPS2_HEADER_PATH}/gqmps2/gqmps2.h)
set(GQMPS2_BASIC_SRC
  case_parser.cc
  mps_ops.cc
  mpogen.cc
  lanczos.cc
  two_site_algo.cc
  mps_measu.cc
  )


if(GQMPS2_MPI_PARALLEL)
  add_library(gqmps2
    ${GQMPS2_BASIC_SRC}
    mps_measu.cc
    mpi_lanczos.cc
    mpi_two_site_algo.cc)

  target_include_directories(gqmps2
    PRIVATE ${GQMPS2_HEADER_PATH}
    PRIVATE ${GQMPS2_TENSOR_LIB_HEADER_PATH}
    PRIVATE ${GQMPS2_JSON_LIB_HEADER_PATH}
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

  target_compile_options(gqten
    PRIVATE ${MPI_CXX_COMPILE_FLAGS})
else()
  add_library(gqmps2
    ${GQMPS2_BASIC_SRC})

  target_include_directories(gqmps2
    PRIVATE ${GQMPS2_HEADER_PATH}
    PRIVATE ${GQMPS2_TENSOR_LIB_HEADER_PATH}
    PRIVATE ${GQMPS2_JSON_LIB_HEADER_PATH})
endif()


install(
  FILES ${GQMPS2_HEASERS}
  DESTINATION ${GQMPS2_INCLUDE_DEST})
install(
  TARGETS gqmps2
  DESTINATION "${GQMPS2_LIB_DEST}")
