# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 14:20
#  
#  Description: GraceQ/mps2 project. Cmake file to control the library source code.
# 


if(GQMPS2_MPI_PARALLEL)
  add_library(gqmps2
    case_parser.cc
    mps_ops.cc
    mpogen.cc
    lanczos.cc
    two_site_algo.cc
    mpi_lanczos.cc
    mpi_two_site_algo.cc
    mps_measu.cc
    gqmps2.cc)

  target_include_directories(gqmps2
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/gqten/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/nlohmann_json/include"
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

  target_compile_options(gqten
    PRIVATE ${MPI_CXX_COMPILE_FLAGS})

  target_link_libraries(gqmps2
    ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
    gqten)
else()
  add_library(gqmps2
    case_parser.cc
    mps_ops.cc
    mpogen.cc
    lanczos.cc
    two_site_algo.cc
    mps_measu.cc
    gqmps2.cc)

  target_include_directories(gqmps2
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/gqten/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/nlohmann_json/include")

  target_link_libraries(gqmps2
    gqten)
endif()
