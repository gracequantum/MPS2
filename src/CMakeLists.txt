# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 14:20
#  
#  Description: GraceQ/mps2 project. Cmake file to control the library source code.
# 


add_library(gqmps2
  case_parser.cc
  mps_ops.cc
  mpogen.cc
  lanczos.cc
  two_site_algo.cc
  gqmps2.cc)

target_include_directories(gqmps2
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PUBLIC "${PROJECT_SOURCE_DIR}/external/gqten/include"
  PUBLIC "${PROJECT_SOURCE_DIR}/external/nlohmann_json/include")

target_link_libraries(gqmps2
  gqten)
