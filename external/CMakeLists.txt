# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-11 14:11
#  
#  Description: GraceQ/mps2 project. CMake file to control external dependencies.
# 


# GraceQ/tensor
add_subdirectory(gqten)

# nlohmann_json
set(JSON_BuildTests OFF CACHE INTERNAL "")
option(JSON_Install OFF)
add_subdirectory(nlohmann_json)
install(
  FILES ${PROJECT_SOURCE_DIR}/external/nlohmann_json/single_include/nlohmann/json.hpp
  DESTINATION ${GQMPS2_INCLUDE_DEST}/nlohmann)
