#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-09-10 13:01
#  
#  Description: GraceQ/MPS2 project. CMake file to control tool scripts.
#
set(GQMPS2_TOOL_SCRIPT_PATH "${PROJECT_SOURCE_DIR}/script")
set(GQMPS2_TOOL_SCRIPTS
  ${GQMPS2_TOOL_SCRIPT_PATH}/gqmps2_timing_analyser.py
  ${GQMPS2_TOOL_SCRIPT_PATH}/gqmps2_log_simplifier.sh)


install(
  FILES ${GQMPS2_TOOL_SCRIPTS}
  DESTINATION "${GQMPS2_TOOL_SCRIPT_DEST}")
