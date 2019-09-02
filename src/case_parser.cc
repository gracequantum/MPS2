// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-03 22:30
* 
* Description: GraceQ/mps2 project. Simulation case parser.
*/
#include "gqmps2/gqmps2.h"

#include <fstream>

#include "nlohmann/json.hpp"


namespace gqmps2 {
using json = nlohmann::json;


CaseParamsParserBasic::CaseParamsParserBasic(const char *file) {
  std::ifstream ifs(file);
  ifs >> raw_json_;
  ifs.close();
  if (raw_json_.find(kCaseParamsJsonObjName) != raw_json_.end()) {
    case_params = raw_json_[kCaseParamsJsonObjName];
  } else {
    std::cout << "CaseParams object not found, exit!" << std::endl;
    exit(1);
  }
}


int CaseParamsParserBasic::ParseInt(const std::string &item) {
  return case_params[item].get<int>();
}


double CaseParamsParserBasic::ParseDouble(const std::string &item) {
  return case_params[item].get<double>();
}


char CaseParamsParserBasic::ParseChar(const std::string &item) {
  auto char_str = case_params[item].get<std::string>();
  return char_str.at(0);
}


std::string CaseParamsParserBasic::ParseStr(const std::string &item) {
  return case_params[item].get<std::string>();
}


bool CaseParamsParserBasic::ParseBool(const std::string &item) {
  return case_params[item].get<bool>();
}
} /* gqmps2 */ 
