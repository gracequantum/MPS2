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


CaseParserBasic::CaseParserBasic(const char *file) {
  std::ifstream ifs(file);
  ifs >> raw_json_;
  ifs.close();
  case_params = raw_json_[kCaseParamsJsonObjName];
}


int CaseParserBasic::ParseInt(const std::string &item) {
  return case_params[item].get<int>();
}


double CaseParserBasic::ParseDouble(const std::string &item) {
  return case_params[item].get<double>();
}


char CaseParserBasic::ParseChar(const std::string &item) {
  return case_params[item].get<char>();
}


std::string CaseParserBasic::ParseStr(const std::string &item) {
  return case_params[item].get<std::string>();
}
} /* gqmps2 */ 
