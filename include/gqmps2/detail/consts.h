// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-28 15:45
* 
* Description: GraceQ/MPS2 project. Constant declarations.
*/

/**
@file consts.h
@brief Constant declarations.
*/
#ifndef GQMPS2_DETAIL_CONSTS_H
#define GQMPS2_DETAIL_CONSTS_H


#include <string>     // string
#include <vector>     // vector


namespace gqmps2 {


/// JSON object name of the simulation case parameter parsed by @link gqmps2::CaseParamsParserBasic `CaseParamsParser` @endlink.
const std::string kCaseParamsJsonObjName = "CaseParams";

const std::string kMpsPath = "mps";
const std::string kRuntimeTempPath = ".temp";
const std::string kBlockFileBaseName = "block";
const std::string kMpsTenBaseName = "mps_ten";

const char kTwoSiteAlgoWorkflowInitial = 'i';
const char kTwoSiteAlgoWorkflowRestart = 'r';
const char kTwoSiteAlgoWorkflowContinue = 'c';

const int kLanczEnergyOutputPrecision = 16;

const std::vector<std::vector<size_t>> kNullIntVecVec;
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_DETAIL_CONSTS_H */
