/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 12:15
* 
* Description: GraceQ/mps2 project. Private objects for two sites algorithm.
*/
#ifndef GQMPS2_TWO_SITE_ALGO_H
#define GQMPS2_TWO_SITE_ALGO_H


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <vector>


namespace gqmps2 {
using namespace gqten;


std::pair<std::vector<GQTensor *>, std::vector<GQTensor *>> InitBlocks(
    const std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteSweep(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteUpdate(
    const long,
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &, const char);
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_TWO_SITE_ALGO_H */
