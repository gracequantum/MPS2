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
#include <cmath>


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


inline double MeasureEE(const GQTensor *s, const long sdim) {
  double ee = 0;
  double p;
  for (long i = 0; i < sdim; ++i) {
    p = std::pow(s->Elem({i, i}), 2.0);
    ee += -p * std::log(p);
  }
  return ee;
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_TWO_SITE_ALGO_H */
